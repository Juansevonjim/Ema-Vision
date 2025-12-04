#!/usr/bin/env python3

"""
YOLO11 + Reconocimiento Facial - OPTIMIZADO SIN FILTROS DE CALIDAD
- Procesa TODAS las caras detectadas
- No rechaza por calidad
- Desconocido cuando no hay match
"""

import os
import sys
import cv2
import torch
import numpy as np
import time
from threading import Thread
from queue import Queue, Empty
from collections import deque, defaultdict
from facenet_pytorch import MTCNN, InceptionResnetV1
import gc
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
DEFAULT_MODEL = "yolo11m.pt"
CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# YOLO
USE_FP16 = True
USE_CUDA = torch.cuda.is_available()
YOLO_CONF_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.45
YOLO_MAX_DET = 100

# Face Recognition - MÁS PERMISIVO
KNOWN_FACES_DIR = "known_people"
FACE_SCALE = 0.6
MIN_FACE_SIZE = 20  # Mínimo tamaño
FACE_CONFIDENCE_THRESHOLD = 0.85  # Reducido aún más

# Umbrales de similitud
SIM_THRESHOLD = 0.55  # Threshold único más bajo
SIM_MARGIN = 0.08

# Performance
QUEUE_MAXSIZE = 2
FACE_PROCESS_INTERVAL = 2
FACE_TRACKING_FRAMES = 8
MEMORY_CLEANUP_INTERVAL = 100

# Debug
DEBUG = True
SHOW_FPS = True
SHOW_GPU_USAGE = True

# ==================== SETUP ====================

def setup_device():
    if not USE_CUDA:
        print("[INFO] CUDA no disponible, usando CPU")
        return torch.device('cpu'), False
        
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    if USE_FP16:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[INFO] FP16 inference habilitado")
    
    props = torch.cuda.get_device_properties(0)
    print(f"[INFO] GPU: {props.name}")
    print(f"[INFO] Memoria: {props.total_memory / 1024**3:.1f} GB")
    
    return device, True

device, use_fp16 = setup_device()

# ==================== MODELOS ====================

print("[INFO] Cargando modelos...")

mtcnn = MTCNN(
    keep_all=True,
    device=device,
    min_face_size=MIN_FACE_SIZE,
    thresholds=[0.5, 0.6, 0.6],
    post_process=True,
    select_largest=False
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

for param in resnet.parameters():
    param.requires_grad = False

print("[INFO] ✓ Modelos cargados")

# ==================== UTILIDADES ====================

class PerformanceMonitor:
    def __init__(self):
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        self.last_cleanup = 0
        
    def update(self, fps):
        self.fps_history.append(fps)
        self.frame_count += 1
        
    def get_avg_fps(self):
        return np.mean(self.fps_history) if self.fps_history else 0
        
    def should_cleanup(self):
        if self.frame_count - self.last_cleanup > MEMORY_CLEANUP_INTERVAL:
            self.last_cleanup = self.frame_count
            return True
        return False

class FaceTracker:
    def __init__(self, max_age=FACE_TRACKING_FRAMES):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.frame_count = 0
        
    def update(self, boxes, names):
        self.frame_count += 1
        current_boxes = []
        current_names = []
        
        for box, name in zip(boxes, names):
            matched = False
            x1, y1, x2, y2 = box
            center = ((x1+x2)/2, (y1+y2)/2)
            
            for track_id, track in list(self.tracks.items()):
                if self.frame_count - track['last_seen'] > self.max_age:
                    del self.tracks[track_id]
                    continue
                    
                last_box = track['boxes'][-1]
                last_center = ((last_box[0]+last_box[2])/2, (last_box[1]+last_box[3])/2)
                dist = np.sqrt((center[0]-last_center[0])**2 + (center[1]-last_center[1])**2)
                
                if dist < 80:
                    track['boxes'].append(box)
                    track['names'].append(name)
                    track['last_seen'] = self.frame_count
                    
                    names_count = defaultdict(int)
                    for n in track['names']:
                        names_count[n] += 1
                    stable_name = max(names_count.items(), key=lambda x: x[1])[0]
                    
                    current_boxes.append(box)
                    current_names.append(stable_name)
                    matched = True
                    break
                    
            if not matched:
                self.tracks[self.next_id] = {
                    'boxes': deque([box], maxlen=self.max_age),
                    'names': deque([name], maxlen=self.max_age),
                    'last_seen': self.frame_count
                }
                self.next_id += 1
                current_boxes.append(box)
                current_names.append(name)
                
        return current_boxes, current_names

# ==================== FACE PROCESSING ====================

@torch.no_grad()
def extract_embeddings_batch(face_tensors):
    """Extrae embeddings sin filtros"""
    if not face_tensors:
        return []
        
    try:
        batch = torch.stack([f.to(device).float() for f in face_tensors])
        embeddings = resnet(batch).cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    except Exception as e:
        print(f"[ERROR] Batch embedding: {e}")
        return []

def load_known_embeddings(folder):
    """Carga embeddings SIN filtros de calidad"""
    if not os.path.exists(folder):
        print(f"[INFO] Creando carpeta {folder}...")
        os.makedirs(folder, exist_ok=True)
        return {}
        
    known = {}
    subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    
    if not subdirs:
        print(f"[INFO] No hay personas en {folder}. Sistema funcionará en modo 'Desconocido'.")
        return {}
        
    print(f"[INFO] Cargando personas desde {folder}...")
    
    for person in sorted(subdirs):
        person_folder = os.path.join(folder, person)
        person_embs = []
        
        files = [f for f in os.listdir(person_folder) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not files:
            print(f"[WARN] {person}: sin imágenes")
            continue
            
        print(f"  Procesando {person}...", end=" ")
        
        for fn in files:
            path = os.path.join(person_folder, fn)
            img = cv2.imread(path)
            
            if img is None:
                continue
                
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detectar y extraer caras
            try:
                boxes, probs = mtcnn.detect(rgb)
                
                if boxes is None:
                    continue
                
                # Procesar TODAS las caras detectadas
                for box, prob in zip(boxes, probs):
                    # Solo filtro mínimo de confianza
                    if prob < FACE_CONFIDENCE_THRESHOLD:
                        continue
                    
                    # Extraer tensor de cara alineada
                    face_tensor = mtcnn.extract(rgb, [box], save_path=None)
                    
                    if face_tensor is None or len(face_tensor) == 0:
                        continue
                    
                    # Extraer embedding
                    emb = extract_embeddings_batch([face_tensor[0]])
                    if len(emb) > 0:
                        person_embs.append(emb[0])
                        
            except Exception as e:
                continue
                    
        if person_embs:
            known[person] = np.vstack(person_embs)
            print(f"✓ {len(person_embs)} embeddings cargados")
        else:
            print(f"✗ No se cargaron embeddings")
            
    if not known:
        print("[INFO] No hay personas conocidas. Sistema funcionará en modo 'Desconocido'.")
        print("[INFO] Agrega carpetas en 'known_people/NombrePersona/' con fotos")
    
    return known

def recognize_embedding(emb, known_embeddings):
    """Reconocimiento simple sin filtros extras"""
    if not known_embeddings:
        return "Desconocido", 0.0, []
        
    best_person = "Desconocido"
    best_sim = -1.0
    second_best = -1.0
    scores = []
    
    for person, arr in known_embeddings.items():
        sims = arr @ emb
        max_sim = float(np.max(sims))
        scores.append((person, max_sim))
        
        if max_sim > best_sim:
            second_best = best_sim
            best_sim = max_sim
            best_person = person
        elif max_sim > second_best:
            second_best = max_sim
    
    # Solo verificar threshold y margen
    if best_sim >= SIM_THRESHOLD and (best_sim - second_best) >= SIM_MARGIN:
        name = best_person
    else:
        name = "Desconocido"
        
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    return name, best_sim, scores_sorted

# ==================== YOLO THREAD ====================

def yolo_inference_thread(model, frames_q, results_q, stop_flag):
    while not stop_flag["stop"]:
        try:
            frame = frames_q.get(timeout=0.5)
        except Empty:
            continue
            
        try:
            results = model.predict(
                frame,
                conf=YOLO_CONF_THRESHOLD,
                iou=YOLO_IOU_THRESHOLD,
                max_det=YOLO_MAX_DET,
                half=use_fp16 and USE_FP16,
                device=device if USE_CUDA else 'cpu',
                verbose=False
            )
            
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else np.array([])
            confs = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else np.array([])
            cls_ids = result.boxes.cls.cpu().numpy() if len(result.boxes) > 0 else np.array([])
            names = model.names
            
            results_q.put((frame, boxes, confs, cls_ids, names))
            
        except Exception as e:
            print(f"[ERROR YOLO]: {e}")
            results_q.put((frame, np.array([]), np.array([]), np.array([]), {}))

# ==================== FACE THREAD ====================

def face_inference_thread(frames_q, results_q, stop_flag, known_embeddings, face_tracker):
    frame_count = 0
    last_result = []
    
    while not stop_flag["stop"]:
        try:
            frame = frames_q.get(timeout=0.5)
        except Empty:
            continue
            
        frame_count += 1
        
        if frame_count % FACE_PROCESS_INTERVAL != 0:
            results_q.put(last_result)
            continue
            
        try:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (int(w * FACE_SCALE), int(h * FACE_SCALE)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            boxes, probs = mtcnn.detect(rgb)
            
            detected_boxes = []
            detected_names = []
            detected_sims = []
            detected_top3 = []
            
            if boxes is not None and len(boxes) > 0:
                scale_factor = 1.0 / FACE_SCALE
                boxes_scaled = boxes * scale_factor
                
                face_tensors = []
                valid_indices = []
                
                # Preparar batch - TODAS las caras con confianza mínima
                for idx, (box, prob) in enumerate(zip(boxes_scaled, probs)):
                    if prob < FACE_CONFIDENCE_THRESHOLD:
                        continue
                    
                    try:
                        # Extraer cara alineada
                        face_tensor = mtcnn.extract(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            [box],
                            save_path=None
                        )
                        
                        if face_tensor is not None and len(face_tensor) > 0:
                            face_tensors.append(face_tensor[0])
                            valid_indices.append(idx)
                    except:
                        continue
                        
                # Batch embedding extraction
                if face_tensors:
                    embeddings = extract_embeddings_batch(face_tensors)
                    
                    for idx, emb in zip(valid_indices, embeddings):
                        box = boxes_scaled[idx]
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Reconocimiento
                        name, sim, top3 = recognize_embedding(emb, known_embeddings)
                        
                        detected_boxes.append((x1, y1, x2, y2))
                        detected_names.append(name)
                        detected_sims.append(sim)
                        detected_top3.append(top3)
                        
            # Tracking temporal
            if detected_boxes:
                stable_boxes, stable_names = face_tracker.update(detected_boxes, detected_names)
                last_result = list(zip(stable_boxes, stable_names, detected_sims, detected_top3))
            else:
                last_result = []
                
            results_q.put(last_result)
            
        except Exception as e:
            print(f"[ERROR FACE]: {e}")
            results_q.put([])

# ==================== DISPLAY ====================

def cleanup_memory():
    gc.collect()
    if USE_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def draw_detections(frame, boxes, confs, cls_ids, names):
    for box, conf, cls in zip(boxes, confs, cls_ids):
        if conf < YOLO_CONF_THRESHOLD:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        label = f"{names.get(int(cls), str(int(cls)))} {conf:.2f}"
        
        if conf > 0.7:
            color = (0, 255, 0)
        elif conf > 0.5:
            color = (0, 200, 200)
        else:
            color = (0, 150, 255)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 0, 0), 1, cv2.LINE_AA)
                   
    return frame

def draw_faces(frame, face_results):
    for (x1, y1, x2, y2), name, sim, top3 in face_results:
        if name != "Desconocido":
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{name} ({sim:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        # Mostrar top-3 en debug
        if DEBUG and top3:
            y_offset = y2 + 20
            for person, score in top3[:3]:
                text = f"{person}:{score:.2f}"
                cv2.putText(frame, text, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
                y_offset += 15
                    
    return frame

def capture_frames(cap, q_yolo, q_face, stop_flag):
    while not stop_flag["stop"]:
        ret, frame = cap.read()
        if not ret:
            stop_flag["stop"] = True
            break
            
        for q in [q_yolo, q_face]:
            try:
                q.put_nowait(frame.copy())
            except:
                try:
                    _ = q.get_nowait()
                    q.put_nowait(frame.copy())
                except:
                    pass

# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("SISTEMA DE VISIÓN COMPUTACIONAL OPTIMIZADO")
    print("=" * 60)
    
    # Cargar embeddings (puede estar vacío)
    known_embeddings = load_known_embeddings(KNOWN_FACES_DIR)
    
    if known_embeddings:
        print(f"\n[INFO] ✓ Sistema listo con {len(known_embeddings)} personas conocidas")
    else:
        print("\n[INFO] ✓ Sistema listo en modo 'Desconocido'")
    
    # Cargar YOLO
    try:
        from ultralytics import YOLO
        print(f"\n[INFO] Cargando {DEFAULT_MODEL}...")
        model = YOLO(DEFAULT_MODEL)
        
        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict(dummy, verbose=False, half=False)
        print("[INFO] ✓ YOLO listo")
        
    except Exception as e:
        print(f"[ERROR] No se pudo cargar YOLO: {e}")
        return
        
    # Cámara
    print(f"\n[INFO] Abriendo cámara {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Queues
    frames_q_yolo = Queue(maxsize=QUEUE_MAXSIZE)
    frames_q_face = Queue(maxsize=QUEUE_MAXSIZE)
    results_q_yolo = Queue(maxsize=QUEUE_MAXSIZE)
    results_q_face = Queue(maxsize=QUEUE_MAXSIZE)
    
    stop_flag = {"stop": False}
    perf_monitor = PerformanceMonitor()
    face_tracker = FaceTracker()
    
    # Threads
    print("\n[INFO] Iniciando procesamiento...")
    
    Thread(target=capture_frames, 
           args=(cap, frames_q_yolo, frames_q_face, stop_flag), 
           daemon=True).start()
           
    Thread(target=yolo_inference_thread,
           args=(model, frames_q_yolo, results_q_yolo, stop_flag),
           daemon=True).start()
           
    Thread(target=face_inference_thread,
           args=(frames_q_face, results_q_face, stop_flag, known_embeddings, face_tracker),
           daemon=True).start()
    
    print("\n[INFO] ✓ Sistema iniciado - Presiona 'q' para salir\n")
    
    # Main loop
    t_prev = time.time()
    fps_smooth = 30.0
    yolo_result = None
    face_result = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            display_frame = frame.copy()
            
            # YOLO results
            try:
                yolo_result = results_q_yolo.get_nowait()
            except Empty:
                pass
                
            if yolo_result is not None:
                _, boxes, confs, cls_ids, names = yolo_result
                display_frame = draw_detections(display_frame, boxes, confs, cls_ids, names)
                
            # Face results
            try:
                face_result = results_q_face.get_nowait()
            except Empty:
                pass
                
            if face_result:
                display_frame = draw_faces(display_frame, face_result)
                
            # FPS
            t_now = time.time()
            fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            fps_smooth = fps_smooth * 0.9 + fps * 0.1
            perf_monitor.update(fps_smooth)
            
            if SHOW_FPS:
                cv2.putText(display_frame, f"FPS: {fps_smooth:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 255, 255), 2, cv2.LINE_AA)
                           
            if SHOW_GPU_USAGE and USE_CUDA:
                mem_used = torch.cuda.memory_allocated() / 1024**2
                mem_cached = torch.cuda.memory_reserved() / 1024**2
                cv2.putText(display_frame, f"GPU: {mem_used:.0f}MB / {mem_cached:.0f}MB", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 255), 1, cv2.LINE_AA)
                           
            status = f"Objetos: {len(boxes) if yolo_result else 0} | Caras: {len(face_result)}"
            cv2.putText(display_frame, status, 
                       (10, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow("Vision Sistema Optimizado", display_frame)
            
            if perf_monitor.should_cleanup():
                cleanup_memory()
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción del usuario")
        
    finally:
        print("\n[INFO] Cerrando sistema...")
        stop_flag["stop"] = True
        cap.release()
        cv2.destroyAllWindows()
        cleanup_memory()
        
        avg_fps = perf_monitor.get_avg_fps()
        print(f"[INFO] ✓ FPS promedio: {avg_fps:.1f}")
        print("[INFO] Sistema cerrado correctamente")

if __name__ == "__main__":
    main()
