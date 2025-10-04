#!/usr/bin/env python3
"""
YOLO11 + Reconocimiento Facial (mejorado)
- Usa MTCNN alineado + InceptionResnetV1 (facenet-pytorch)
- Calcula embeddings normalizados y compara con cosine similarity
- Requiere umbral y margen entre top1/top2 para aceptar una etiqueta
- Muestra top-3 similitudes en modo debug para tuning
"""

import os
import sys
import cv2
import torch
import numpy as np
import time
from threading import Thread
from queue import Queue, Empty
from facenet_pytorch import MTCNN, InceptionResnetV1

# ---------------- Configuración ----------------
DEFAULT_MODEL = "yolo11s.pt"     # YOLO (se descarga la primera vez)
KNOWN_FACES_DIR = "known_people"  # estructura: known_people/PersonaA/*.jpg, PersonaB/*.jpg
CAM_INDEX = 0
FRAME_SCALE = 0.5                 # escala para detección facial (mejor rendimiento)
DEFAULT_CONF = 0.25
QUEUE_MAXSIZE = 4
FRAME_SKIP = 2                    # procesar cada N frames para la parte facial
SIM_THRESHOLD = 0.60              # similitud min (coseno) para aceptar etiqueta (ajustable)
SIM_MARGIN = 0.05                 # diferencia mínima entre top1 y top2 para evitar ambigüedad
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DEBUG = True                      # si True muestra top-3 similitudes en pantalla
# ------------------------------------------------

# --------------------------------
# Inicialización modelos
# --------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device (general): {device}")

# mtcnn produce caras alineadas (o tensors) listas para la red
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --------------------------------
# Utilidades de cara y embeddings
# --------------------------------
def tensorize_faces_from_image(img_rgb):
    """
    Intenta obtener tensores alineados desde MTCNN.
    Devuelve lista de tensors (N,3,160,160) en device.
    """
    with torch.no_grad():
        faces = mtcnn(img_rgb)  # puede devolver None, Tensor (1x3x160x160) o tensor Nx3x160x160
    if faces is None:
        return []
    # faces puede ser Tensor (3,160,160) o (N,3,160,160)
    if torch.is_tensor(faces):
        if faces.ndim == 3:
            faces = faces.unsqueeze(0)
        # asegurar float y device
        faces = faces.to(device).float()
        return [f for f in faces]
    # si por alguna razón faces es lista
    return [f.to(device).float() for f in faces]

def embedding_from_face_tensor(face_tensor):
    """Toma tensor (3,160,160) o (1,3,160,160) y devuelve embedding L2-normalizado (np.array 512)"""
    with torch.no_grad():
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)
        emb = resnet(face_tensor.to(device)).detach().cpu().numpy()[0]
    # L2 normalize
    emb = emb / np.linalg.norm(emb)
    return emb

def load_known_embeddings(folder):
    """
    Carga todas las imágenes por persona y construye una dict:
        known_embeddings[person] = np.array shape (n,512) (L2 normalized)
    """
    known = {}
    for person in sorted(os.listdir(folder)):
        person_folder = os.path.join(folder, person)
        if not os.path.isdir(person_folder):
            continue
        person_embs = []
        for fn in sorted(os.listdir(person_folder)):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            p = os.path.join(person_folder, fn)
            img = cv2.imread(p)
            if img is None:
                print(f"[WARN] No pude leer {p}")
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_tensors = tensorize_faces_from_image(rgb)
            if not face_tensors:
                # intentar detect + crop manual si mtcnn falla (fallback)
                print(f"[WARN] No detectó cara en {p} con MTCNN")
                continue
            # tomar todas las caras detectadas en la imagen (normalmente 1)
            for ft in face_tensors:
                emb = embedding_from_face_tensor(ft)
                person_embs.append(emb)
        if person_embs:
            known[person] = np.vstack(person_embs)
            print(f"[INFO] {person}: cargadas {len(person_embs)} embeddings")
        else:
            print(f"[WARN] {person}: no se cargaron embeddings (revisa imágenes)")
    return known

def recognize_embedding(emb, known_embeddings):
    """
    emb: np.array (512,) L2-normalized
    known_embeddings: dict person -> np.array (n,512)
    Devuelve (best_person, best_sim, second_best_sim, all_top3 list of (person,sim))
    """
    best_person = "Desconocido"
    best_sim = -1.0
    second_best = -1.0
    scores = []
    for person, arr in known_embeddings.items():
        # arr shape (n,512), ya normalizados
        sims = (arr @ emb)  # vector de similitudes (cosine) ya que todo normalizado
        max_sim = float(np.max(sims))
        scores.append((person, max_sim))
        if max_sim > best_sim:
            second_best = best_sim
            best_sim = max_sim
            best_person = person
        elif max_sim > second_best:
            second_best = max_sim
    # ordenar top3 para debug
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    return best_person, best_sim, second_best, scores_sorted

# --------------------------------
# YOLO thread (igual que antes)
# --------------------------------
def infer_results_to_numpy(r):
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
    return np.array([]), np.array([]), np.array([])

def draw_boxes(frame, boxes, confs, cls_ids, names, conf_thres):
    for box, conf, cls in zip(boxes, confs, cls_ids):
        if conf < conf_thres: continue
        x1, y1, x2, y2 = map(int, box)
        label = f"{names.get(int(cls), str(int(cls)))} {conf:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw, y1), (0,200,0), -1)
        cv2.putText(frame, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return frame

def yolo_inference_thread(model, frames_q, results_q, stop_flag, conf_thres, device):
    while not stop_flag["stop"]:
        try:
            frame = frames_q.get(timeout=0.5)
        except Empty:
            continue
        try:
            res = model(frame, conf=conf_thres, device=device) if device else model(frame, conf=conf_thres)
            r = res[0]
            boxes, confs, cls_ids = infer_results_to_numpy(r)
            names = model.model.names if hasattr(model, "model") else getattr(model, "names", {})
            results_q.put((frame, boxes, confs, cls_ids, names))
        except Exception as e:
            # en caso de fallo devolvemos vacío
            results_q.put((frame, np.array([]), np.array([]), np.array([]), {}))

# --------------------------------
# Face thread (usa MTCNN preprocesado + reconocimiento)
# --------------------------------
def face_inference_thread(frames_q, results_q, stop_flag, known_embeddings):
    frame_count = 0
    last_faces = []
    while not stop_flag["stop"]:
        try:
            frame = frames_q.get(timeout=0.5)
        except Empty:
            continue
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            results_q.put(last_faces)
            continue

        small = cv2.resize(frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Detectamos todas las caras y obtenemos sus boxes
        boxes, _ = mtcnn.detect(rgb)  # boxes en escala reducida
        detected = []

        if boxes is not None:
            for box in boxes:
                try:
                    # Convertimos coords a enteros en la escala original
                    x1, y1, x2, y2 = map(int, box / FRAME_SCALE)

                    # Crop cara de la imagen original (no la reducida para más calidad)
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue

                    # Preprocesar la cara con MTCNN para alinear
                    face_tensor = mtcnn.extract(rgb, [box], save_path=None)
                    if face_tensor is None:
                        continue

                    emb = embedding_from_face_tensor(face_tensor[0])  # np.array normalizado

                    # reconocimiento por similitud coseno
                    best_person, best_sim, second_best, top3 = recognize_embedding(emb, known_embeddings)

                    # aplicar checks: umbral y margen (top1-top2)
                    if best_sim >= SIM_THRESHOLD and (best_sim - second_best) >= SIM_MARGIN:
                        name = best_person
                    else:
                        name = "Desconocido"

                    detected.append((x1, y1, x2, y2, name, best_sim, top3))

                except Exception as e:
                    print(f"[ERROR FACE]: {e}")
                    continue

        last_faces = detected
        results_q.put(detected)


# --------------------------------
# Main unificado
# --------------------------------
def main():
    # Carga known embeddings
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"[ERROR] Crea la carpeta {KNOWN_FACES_DIR} con subcarpetas por persona y fotos.")
        return
    print("[INFO] Cargando imágenes y calculando embeddings (puede tardar)...")
    known_embeddings = load_known_embeddings(KNOWN_FACES_DIR)
    if not known_embeddings:
        print("[ERROR] No se cargaron embeddings. Revisa tus imágenes en known_people/")
        return
    # Diagnóstico entre-personas (opcional)
    if DEBUG:
        print("[DEBUG] Matriz de similitud máxima entre personas (top cross-match):")
        persons = list(known_embeddings.keys())
        for i,p1 in enumerate(persons):
            for j,p2 in enumerate(persons):
                if i>=j: continue
                max_sim = (known_embeddings[p1] @ known_embeddings[p2].T).max()
                print(f"  {p1} vs {p2} => max_sim = {max_sim:.3f}")

    # YOLO
    try:
        from ultralytics import YOLO
    except Exception:
        print("ERROR: Instala ultralytics: pip install -U ultralytics")
        sys.exit(1)
    device_yolo = "cuda:0" if torch.cuda.is_available() else None
    print(f"[INFO] Device YOLO: {device_yolo or 'auto/CPU'}")
    model = YOLO(DEFAULT_MODEL)

    # Cámara y colas
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frames_q_yolo = Queue(maxsize=QUEUE_MAXSIZE)
    frames_q_face = Queue(maxsize=QUEUE_MAXSIZE)
    results_q_yolo = Queue(maxsize=QUEUE_MAXSIZE)
    results_q_face = Queue(maxsize=QUEUE_MAXSIZE)
    stop_flag = {"stop": False}

    # lanzar threads
    Thread(target=lambda: capture_frames(cap, frames_q_yolo, frames_q_face, stop_flag), daemon=True).start()
    Thread(target=yolo_inference_thread, args=(model, frames_q_yolo, results_q_yolo, stop_flag, DEFAULT_CONF, device_yolo), daemon=True).start()
    Thread(target=face_inference_thread, args=(frames_q_face, results_q_face, stop_flag, known_embeddings), daemon=True).start()

    t_prev = time.time()
    fps_smooth = None
    print("[INFO] Ejecutando. Presiona 'q' para salir.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO
            try:
                _, boxes, confs, cls_ids, names = results_q_yolo.get_nowait()
                frame_out = draw_boxes(frame.copy(), boxes, confs, cls_ids, names, DEFAULT_CONF)
            except Empty:
                frame_out = frame.copy()

            # Faces
            try:
                faces = results_q_face.get_nowait()
                for x1,y1,x2,y2,name,sim,top3 in faces:
                    # dibujar bbox y nombre
                    if x1==x2==y1==y2==0:
                        # sin bbox, dibujar texto en esquina
                        cv2.putText(frame_out, f"{name} ({sim:.2f})", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
                    else:
                        cv2.rectangle(frame_out, (x1,y1),(x2,y2),(0,0,255),2)
                        cv2.putText(frame_out, f"{name} ({sim:.2f})",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    if DEBUG and top3:
                        y0 = y2 + 15
                        for p, s in top3:
                            cv2.putText(frame_out, f"{p}:{s:.2f}", (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255),1)
                            y0 += 12
            except Empty:
                pass

            # FPS
            t_now = time.time()
            fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            fps_smooth = fps if fps_smooth is None else fps_smooth*0.9 + fps*0.1
            cv2.putText(frame_out, f"FPS: {fps_smooth:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)

            cv2.imshow("YOLO11 + Reconocimiento Facial (mejorado)", frame_out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_flag["stop"] = True
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Finalizado.")

def capture_frames(cap, q1, q2, stop_flag):
    while not stop_flag["stop"]:
        ret, frame = cap.read()
        if not ret:
            stop_flag["stop"] = True
            break
        for q in (q1,q2):
            try:
                q.put(frame, timeout=0.02)
            except:
                try:
                    _ = q.get_nowait()
                    q.put(frame)
                except:
                    pass

if __name__ == "__main__":
    main()
