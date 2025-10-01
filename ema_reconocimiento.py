#!/usr/bin/env python3
"""

- Detecta objetos en tiempo real con YOLO11 (se descarga automáticamente la primera vez).
- Detecta y reconoce personas conocidas usando múltiples imágenes por persona (GPU automática).
- Muestra detecciones de objetos y rostros sobre la misma cámara.
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
DEFAULT_MODEL = "yolo11s.pt"
KNOWN_FACES_DIR = "known_people"
CAM_INDEX = 0
FRAME_SCALE = 0.5
DEFAULT_CONF = 0.25
QUEUE_MAXSIZE = 4
FRAME_SKIP = 2
THRESHOLD = 0.5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# ------------------------------------------------

# --------------------------------
# Detección de objetos YOLO11
# --------------------------------
def choose_device():
    try:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return None

def draw_boxes(frame, boxes, confs, cls_ids, names, conf_thres):
    for box, conf, cls in zip(boxes, confs, cls_ids):
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = f"{names.get(int(cls), str(int(cls)))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,200,0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw, y1), (0,200,0), -1)
        cv2.putText(frame, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1,cv2.LINE_AA)
    return frame

def infer_results_to_numpy(r):
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
    return np.array([]), np.array([]), np.array([])

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
        except Exception:
            results_q.put((frame, np.array([]), np.array([]), np.array([]), {}))

# --------------------------------
# Reconocimiento facial
# --------------------------------


device_face = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device_face)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device_face)

def load_known_faces(folder):
    encodings = []
    names = []
    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        if not os.path.isdir(person_folder):
            continue
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg",".png")):
                path = os.path.join(person_folder, filename)
                img = cv2.imread(path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces, _ = mtcnn.detect(img_rgb)
                if faces is not None:
                    x1, y1, x2, y2 = map(int, faces[0])
                    face_crop = img_rgb[y1:y2, x1:x2]
                    face_crop = cv2.resize(face_crop, (160,160))
                    face_tensor = torch.tensor(face_crop.transpose((2,0,1)), dtype=torch.float32).unsqueeze(0).to(device_face)
                    embedding = resnet(face_tensor).detach()
                    encodings.append(embedding)
                    names.append(person_name)
    return encodings, names

def face_inference_thread(frames_q, results_q, stop_flag, known_encodings, known_names):
    frame_count = 0
    last_faces = []
    while not stop_flag["stop"]:
        try:
            frame = frames_q.get(timeout=0.5)
        except Empty:
            continue
        frame_count +=1
        if frame_count % FRAME_SKIP != 0:
            results_q.put(last_faces)
            continue

        small_frame = cv2.resize(frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)
        detected_faces = []

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box / FRAME_SCALE)
                face_crop = frame[y1:y2, x1:x2]
                try:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_crop_resized = cv2.resize(face_crop_rgb, (160,160))
                    face_tensor = torch.tensor(face_crop_resized.transpose((2,0,1)), dtype=torch.float32).unsqueeze(0).to(device_face)
                    embedding = resnet(face_tensor).detach()
                    name = "Desconocido"
                    distances = [torch.norm(embedding - e).item() for e in known_encodings]
                    if distances:
                        min_idx = np.argmin(distances)
                    if distances[min_idx] < THRESHOLD:   # <= más estricto
                        name = known_names[min_idx]
                    detected_faces.append((x1,y1,x2,y2,name))
                except:
                    continue
        last_faces = detected_faces
        results_q.put(detected_faces)

# --------------------------------
# Main unificado
# --------------------------------
def main():
    # Cargar caras conocidas
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    print(f"[INFO] Cargadas {len(known_names)} imágenes de personas conocidas: {set(known_names)}")

    # Cargar YOLO11
    try:
        from ultralytics import YOLO
    except Exception:
        print("ERROR: Instala ultralytics con: pip install -U ultralytics")
        sys.exit(1)

    device_yolo = choose_device()
    print(f"[INFO] Usando device YOLO: {device_yolo or 'auto'}")
    model = YOLO(DEFAULT_MODEL)

    # Cámara
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

    # Threads
    Thread(target=lambda: capture_frames(cap, frames_q_yolo, frames_q_face, stop_flag), daemon=True).start()
    Thread(target=yolo_inference_thread, args=(model, frames_q_yolo, results_q_yolo, stop_flag, DEFAULT_CONF, device_yolo), daemon=True).start()
    Thread(target=face_inference_thread, args=(frames_q_face, results_q_face, stop_flag, known_encodings, known_names), daemon=True).start()

    t_prev = time.time()
    fps_smooth = None

    print("[INFO] Ejecutando. Presiona 'q' para salir.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detección YOLO
            try:
                f_yolo, boxes, confs, cls_ids, names = results_q_yolo.get_nowait()
                frame_yolo = draw_boxes(frame.copy(), boxes, confs, cls_ids, names, DEFAULT_CONF)
            except Empty:
                frame_yolo = frame.copy()

            # Detección facial
            try:
                faces = results_q_face.get_nowait()
                for x1,y1,x2,y2,name in faces:
                    cv2.rectangle(frame_yolo, (x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(frame_yolo,name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            except Empty:
                pass

            # FPS
            t_now = time.time()
            fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            fps_smooth = fps if fps_smooth is None else fps_smooth*0.9 + fps*0.1
            cv2.putText(frame_yolo, f"FPS: {fps_smooth:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

            cv2.imshow("YOLO11 + Reconocimiento Facial", frame_yolo)
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
                q.put(frame, timeout=0.01)
            except:
                try:
                    _ = q.get_nowait()
                    q.put(frame)
                except:
                    pass

if __name__ == "__main__":
    main()