import os

# Forcer le mono-threading CPU (utile pour compatibilité, pas pour CUDA)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from ultralytics import YOLO
import cv2
import time

# === Paramètres à modifier selon ton organisation ===
MODEL_PATH = 'runs/train/exp_yolo11n_320_50-8/weights/best.pt'
VIDEO_DIR = 'TEST/videos/cible'
OUTPUT_DIR = 'TEST/results_exp_yolo11n_320_50-8/resultvideo_exp_yolo11n_320_50-8'

CROP_SIZE = 320 # carré central 640x640 pour YOLO

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choix auto du device : GPU si dispo, sinon CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Utilisation du device : {device}")
if device == 'cuda':
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")

model = YOLO(MODEL_PATH)
model.to(device)
class_names = model.names

video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(video_extensions)]

for video_name in video_files:
    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la vidéo : {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    basename = os.path.splitext(os.path.basename(video_name))[0]
    output_video_path = os.path.join(OUTPUT_DIR, f"{basename}_center640_annotated_exp142.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_time_ns = 0  # temps total en nanosecondes

    # Calcul du carré central (offsets)
    x_offset = max((width - CROP_SIZE) // 2, 0)
    y_offset = max((height - CROP_SIZE) // 2, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop central 640x640 (attention aux bords)
        crop = frame[y_offset:y_offset + CROP_SIZE, x_offset:x_offset + CROP_SIZE]

        # Si la vidéo est plus petite que 640, padder
        if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
            crop = cv2.copyMakeBorder(
                crop,
                0, max(0, CROP_SIZE - crop.shape[0]),
                0, max(0, CROP_SIZE - crop.shape[1]),
                cv2.BORDER_CONSTANT,
                value=[114, 114, 114]
            )

        start_ns = time.perf_counter_ns()
        results = model(crop, device=device)[0]
        process_time_ns = time.perf_counter_ns() - start_ns
        total_time_ns += process_time_ns
        frame_count += 1

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            label = class_names[cls]
            conf_percent = int(conf * 100)
            # Remettre la box sur l'image complète
            x1_full = x1 + x_offset
            y1_full = y1 + y_offset
            x2_full = x2 + x_offset
            y2_full = y2 + y_offset

            text = f"{label} {conf_percent}%"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), color, 2)
            cv2.putText(frame, text, (x1_full, y1_full - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        process_time_s = process_time_ns / 1e9
        processed_fps = 1.0 / process_time_s if process_time_s > 0 else 0.0
        fps_text = f"Traitement: {processed_fps:.1f} FPS"
        cv2.putText(frame, fps_text, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(frame)

        percent_done = 100 * frame_count / total_frames if total_frames > 0 else 0
        print(f"{video_name} : {frame_count}/{total_frames} frames traitées ({percent_done:.1f}%)", end='\r')

    cap.release()
    out.release()
    print(f"\nVidéo annotée sauvegardée ici : {output_video_path}")
    if frame_count > 0:
        avg_fps = frame_count / (total_time_ns / 1e9)
        print(f"FPS moyen de traitement pour {video_name}: {avg_fps:.2f}")
        print(f"Temps total de traitement IA (zone centrale) : {total_time_ns / 1e9:.3f} secondes")
        print(
            f"Temps moyen par frame (IA) : {total_time_ns / frame_count / 1e6:.3f} ms ({total_time_ns / frame_count} ns)")

    print(f"Traitement terminé pour {video_name}. Passage à la vidéo suivante.\n")

print("Traitement de toutes les vidéos terminé.")