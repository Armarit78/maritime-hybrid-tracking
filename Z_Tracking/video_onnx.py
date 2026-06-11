import os
import cv2
import numpy as np
import time
import onnxruntime as ort

# === Paramètres à modifier ===
ONNX_PATH = "converted_models/yolo11n_320_50-8/best.onnx"
VIDEO_DIR = "TEST/videos/cible"
OUTPUT_DIR = "TEST/results_yolo11n_320_50-8/videos_onnx_yolo11n_320_50-8"
CLASS_NAMES = ["Buoy", "Ship", "Sailboat", "Drone"]

CROP_SIZE = 320
CONF_THRESHOLD = 0.5  # seuil pour la détection (NMS + extraction des boxes)
DISPLAY_THRESHOLD = 0.5  # seuil pour l'affichage (seules les boxes >= 50% sont affichées)
NMS_THRESHOLD = 0.5 #Seuil d’overlap pour la suppression des doublons via NMS

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(video_extensions)]

# Charger le modèle ONNX avec ONNXRuntime
print(f"\n== Chargement du modèle ONNX depuis : {ONNX_PATH}")
session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Entrée ONNX : {input_name} | Sortie ONNX : {output_name}")

for video_idx, video_name in enumerate(video_files):
    video_path = os.path.join(VIDEO_DIR, video_name)
    print(f"\n--- Ouverture de la vidéo [{video_idx + 1}/{len(video_files)}] : {video_path} ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la vidéo : {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Infos vidéo : FPS={fps}, width={width}, height={height}, total_frames={total_frames}")

    if total_frames == 0:
        print(f"ATTENTION : La vidéo {video_name} est vide ou corrompue.")
        cap.release()
        continue

    basename = os.path.splitext(os.path.basename(video_name))[0]
    output_video_path = os.path.join(OUTPUT_DIR, f"{basename}_onnx_YOLOv11_V3S.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_time_ns = 0

    # Calcul du crop central et scale (pour remettre à l'échelle)
    x_offset = max((width - CROP_SIZE) // 2, 0)
    y_offset = max((height - CROP_SIZE) // 2, 0)
    scale_x = width / CROP_SIZE
    scale_y = height / CROP_SIZE

    print("Traitement des frames ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\nFin de lecture à la frame {frame_count}")
            break

        crop = frame[y_offset:y_offset + CROP_SIZE, x_offset:x_offset + CROP_SIZE]
        # Si trop petit, padder (et donc ne pas décaler offset)
        if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
            crop = cv2.copyMakeBorder(
                crop,
                0, max(0, CROP_SIZE - crop.shape[0]),
                0, max(0, CROP_SIZE - crop.shape[1]),
                cv2.BORDER_CONSTANT,
                value=[114, 114, 114]
            )

        # Conversion pour ONNX (RGB, normalisation, CHW, batch)
        img_input = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[None, :, :, :]  # (1, 3, 640, 640)

        start_ns = time.perf_counter_ns()
        outputs = session.run([output_name], {input_name: img_input})[0]
        process_time_ns = time.perf_counter_ns() - start_ns
        frame_time_ms = process_time_ns / 1e6
        processed_fps = 1.0 / (frame_time_ms / 1000) if frame_time_ms > 0 else 0.0
        total_time_ns += process_time_ns
        frame_count += 1

        outputs = np.squeeze(outputs)  # (7, 8400)
        num_classes = outputs.shape[0] - 4

        boxes, confidences, class_ids = [], [], []

        for det_idx in range(outputs.shape[1]):
            det = outputs[:, det_idx]
            cx, cy, bw, bh = det[:4]
            class_scores = det[4:4 + num_classes]
            class_id = np.argmax(class_scores)
            conf = class_scores[class_id]
            if conf > CONF_THRESHOLD:
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)
                box_w = x2 - x1
                box_h = y2 - y1
                boxes.append([x1, y1, box_w, box_h])
                confidences.append(float(conf))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        # Filtrage à l'affichage : boxes >= 50% de certitude
        box_affichee = 0
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            if confidences[i] < DISPLAY_THRESHOLD:
                continue
            box_affichee += 1

        print(f"Frame {frame_count}/{total_frames} "
              f"({100 * frame_count / total_frames:.1f}%) "
              f"| {len(boxes)} boxes avant NMS | {len(indices)} boxes après NMS | "
              f"{box_affichee} boxes affichées (>=50%) | "
              f"Traitement IA : {frame_time_ms:.1f} ms | "
              f"FPS IA: {processed_fps:.2f}")

        # Trouve la box avec la confiance max (>=50%)
        max_conf = -1
        max_idx = -1
        for idx_box, conf in enumerate(confidences):
            if conf >= DISPLAY_THRESHOLD and conf > max_conf:
                max_conf = conf
                max_idx = idx_box

        # Rectangle noir au centre pour repérer la zone d'inférence
        rect_thickness = 3
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + CROP_SIZE, y_offset + CROP_SIZE),
            (0, 0, 0),
            rect_thickness
        )

        # Affichage des boxes sur la frame (seulement >= 50%)
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            if confidences[i] < DISPLAY_THRESHOLD:
                continue
            x, y, w_box, h_box = boxes[i]
            class_id = class_ids[i]
            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
            conf_percent = int(confidences[i] * 100)
            color = (0, 0, 255) if i == max_idx else (0, 255, 0)
            x1_full = x + x_offset
            y1_full = y + y_offset
            x2_full = x1_full + w_box
            y2_full = y1_full + h_box
            cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), color, 2)
            cv2.putText(frame, f"{label} {conf_percent}%", (x1_full, y1_full - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color, 2, cv2.LINE_AA)

        fps_text = f"{processed_fps:.1f} FPS"
        cv2.putText(frame, fps_text, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"\nVidéo annotée sauvegardée ici : {output_video_path}")
    if frame_count > 0:
        avg_fps_ia = frame_count / (total_time_ns / 1e9)
        print(f"Résumé vidéo :")
        print(f"- FPS original de la vidéo : {fps:.2f}")
        print(f"- FPS moyen IA (traitement ONNX) : {avg_fps_ia:.2f}")
        print(f"- Temps total de traitement IA : {total_time_ns / 1e9:.3f} secondes")
        print(f"- Temps moyen par frame IA : {total_time_ns / frame_count / 1e6:.3f} ms")
    print(f"Traitement terminé pour {video_name}. Passage à la vidéo suivante.\n")

print("Traitement de toutes les vidéos terminé.")