import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from ultralytics import YOLO
import cv2
import time
import torch
import matplotlib.pyplot as plt

# === Paramètres à modifier selon ton organisation ===
MODEL_PATH = 'runs/train/exp_yolov8n_640_100-16/weights/best.pt'
VIDEO_DIR = 'TEST/videos/cible'
CROP_SIZE = 320 # carré central 640x640 pour YOLO
PLOTS_DIR = 'TEST/results_exp_yolov8n_640_100-16/resultvideo_exp_yolov8n_640_100-16'

os.makedirs(PLOTS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Utilisation du device : {device}")

model = YOLO(MODEL_PATH)
model.to(device)
class_names = model.names

video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(video_extensions)]

FPS_YMAX = 150  # Échelle fixe pour FPS
MS_YMAX = 100   # Échelle fixe pour ms

for video_name in video_files:
    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la vidéo : {video_path}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    total_time_ns = 0  # temps total en nanosecondes

    frame_times_ms = []  # temps de traitement en ms pour chaque frame
    frame_fps = []       # FPS instantané pour chaque frame

    # Calcul du carré central (offsets)
    x_offset = max((width - CROP_SIZE) // 2, 0)
    y_offset = max((height - CROP_SIZE) // 2, 0)

    print(f"Début traitement vidéo : {video_name} ({total_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop central 640x640 (attention aux bords)
        crop = frame[y_offset:y_offset+CROP_SIZE, x_offset:x_offset+CROP_SIZE]

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
        _ = model(crop, device=device)[0]  # IA seulement, pas de post-traitement
        process_time_ns = time.perf_counter_ns() - start_ns
        total_time_ns += process_time_ns
        frame_count += 1

        process_time_ms = process_time_ns / 1e6
        frame_times_ms.append(process_time_ms)
        frame_fps.append(1000. / process_time_ms if process_time_ms > 0 else 0)

        # Affichage progression toutes les 50 frames
        if total_frames > 0 and frame_count % 50 == 0:
            completion = 100 * frame_count / total_frames
            print(f"Progression : {completion:.1f}% ({frame_count}/{total_frames})")

    cap.release()
    print(f"\nVidéo terminée : {video_name}")
    if frame_count > 0:
        avg_fps = frame_count / (total_time_ns / 1e9)
        print(f"FPS moyen IA (zone centrale) : {avg_fps:.2f}")
        print(f"Temps total IA : {total_time_ns/1e9:.3f} sec")
        print(f"Temps moyen/frame : {total_time_ns/frame_count/1e6:.3f} ms ({total_time_ns/frame_count} ns)")
        if total_frames > 0:
            completion = 100 * frame_count / total_frames
            print(f"Taux de complétion vidéo : {completion:.1f}% ({frame_count}/{total_frames})")
        else:
            print(f"Taux de complétion vidéo : {frame_count} frames (total inconnu)")

        # === MULTIPLOT : ms et FPS dans la même image ===
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot ms
        ax1.plot(range(frame_count), frame_times_ms, color='tab:red')
        ax1.set_ylabel('Temps de traitement (ms)', color='tab:red')
        ax1.set_title(f'Temps de traitement par frame : {video_name}')
        ax1.grid(True)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.set_ylim(0, MS_YMAX)  # Échelle fixe de 0 à 120 ms

        # Plot FPS
        ax2.plot(range(frame_count), frame_fps, color='tab:blue')
        ax2.set_ylabel('FPS instantané', color='tab:blue')
        ax2.set_xlabel('Numéro de frame')
        ax2.set_title('FPS instantané par frame')
        ax2.set_ylim(0, FPS_YMAX)  # Échelle fixe de 0 à 120 FPS
        ax2.grid(True)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        fig.suptitle(f'Performance IA YOLO : {video_name}', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = os.path.join(PLOTS_DIR, f"{os.path.splitext(video_name)[0]}_detect.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Multipanel performance enregistré : {plot_path}")

    print(f"Traitement terminé pour {video_name}.\n")

print("Traitement de toutes les vidéos terminé.")