import os
import cv2

input_folder = os.getcwd()
output_folder = os.path.abspath(os.path.join(input_folder, os.pardir))

target_width = 1280
target_height = 720
target_fps = 50  # FPS fixe

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        input_path = os.path.join(input_folder, filename)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_50fps{ext}")
        print(f"Traitement de {filename}...")

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_width, target_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (target_width, target_height))
            out.write(resized_frame)

        cap.release()
        out.release()

print("Redimensionnement terminé pour toutes les vidéos.")