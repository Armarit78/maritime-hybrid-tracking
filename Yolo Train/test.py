import os
from ultralytics import YOLO
import cv2

# === Paramètres à modifier selon ton organisation ===
MODEL_PATH = 'runs/train/exp_yolo11n_320_50-8/weights/best.pt'  # chemin vers ton modèle entraîné
IMAGE_DIR = 'TEST/images'                       # dossier avec toutes les images à traiter
OUTPUT_DIR = 'TEST/results_exp_yolo11n_320_50-8/images_traitée'   # dossier où seront sauvegardées les images annotées

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger le modèle
model = YOLO(MODEL_PATH)

print(model.names)  # Donne le mapping effectif
print(len(model.names))  # Doit faire 3

# Charger les noms des classes depuis le modèle
class_names = model.names

# Boucle sur toutes les images du dossier
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(IMAGE_DIR, img_name)

    # Prédire avec le modèle
    results = model(img_path)[0]  # batch de taille 1

    # Charger l'image avec OpenCV pour visualisation
    img = cv2.imread(img_path)

    # Boucle sur chaque détection
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = class_names[cls]
        conf_percent = int(conf * 100)

        # Texte à afficher
        text = f"{label} {conf_percent}%"

        # Dessiner la boîte et le texte
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Sauvegarder l'image annotée
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, img)
    print(f"Image traitée et sauvegardée : {out_path}")