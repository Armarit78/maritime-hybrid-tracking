import cv2
import numpy as np
import onnxruntime as ort
import os
import glob

# === PARAMÈTRES À ADAPTER ===
ONNX_PATH = "converted_models/yolo11n_320_100-16/best.onnx"
INPUT_DIR = "TEST/images/"
OUTPUT_DIR = "TEST/results_yolo11n_320_100-16/images_onnx_yolo11n_320_100-16"
LOG_PATH = os.path.join(OUTPUT_DIR, "result_onnx_batch_log.txt")

CLASS_NAMES = ["Buoy", "Ship", "Sailboat", "Drone"]

CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
IMG_SIZE = 320

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + \
              glob.glob(os.path.join(INPUT_DIR, "*.jpeg")) + \
              glob.glob(os.path.join(INPUT_DIR, "*.png"))

logs = []

def log(msg):
    print(msg)
    logs.append(str(msg))

log(f"Nombre d'images trouvées dans {INPUT_DIR} : {len(image_paths)}")

log(f"\n== Chargement du modèle ONNX depuis : {ONNX_PATH}")
session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
log(f"Entrée ONNX : {input_name} | Sortie ONNX : {output_name}")

for idx, image_path in enumerate(image_paths):
    log(f"\nTraitement [{idx+1}/{len(image_paths)}] : {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        log(f"ERREUR : Impossible de lire {image_path}")
        continue

    log(f"Taille originale de l'image : {img.shape}")

    # Prétraitement (crop central 640x640 + padding si besoin)
    h, w = img.shape[:2]
    x_offset = max((w - IMG_SIZE) // 2, 0)
    y_offset = max((h - IMG_SIZE) // 2, 0)
    crop = img[y_offset:y_offset + IMG_SIZE, x_offset:x_offset + IMG_SIZE]
    if crop.shape[0] != IMG_SIZE or crop.shape[1] != IMG_SIZE:
        crop = cv2.copyMakeBorder(
            crop,
            0, max(0, IMG_SIZE - crop.shape[0]),
            0, max(0, IMG_SIZE - crop.shape[1]),
            cv2.BORDER_CONSTANT,
            value=[114, 114, 114]
        )
    log(f"Taille crop centrale / finale : {crop.shape}")

    # Conversion pour ONNX (RGB, normalisation, CHW, batch)
    img_input = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[None, :, :, :]  # (1, 3, 640, 640)
    log(f"Shape tenseur pour ONNX : {img_input.shape}")

    log("\n== Inference ONNX ...")
    outputs = session.run([output_name], {input_name: img_input})[0]
    log(f"Shape brute sortie ONNX : {outputs.shape}")
    outputs = np.squeeze(outputs)  # (7, 8400) pour 3 classes
    log(f"Shape squeeze : {outputs.shape}")

    num_classes = outputs.shape[0] - 4
    log(f"Nombre de classes détectées dans la sortie ONNX : {num_classes}")

    # Analyse rapide des valeurs de sortie
    log("\n== Analyse des valeurs de sortie (class_scores) :")
    for c in range(4, 4 + num_classes):
        log(f"class {c-4} min/max : {np.min(outputs[c])} / {np.max(outputs[c])}")

    log("\n== Debug sur les 10 premières prédictions :")
    for i in range(min(10, outputs.shape[1])):
        det = outputs[:, i]
        box = det[:4]
        class_scores = det[4:4 + num_classes]  # Déjà "objectness * class_score"
        class_id = np.argmax(class_scores)
        conf = class_scores[class_id]
        log(f"Pred {i}: conf={conf:.3f} | class={class_id} | scores={class_scores}")

    # Post-processing : extraction des boxes ayant une confiance suffisante
    log("\n== Post-processing complet ...")
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

    log(f"Nombre de boxes détectées avant NMS : {len(boxes)}")
    log("Confidences : " + str(confidences[:10]))
    log("Classes : " + str(class_ids[:10]))

    # NMS (suppression des doublons)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    log(f"Boxes : {boxes}")
    log(f"Indices NMS (boxes conservées) : {indices}")

    # Trouve la box avec la confiance max
    max_conf = -1
    max_idx = -1
    for idx_box, conf in enumerate(confidences):
        if conf > max_conf:
            max_conf = conf
            max_idx = idx_box

    # Rectangle noir au centre pour repérer la zone d'inférence
    rect_thickness = 3
    cv2.rectangle(
        img,
        (x_offset, y_offset),
        (x_offset + IMG_SIZE, y_offset + IMG_SIZE),
        (0, 0, 0),
        rect_thickness
    )

    # Affichage des boxes
    box_affichee = 0
    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, w_box, h_box = boxes[i]
        class_id = class_ids[i]
        label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        conf_percent = int(confidences[i] * 100)
        color = (0, 0, 255) if i == max_idx else (0, 255, 0)
        x1_full = x + x_offset
        y1_full = y + y_offset
        x2_full = x1_full + w_box
        y2_full = y1_full + h_box
        cv2.rectangle(img, (x1_full, y1_full), (x2_full, y2_full), color, 2)
        cv2.putText(img, f"{label} {conf_percent}%", (x1_full, y1_full - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        log(f"Box affichée : [{x1_full},{y1_full},{x2_full},{y2_full}] classe : {label} conf : {conf_percent}% scores: {class_scores}")
        box_affichee += 1

    log(f"Nombre de boxes affichées : {box_affichee}")

    # Enregistrement de l'image
    base_name = os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_DIR, base_name)
    cv2.imwrite(save_path, img)
    log(f"Image sauvegardée avec les boxes : {save_path}")

log("Traitement terminé !")

# Sauvegarde des logs dans un fichier texte dans OUTPUT_DIR
with open(LOG_PATH, "w") as f:
    for line in logs:
        f.write(line + "\n")