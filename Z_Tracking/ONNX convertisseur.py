import os
import shutil
from ultralytics import YOLO

dim_crop = 320

pt_path = 'best.pt'
dst_dir = 'converted_models/yolo11n_320_50-8'
os.makedirs(dst_dir, exist_ok=True)

if not os.path.isfile(pt_path):
    raise FileNotFoundError(f"Fichier modèle .pt introuvable : {pt_path}")

# Nettoyage du dossier de destination
for f in os.listdir(dst_dir):
    fp = os.path.join(dst_dir, f)
    if os.path.isfile(fp):
        os.remove(fp)

pt_filename = os.path.basename(pt_path)
pt_dst = os.path.join(dst_dir, pt_filename)
try:
    shutil.copy(pt_path, pt_dst)
    print(f"Copie du modèle .pt terminée : {pt_dst}")
except Exception as e:
    print(f"Erreur lors de la copie du modèle .pt : {e}")
    exit(1)

try:
    model = YOLO(pt_dst)
    print("Chargement du modèle .pt OK.")
    print(f"Classes du modèle : {model.names}")
    print(f"Nombre de classes : {len(model.names)}")
    print("model.model.yaml:", model.model.yaml)  # YAML d'origine
    result = model.export(format='onnx', imgsz=dim_crop)
    onnx_path = getattr(result, "save_path", None)
    print(f"Chemin ONNX (brut): {onnx_path}")
    # Fallback si save_path est None
    if not onnx_path or not isinstance(onnx_path, str):
        # Cherche best.onnx dans le dossier de destination
        onnx_path = os.path.join(dst_dir, "best.onnx")
    print("Fichiers présents dans le dossier : ", os.listdir(dst_dir))
    print("Chemin absolu attendu : ", os.path.abspath(onnx_path))
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError("La conversion ONNX semble avoir échoué : fichier non trouvé.")
    print(f"Conversion ONNX terminée et trouvée : {onnx_path}")

    # Vérification du shape ONNX
    import onnxruntime as ort
    import numpy as np
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dummy_input = np.zeros((1, 3, dim_crop, dim_crop), dtype=np.float32)
    output = session.run([output_name], {input_name: dummy_input})[0]
    print("Shape ONNX :", output.shape)
except Exception as e:
    print(f"Erreur lors de la conversion ONNX : {e}")
    exit(1)

print("Conversion complète et vérifiée !")