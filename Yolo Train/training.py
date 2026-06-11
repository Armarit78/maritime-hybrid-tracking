import os
import shutil
from pprint import pprint
from ultralytics import YOLO
import torch

def main():
    data_yaml = 'boat_detection/data.yaml'
    model_ckpt = 'yolo11n.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Utilisation du device: {device}")

    params = {
        'data': data_yaml,
        'epochs': 50,
        'batch': 8,
        'imgsz': 320,
        'device': device,
        'project': 'runs/train',
        'name': 'exp_yolo11n_320_50-8',
        'workers': min(os.cpu_count(), 4),  # <= Limite à 4 workers pour Windows
        'verbose': True,
        'save_period': 10,
    }

    print("Paramètres d'entraînement :")
    pprint(params)

    # Création du dossier de run et sauvegarde du script/params
    run_dir = os.path.join(params['project'], params['name'])
    os.makedirs(run_dir, exist_ok=True)
    # Protéger la copie du script (utile seulement si lancé dans un IDE)
    try:
        shutil.copy(__file__, os.path.join(run_dir, 'train_script.py'))
    except Exception as e:
        print(f"Copie du script ignorée : {e}")
    with open(os.path.join(run_dir, 'params.txt'), 'w') as f:
        f.write(str(params))

    # Chargement et entraînement
    model = YOLO(model_ckpt)
    try:
        results = model.train(**params)
    except Exception as e:
        print("Erreur lors de l'entraînement :", e)
        exit(1)

    print("Résumé de l'entraînement:")
    print(results)

if __name__ == "__main__":
    main()