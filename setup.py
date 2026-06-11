modules = [
    ("ultralytics", "YOLO (Ultralytics)"),
    ("cv2", "OpenCV"),
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("tqdm", "tqdm"),
]

for mod, name in modules:
    try:
        if mod == "cv2":
            import cv2
            print(f"{name}: {cv2.__version__}")
        else:
            m = __import__(mod)
            print(f"{name}: {m.__version__}")
    except ImportError:
        print(f"{name}: Non installé")

import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
