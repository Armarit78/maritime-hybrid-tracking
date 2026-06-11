import cv2

input_file = "EO_GC_PC_SpeedBoat_traj_C.avi"
output_file = "output_coupe.avi"
start_time = 3   # en secondes
end_time = 140    # en secondes

# Ouvre la vidéo
cap = cv2.VideoCapture(input_file)

# Récupère les infos de la vidéo
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Détermine le range de frames à garder
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Définit le codec et crée l'objet VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_num > end_frame:
        break
    if frame_num >= start_frame:
        out.write(frame)
    frame_num += 1

cap.release()
out.release()
print("Découpage terminé !")