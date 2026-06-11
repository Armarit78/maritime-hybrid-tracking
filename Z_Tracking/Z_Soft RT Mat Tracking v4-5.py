import os
import cv2
import numpy as np
import time
import onnxruntime as ort
import threading
import queue
from collections import deque
import csv
import json
import matplotlib.pyplot as plt

# === PARAMÈTRES PRINCIPAUX DU MODÈLE ET DES DOSSIERS ===
ONNX_PATH = "converted_models/yolo11n_640_50-8/best.onnx"
CLASS_NAMES = ["Buoy", "Ship", "Sailboat", "Drone"]
OUTPUT_DIR = "TEST/results_yolo11n_640_50-8/SOFT_RT_MAT_cam_onnx_yolo11n_640_50-8"

# === PARAMÈTRES DE DÉTECTION ET TRAITEMENT DES IMAGES ===
CROP_SIZE = 640
CONF_THRESHOLD = 0.5
DISPLAY_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
MIN_BOX_RATIO = 0.08
MAX_BOX_RATIO = 0.95

# === PARAMÈTRES DU TRACKING ===
TRACKER_TYPE = "KCF"
KALMAN_LISSAGE = True
KALMAN_MEM_SIZE = 5
EDGE_MARGIN = 10
EDGE_FRAMES_LIMIT = 5

# === PARAMÈTRES DE RÉ-IDENTIFICATION (ReID) ===
REID_ENABLED = True
HIST_REID_THRESHOLD = 0.7
REID_HIST_SIZE = (8,8,8)
REID_MEM_SIZE = 5
DESCRIPTOR_TYPE = "hist"
REID_WEIGHT = 0.3

# === PARAMÈTRES DE FRÉQUENCE & PERFORMANCE ===
YOLO_AUTO_FREQ = True
YOLO_FREQ_MIN = 0.5
YOLO_FREQ_MAX = 10
YOLO_FREQ_START = 3
TARGET_FPS = 50
TARGET_PERIOD = 1.0 / TARGET_FPS
ELAPSED_SKIP_LIMIT = 0.02

# === PARAMÈTRES D'AFFICHAGE ET EXPORTS ===
SHOW_KCF_BOX = False
SHOW_YOLO_BOX = True
SHOW_INTERP_BOX = True
SHOW_KALMAN_BOX = True
SHOW_KALMAN_TRAJ = True
EXPORT_CSV = True
EXPORT_DEBUG_IMAGES = False
EXPORT_METRICS_PLOTS = False
EXPORT_SUMMARY_JSON = False

EXPORT_VIDEO = False  # activer/désactiver l'enregistrement vidéo
ENABLE_LIVE_VIEW = True  # activer/désactiver le thread vidéo

os.makedirs(OUTPUT_DIR, exist_ok=True)
if EXPORT_DEBUG_IMAGES:
    os.makedirs(os.path.join(OUTPUT_DIR, "debug"), exist_ok=True)

print(f"\n== Chargement du modèle ONNX depuis : {ONNX_PATH}")
session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Entrée ONNX : {input_name} | Sortie ONNX : {output_name}")

# ========== THREAD LIVE DISPLAY ==========
class VideoDisplayThread(threading.Thread):
    def __init__(self, window_name="Live Tracking", max_queue_size=10):
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_flag = threading.Event()
        self.window_name = window_name
        self.reset_flag = threading.Event()  # Ajout pour le reset cible

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while not self.stop_flag.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    self.stop()
                elif key == ord('m'):
                    self.reset_flag.set()
            except queue.Empty:
                continue
        cv2.destroyWindow(self.window_name)

    def submit(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def stop(self):
        self.stop_flag.set()

    def check_and_clear_reset(self):
        if self.reset_flag.is_set():
            self.reset_flag.clear()
            return True
        return False

# ========== FONCTIONS UTILES ==========

def get_appearance_descriptor(frame, box, desc_type="hist", hist_size=REID_HIST_SIZE):
    x, y, w, h = [int(round(v)) for v in box]
    x, y = max(x,0), max(y,0)
    w, h = max(w,1), max(h,1)
    x2, y2 = min(x+w, frame.shape[1]), min(y+h, frame.shape[0])
    roi = frame[y:y2, x:x2]
    if roi.shape[0] <= 1 or roi.shape[1] <= 1:
        return None
    if desc_type == "hist":
        hist = cv2.calcHist([roi], [0,1,2], None, hist_size, [0,256,0,256,0,256])
        return cv2.normalize(hist, hist).flatten()
    elif desc_type == "texture":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lbp = cv2.Laplacian(gray, cv2.CV_64F).var()
        return np.array([lbp])
    elif desc_type == "deep":
        return np.mean(roi, axis=(0,1))
    else:
        return None

def descriptor_similarity(desc_list, candidate, desc_type="hist"):
    if candidate is None or len(desc_list) == 0:
        return 0, 0
    if desc_type == "hist":
        sims = [cv2.compareHist(h, candidate, cv2.HISTCMP_CORREL) for h in desc_list]
    else:
        sims = [np.dot(h, candidate) / (np.linalg.norm(h)*np.linalg.norm(candidate)+1e-8) for h in desc_list]
    return max(sims), float(np.mean(sims))

def box_on_edge(box, crop_size=CROP_SIZE, margin=EDGE_MARGIN):
    x, y, w, h = [int(round(v)) for v in box]
    touches_left   = x <= margin
    touches_top    = y <= margin
    touches_right  = (x + w) >= (crop_size - margin)
    touches_bottom = (y + h) >= (crop_size - margin)
    return touches_left or touches_right or touches_top or touches_bottom

def adaptive_interpolation(box1, box2, edge_frames, crop_size=CROP_SIZE, margin=EDGE_MARGIN):
    alpha_xy = 0.3
    alpha_wh = 0.05
    min_w = crop_size * MIN_BOX_RATIO
    min_h = crop_size * MIN_BOX_RATIO
    max_w = crop_size * MAX_BOX_RATIO
    max_h = crop_size * MAX_BOX_RATIO
    if edge_frames > 2:
        alpha_wh = max(0.01, alpha_wh - 0.01 * edge_frames)
    new_x = box1[0] * (1 - alpha_xy) + box2[0] * alpha_xy
    new_y = box1[1] * (1 - alpha_xy) + box2[1] * alpha_xy
    new_w = box1[2] * (1 - alpha_wh) + box2[2] * alpha_wh
    new_h = box1[3] * (1 - alpha_wh) + box2[3] * alpha_wh
    new_w = np.clip(new_w, min_w, max_w)
    new_h = np.clip(new_h, min_h, max_h)
    x2, y2, w2, h2 = box2
    touches_left   = x2 <= margin
    touches_top    = y2 <= margin
    touches_right  = (x2 + w2) >= (crop_size - margin)
    touches_bottom = (y2 + h2) >= (crop_size - margin)
    if touches_left and touches_right:
        new_w = max(box1[2], box2[2])
    if touches_top and touches_bottom:
        new_h = max(box1[3], box2[3])
    if new_w < min_w and (touches_left or touches_right):
        new_w = max(box1[2], min_w)
    if new_h < min_h and (touches_top or touches_bottom):
        new_h = max(box1[3], min_h)
    return [new_x, new_y, new_w, new_h], (touches_left or touches_right or touches_top or touches_bottom)

def iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, boxA[2]) * max(1, boxA[3])
    boxBArea = max(1, boxB[2]) * max(1, boxB[3])
    return interArea / float(boxAArea + boxBArea - interArea)

def compute_global_conf(tracker_conf, yolo_conf, kalman_var, reid_sim, weights=(0.3,0.3,0.2,0.2)):
    tracker_c = np.clip(tracker_conf, 0, 1)
    yolo_c = np.clip(yolo_conf, 0, 1)
    kalman_c = 1.0 / (1 + np.sqrt(kalman_var))
    reid_c = np.clip(reid_sim, 0, 1)
    w1, w2, w3, w4 = weights
    return w1*tracker_c + w2*yolo_c + w3*kalman_c + w4*reid_c

class AsyncYoloThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=4)
        self.result_queue = queue.Queue(maxsize=4)
        self.stop_flag = threading.Event()
    def run(self):
        while not self.stop_flag.is_set():
            try:
                crop, frame_id = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            boxes, confidences, class_ids, indices, yolo_time = run_yolo_inference(crop)
            self.result_queue.put({
                "boxes": boxes,
                "confidences": confidences,
                "class_ids": class_ids,
                "indices": indices,
                "frame_id": frame_id,
                "yolo_time": yolo_time
            })
    def submit(self, crop, frame_id):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put((crop, frame_id))
    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    def stop(self):
        self.stop_flag.set()

def run_yolo_inference(crop):
    img_input = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[None, :, :, :]
    start = time.perf_counter()
    outputs = session.run([output_name], {input_name: img_input})[0]
    outputs = np.squeeze(outputs)
    yolo_time = time.perf_counter() - start
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
    return boxes, confidences, class_ids, indices, yolo_time

class FrameBuffer:
    def __init__(self, maxlen=50):
        self.buffer = deque(maxlen=maxlen)
    def add(self, frame_id, frame, crop, tracking_box):
        self.buffer.append((frame_id, frame.copy(), crop.copy(), tracking_box))
    def get_from_frame_id(self, frame_id):
        for it in self.buffer:
            if it[0] == frame_id:
                return it
        return None
    def get_latest(self):
        return self.buffer[-1] if len(self.buffer) > 0 else None

class MotionHistory:
    def __init__(self, maxlen=15):
        self.history = deque(maxlen=maxlen)
    def update(self, frame_id, box):
        if box is not None:
            self.history.append((frame_id, box))
    def get_mean_velocity(self, since_frame=None):
        if len(self.history) < 2:
            return 0, 0, 0, 0
        if since_frame is None:
            a, b = self.history[0], self.history[-1]
        else:
            a = None
            for h in self.history:
                if h[0] == since_frame:
                    a = h
                    break
            b = self.history[-1]
            if a is None:
                a = self.history[0]
        dx = b[1][0] - a[1][0]
        dy = b[1][1] - a[1][1]
        dw = b[1][2] - a[1][2]
        dh = b[1][3] - a[1][3]
        dt = b[0] - a[0]
        if dt == 0:
            return 0, 0, 0, 0
        return dx / dt, dy / dt, dw / dt, dh / dt

class KalmanFilterBoxFull:
    def __init__(self, init_box, dt=1.):
        self.kf = cv2.KalmanFilter(12, 4)
        self.kf.transitionMatrix = np.eye(12, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i+4] = dt
            self.kf.transitionMatrix[i, i+8] = 0.5 * dt * dt
            self.kf.transitionMatrix[i+4, i+8] = dt
        self.kf.measurementMatrix = np.zeros((4, 12), np.float32)
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1
        self.kf.measurementMatrix[2, 2] = 1
        self.kf.measurementMatrix[3, 3] = 1
        self.kf.processNoiseCov = 1e-2 * np.eye(12, dtype=np.float32)
        self.kf.measurementNoiseCov = 1e-1 * np.eye(4, dtype=np.float32)
        self.kf.errorCovPost = 1.0 * np.eye(12, dtype=np.float32)
        x, y, w, h = init_box
        self.kf.statePost = np.zeros((12, 1), dtype=np.float32)
        self.kf.statePost[0, 0] = float(x)
        self.kf.statePost[1, 0] = float(y)
        self.kf.statePost[2, 0] = float(w)
        self.kf.statePost[3, 0] = float(h)
    def correct(self, meas_box):
        x, y, w, h = meas_box
        measurement = np.array([[float(x)], [float(y)], [float(w)], [float(h)]], dtype=np.float32)
        self.kf.correct(measurement)
    def predict(self):
        pred = self.kf.predict()
        box = [float(pred[i, 0]) for i in range(4)]
        vel = [float(pred[i, 0]) for i in range(4, 8)]
        acc = [float(pred[i, 0]) for i in range(8, 12)]
        return box, vel, acc
    def get_variance(self):
        return float(np.mean(np.diag(self.kf.errorCovPost)[:4]))

class KalmanMemory:
    def __init__(self, mem_size=KALMAN_MEM_SIZE):
        self.box_mem = deque(maxlen=mem_size)
    def push(self, frame_idx, box, hist=None):
        self.box_mem.append((frame_idx, box.copy(), hist))
    def get_candidates(self, min_age=2):
        return list(self.box_mem)[::-1]

class MultiKalmanTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
    def add(self, box):
        kf = KalmanFilterBoxFull(box, dt=1.0)
        tid = self.next_id
        self.tracks[tid] = {"kalman": kf, "box": box, "lost": 0}
        self.next_id += 1
        return tid
    def update(self, detections, iou_thresh=0.3):
        assigned = set()
        for tid, track in self.tracks.items():
            best_iou = 0
            best_det = None
            best_idx = -1
            for i, det in enumerate(detections):
                iou_val = iou(track["box"], det)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = det
                    best_idx = i
            if best_iou > iou_thresh and best_det is not None:
                track["kalman"].correct(best_det)
                track["box"] = best_det
                track["lost"] = 0
                assigned.add(best_idx)
            else:
                pred, _, _ = track["kalman"].predict()
                track["box"] = pred
                track["lost"] += 1
        for i, det in enumerate(detections):
            if i not in assigned:
                self.add(det)
    def get_active_tracks(self, max_lost=5):
        return {tid: t["box"] for tid, t in self.tracks.items() if t["lost"] <= max_lost}
    def find_most_similar(self, ref_box, iou_thresh=0.4):
        best_tid = None
        best_iou = 0
        for tid, t in self.tracks.items():
            iou_val = iou(ref_box, t["box"])
            if iou_val > best_iou and iou_val > iou_thresh:
                best_iou = iou_val
                best_tid = tid
        return best_tid, best_iou

def select_initial_roi(frame):
    print("Sélectionnez la cible à suivre (sélectionnez puis appuyez sur Entrée)...")
    roi = cv2.selectROI("Sélection de la cible", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Sélection de la cible")
    x, y, w, h = roi
    return [x, y, w, h] if w > 0 and h > 0 else None

def safe_box(box):
    if box is None or len(box) != 4:
        return None
    try:
        return tuple(int(round(float(x))) for x in box)
    except Exception:
        return None

# === MAIN LOOP POUR LA CAMERA ===

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra.")
    exit()

cap.set(cv2.CAP_PROP_FPS,50)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

basename = "camera_live"
total_frames = 0  # flux infini, pourra être utilisé pour les stats

print(f"{fps}")
print(f"{width,height}")

output_video_path = os.path.join(OUTPUT_DIR, f"{basename}_yoloKCF_SOFT_RT_MAT_v3.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if EXPORT_VIDEO:
    out = cv2.VideoWriter(output_video_path, fourcc, TARGET_FPS, (width, height))
else:
    out = None

csv_path = os.path.join(OUTPUT_DIR, f"{basename}_tracking_log.csv")
csv_file = None
csv_writer = None
if EXPORT_CSV:
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "tracker_box", "yolo_box", "kalman_box", "kalman_vel", "kalman_acc", "status",
        "iou", "global_conf", "yolo_delay", "tracker_type", "skipped", "edge_frames"
    ])

x_offset = max((width - CROP_SIZE) // 2, 0)
y_offset = max((height - CROP_SIZE) // 2, 0)

ret, first_frame = cap.read()
if not ret:
    print("Impossible de lire la première frame pour la sélection de cible.")
    cap.release()
    exit()

print(f"[INFO] Sélection de la cible initiale sur le flux caméra.")
init_box = select_initial_roi(first_frame)
bbox = safe_box(init_box)
if bbox is None:
    print("Aucune cible sélectionnée ou bounding box invalide :", init_box)
    cap.release()
    exit()

print(f"[INFO] Initialisation du tracker ({TRACKER_TYPE}) avec la box {bbox}.")
if TRACKER_TYPE == "KCF":
    tracker = cv2.TrackerKCF_create()
elif TRACKER_TYPE == "CSRT":
    tracker = cv2.TrackerCSRT_create()
elif TRACKER_TYPE == "MOSSE":
    tracker = cv2.TrackerMOSSE_create()
else:
    tracker = cv2.TrackerKCF_create()
tracker.init(first_frame, bbox)

kalman_box = KalmanFilterBoxFull(bbox, dt=1.0)
kalman_traj = []

yolo_thread = AsyncYoloThread()
yolo_thread.start()
frame_buffer = FrameBuffer(maxlen=70)
tracking_box = init_box
tracking_confidence = 1.0
last_yolo_box = None
last_yolo_conf = 0.0
last_yolo_frame_id = -10
last_valid_box = init_box
last_interp_box = None
last_interp_frame = -1
edge_frames = 0

reid_mem = deque(maxlen=REID_MEM_SIZE)
kalman_mem = KalmanMemory(mem_size=KALMAN_MEM_SIZE)
motion_hist = MotionHistory(maxlen=15)
metrics = {"iou": [], "box_w": [], "box_h": [], "tracker_conf": [], "global_conf": [], "yolo_freq": []}
debug_frames = []

yolo_freq = YOLO_FREQ_START
yolo_times = []
last_yolo_submit = -1.0
force_yolo = False
frame_count = 0
tracking_status = "INIT"
finished = False
min_iou_correction = 0.2
yolo_corrections = 0
alerts_iou = 0
alerts_tracking_loss = 0
tracking_only_frames = 0
yolo_delays = []

multi_kalman = MultiKalmanTracker()
multi_kalman.add(bbox)

# === Thread de retour vidéo en direct ===
if ENABLE_LIVE_VIEW:
    display_thread = VideoDisplayThread(window_name=f"Live Tracking - {basename}", max_queue_size=20)
    display_thread.start()
else:
    display_thread = None

print(f"[INFO] Début du traitement du flux caméra.")
try:
    while True:
        frame_start = time.perf_counter_ns()
        ret, frame = cap.read()
        if not ret:
            print(f"\n[INFO] Fin de lecture à la frame {frame_count}")
            break

        # ------ AJOUT RESET CIBLE ------
        if ENABLE_LIVE_VIEW and display_thread is not None and display_thread.check_and_clear_reset():
            print("\n[RESET] Demande de nouvelle sélection de cible à la frame", frame_count)

            # 1. Vide le buffer d'affichage (display_thread.frame_queue)
            while not display_thread.frame_queue.empty():
                try:
                    display_thread.frame_queue.get_nowait()
                except queue.Empty:
                    break

            # 2. Vide le buffer YOLO
            while hasattr(yolo_thread, "frame_queue") and not yolo_thread.frame_queue.empty():
                try:
                    yolo_thread.frame_queue.get_nowait()
                except queue.Empty:
                    break
            while hasattr(yolo_thread, "result_queue") and not yolo_thread.result_queue.empty():
                try:
                    yolo_thread.result_queue.get_nowait()
                except queue.Empty:
                    break

            print(
                f"[RESET] Buffers vidés : display={display_thread.frame_queue.qsize()} YOLO={yolo_thread.frame_queue.qsize()}")

            # Sélection nouvelle box
            new_box = select_initial_roi(frame)
            bbox = safe_box(new_box)
            if bbox is not None:
                print("[RESET] Nouvelle cible sélectionnée:", bbox)
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, bbox)
                kalman_box = KalmanFilterBoxFull(bbox, dt=1.0)
                kalman_traj = []
                frame_buffer = FrameBuffer(maxlen=70)
                tracking_box = new_box
                tracking_confidence = 1.0
                last_yolo_box = None
                last_yolo_conf = 0.0
                last_yolo_frame_id = -10
                last_valid_box = new_box
                last_interp_box = None
                last_interp_frame = -1
                edge_frames = 0
                reid_mem = deque(maxlen=REID_MEM_SIZE)
                kalman_mem = KalmanMemory(mem_size=KALMAN_MEM_SIZE)
                motion_hist = MotionHistory(maxlen=15)
                multi_kalman = MultiKalmanTracker()
                multi_kalman.add(bbox)
                tracking_status = "INIT"
                # Reset aussi les métriques YOLO si besoin
                yolo_times = []
                yolo_freq = YOLO_FREQ_START
                print("[RESET] Historiques, buffers et trackers réinitialisés. Reprise du tracking à la frame",
                      frame_count)
            else:
                print("[RESET] Sélection de cible annulée ou invalide. On conserve la cible actuelle.")

        crop = frame[y_offset:y_offset + CROP_SIZE, x_offset:x_offset + CROP_SIZE]
        if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
            crop = cv2.copyMakeBorder(
                crop,
                0, max(0, CROP_SIZE - crop.shape[0]),
                0, max(0, CROP_SIZE - crop.shape[1]),
                cv2.BORDER_CONSTANT, value=[114, 114, 114]
            )

        frame_buffer.add(frame_count, frame, crop, tracking_box)

        now = time.time()
        if YOLO_AUTO_FREQ and len(yolo_times) > 5:
            avg_yolo = np.mean(yolo_times[-10:])
            yolo_freq = np.clip(1.0 / (avg_yolo + 1e-2), YOLO_FREQ_MIN, YOLO_FREQ_MAX)

        tracker_conf = 0.9 if tracking_box is not None else 0
        kalman_var = kalman_box.get_variance()
        yolo_submit_this = False
        if force_yolo or (last_yolo_submit < 0 or (now - last_yolo_submit) > (1.0 / yolo_freq)):
            yolo_submit_this = True
            last_yolo_submit = now
            force_yolo = False
        if tracker_conf < 0.4 or kalman_var > (CROP_SIZE*0.1)**2:
            yolo_submit_this = True

        # --- LOGIQUE DE PERTE/RECOVERY ---
        if tracking_status == "PERTE" or tracking_box is None:
            kalman_pred_box, kalman_pred_vel, kalman_pred_acc = kalman_box.predict()
            xk, yk, wk, hk = [int(v) for v in kalman_pred_box]
            cv2.rectangle(frame, (xk, yk), (xk + wk, yk + hk), (255, 0, 255), 2)
            cv2.putText(frame, "Projection theorique", (xk, yk-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

            yolo_thread.submit(crop.copy(), frame_count)
            yolo_result_pred = yolo_thread.get_result()
            best_score, best_iou, best_idx = 0, 0, -1
            candidate_box, candidate_hist = None, None
            pred_dir = np.array(kalman_pred_vel[:2]) if kalman_pred_vel is not None else np.zeros(2)
            if yolo_result_pred is not None:
                indices_pred = yolo_result_pred["indices"]
                boxes_pred = yolo_result_pred["boxes"]
                confidences_pred = yolo_result_pred["confidences"]
                for idx, i in enumerate(indices_pred):
                    i0 = i[0] if hasattr(i,'__len__') else i
                    if confidences_pred[i0] < DISPLAY_THRESHOLD:
                        continue
                    yolo_box = boxes_pred[i0]
                    abs_box = [
                        float(yolo_box[0] + x_offset),
                        float(yolo_box[1] + y_offset),
                        float(yolo_box[2]),
                        float(yolo_box[3]),
                    ]
                    hist_cand = get_appearance_descriptor(frame, abs_box, desc_type=DESCRIPTOR_TYPE) if REID_ENABLED else None
                    max_hist_sim, mean_hist_sim = descriptor_similarity(reid_mem, hist_cand, desc_type=DESCRIPTOR_TYPE) if REID_ENABLED else (1.0, 1.0)
                    iou_val = max(iou(kalman_pred_box, abs_box), max([iou(m[1], abs_box) for m in kalman_mem.get_candidates()] + [0]))
                    if np.linalg.norm(pred_dir) > 0.1:
                        box_center = np.array([abs_box[0]+abs_box[2]/2, abs_box[1]+abs_box[3]/2])
                        pred_center = np.array([kalman_pred_box[0]+kalman_pred_box[2]/2, kalman_pred_box[1]+kalman_pred_box[3]/2])
                        move_vec = box_center - pred_center
                        align = np.dot(move_vec, pred_dir) / (np.linalg.norm(move_vec)*np.linalg.norm(pred_dir)+1e-8)
                        dir_bonus = 0.15 * max(0, align)
                    else:
                        dir_bonus = 0
                    score = iou_val + 0.3*max_hist_sim + dir_bonus
                    if score > best_score and max_hist_sim > HIST_REID_THRESHOLD:
                        best_score, best_iou, best_idx = score, iou_val, idx
                        candidate_box = abs_box
                        candidate_hist = hist_cand
            if candidate_box is not None:
                print(f"[PERTE->YOLO] Reprise du tracking sur box YOLO (IoU Kalman/Mem={best_iou:.2f}, sim={best_score:.2f})")
                tracker = cv2.TrackerKCF_create()
                bbox_safe = safe_box(candidate_box)
                if bbox_safe is not None:
                    tracker.init(frame, bbox_safe)
                    tracking_box = candidate_box
                    last_valid_box = candidate_box
                    kalman_box.correct(candidate_box)
                    tracking_status = "OK"
                    if REID_ENABLED and candidate_hist is not None:
                        reid_mem.append(candidate_hist)
                    kalman_mem.push(frame_count, candidate_box, candidate_hist)
            else:
                print(f"[PERTE] Pas de box plausible trouvée à la frame {frame_count}.")
            frame_count += 1
            if ENABLE_LIVE_VIEW and display_thread is not None:
                display_thread.submit(frame)
            continue

        if yolo_submit_this:
            yolo_thread.submit(crop.copy(), frame_count)
        yolo_result = yolo_thread.get_result()

        replay_applied = False
        curr_interp_box = None
        yolo_delay_frames = 0
        yolo_status = "NONE"
        boxes = []
        indices = []
        confidences = []
        class_ids = []
        best_box = None
        best_conf = -1
        best_class = None

        if yolo_result is not None:
            indices = yolo_result["indices"]
            confidences = yolo_result["confidences"]
            boxes = yolo_result["boxes"]
            class_ids = yolo_result["class_ids"]
            yolo_frame_id = yolo_result["frame_id"] if "frame_id" in yolo_result else frame_count

            multi_kalman.update(boxes)

            if len(indices) > 0:
                for i in indices:
                    i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                    if confidences[i] >= DISPLAY_THRESHOLD and confidences[i] > best_conf:
                        best_conf = confidences[i]
                        best_box = boxes[i]
                        best_class = class_ids[i]
            if best_box is not None:
                print(f"[YOLO] Détection frame {frame_count} - Box: {best_box}, Confiance: {best_conf:.2f}, Classe: {CLASS_NAMES[best_class]}")
                abs_box = [
                    float(best_box[0] + x_offset),
                    float(best_box[1] + y_offset),
                    float(best_box[2]),
                    float(best_box[3]),
                ]
                buffer_frame = frame_buffer.get_from_frame_id(yolo_frame_id)
                if buffer_frame is not None:
                    _, yolo_frame_img, _, _ = buffer_frame
                    bbox_yolo = safe_box(abs_box)
                    if bbox_yolo is None:
                        continue
                    tracker_temp = cv2.TrackerKCF_create()
                    tracker_temp.init(yolo_frame_img, bbox_yolo)
                    curr_box = abs_box
                    for idx in range(yolo_frame_id+1, frame_count+1):
                        next_frame = frame_buffer.get_from_frame_id(idx)
                        if next_frame is not None:
                            _, next_img, _, _ = next_frame
                            ok, temp_box = tracker_temp.update(next_img)
                            if ok:
                                curr_box = temp_box
                            else:
                                print(f"[WARNING] Replay KCF KO à frame {idx}")
                                break
                    if last_valid_box is not None:
                        iou_value = iou(last_valid_box, curr_box)
                        curr_interp_box, on_edge = adaptive_interpolation(
                            last_valid_box, curr_box, edge_frames)
                        bbox_interp = safe_box(curr_interp_box)
                        if on_edge:
                            edge_frames = min(edge_frames+1, EDGE_FRAMES_LIMIT)
                        else:
                            edge_frames = 0
                        desc = get_appearance_descriptor(frame, curr_interp_box, desc_type=DESCRIPTOR_TYPE)
                        max_hist_sim, mean_hist_sim = descriptor_similarity(reid_mem, desc, desc_type=DESCRIPTOR_TYPE) if REID_ENABLED else (1.0, 1.0)
                        global_conf = compute_global_conf(tracker_conf, best_conf, kalman_var, max_hist_sim)
                        if iou_value < min_iou_correction or bbox_interp is None or global_conf < 0.35:
                            alerts_iou += 1
                            yolo_status = "IGNORED_IOU"
                            print(f"[ALERTE] Correction YOLO ignorée à la frame {frame_count} (IoU={iou_value:.2f} Conf={global_conf:.2f})")
                        else:
                            print(f"[CORRECTION] Correction YOLO appliquée à la frame {frame_count} (IoU={iou_value:.2f} ConfGlobale={global_conf:.2f})")
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(frame, bbox_interp)
                            tracking_box = curr_interp_box
                            last_yolo_box = abs_box
                            last_yolo_conf = best_conf
                            last_yolo_frame_id = yolo_frame_id
                            last_valid_box = curr_interp_box
                            replay_applied = True
                            yolo_corrections += 1
                            yolo_status = "CORRECTION"
                            last_interp_box = curr_interp_box
                            last_interp_frame = frame_count
                            if REID_ENABLED and desc is not None:
                                reid_mem.append(desc)
                            kalman_mem.push(frame_count, curr_interp_box, desc)
                    else:
                        bbox_curr = safe_box(curr_box)
                        if bbox_curr is None:
                            continue
                        print(f"[CORRECTION] Première correction YOLO appliquée à la frame {frame_count}")
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, bbox_curr)
                        tracking_box = curr_box
                        last_yolo_box = abs_box
                        last_yolo_conf = best_conf
                        last_yolo_frame_id = yolo_frame_id
                        last_valid_box = curr_box
                        replay_applied = True
                        yolo_corrections += 1
                        yolo_status = "CORRECTION"
                        last_interp_box = curr_box
                        last_interp_frame = frame_count
                        desc = get_appearance_descriptor(frame, curr_box, desc_type=DESCRIPTOR_TYPE)
                        if REID_ENABLED and desc is not None:
                            reid_mem.append(desc)
                        kalman_mem.push(frame_count, curr_box, desc)
                else:
                    frames_passed = frame_count - yolo_frame_id
                    dx, dy, dw, dh = motion_hist.get_mean_velocity(since_frame=yolo_frame_id)
                    proj_box = [
                        float(abs_box[0] + dx * frames_passed),
                        float(abs_box[1] + dy * frames_passed),
                        float(abs_box[2] + dw * frames_passed),
                        float(abs_box[3] + dh * frames_passed)
                    ]
                    bbox_proj = safe_box(proj_box)
                    if last_valid_box is not None:
                        proj_box, on_edge = adaptive_interpolation(
                            last_valid_box, proj_box, edge_frames)
                        bbox_proj = safe_box(proj_box)
                        if on_edge:
                            edge_frames = min(edge_frames + 1, EDGE_FRAMES_LIMIT)
                        else:
                            edge_frames = 0
                    if bbox_proj is None:
                        continue
                    print(
                        f"[PROJECTION] Correction YOLO projetée à la frame {frame_count} (delay {frames_passed} frames)")
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, bbox_proj)
                    tracking_box = proj_box
                    last_yolo_box = abs_box
                    last_yolo_conf = best_conf
                    last_yolo_frame_id = yolo_frame_id
                    last_valid_box = proj_box
                    last_interp_box = proj_box
                    last_interp_frame = frame_count
                    yolo_corrections += 1
                    yolo_status = "PROJECTION"
                    desc = get_appearance_descriptor(frame, proj_box, desc_type=DESCRIPTOR_TYPE)
                    if REID_ENABLED and desc is not None:
                        reid_mem.append(desc)
                    kalman_mem.push(frame_count, proj_box, desc)
                yolo_delay_frames = frame_count - (
                    yolo_result["frame_id"] if "frame_id" in yolo_result else frame_count)
                yolo_times.append(yolo_result["yolo_time"] if "yolo_time" in yolo_result else 0)
                yolo_delays.append(yolo_delay_frames)

                # --- Gestion recovery avancée : multi-kalman + resync en cas de saut brutal
            if last_valid_box is not None and tracking_box is not None:
                dx = abs((last_valid_box[0] + last_valid_box[2] / 2) - (tracking_box[0] + tracking_box[2] / 2))
                dy = abs((last_valid_box[1] + last_valid_box[3] / 2) - (tracking_box[1] + tracking_box[3] / 2))
                if dx > 80 or dy > 80:
                    print("[ALERTE] Saut brutal détecté, recherche ancienne cible dans multi-Kalman...")
                    tid, iou_val = multi_kalman.find_most_similar(last_valid_box, iou_thresh=0.5)
                    if tid is not None:
                        print(
                            f"Retrouvé ancienne cible potentielle (IoU={iou_val:.2f}), switch sur track secondaire")
                        recovered_box = multi_kalman.tracks[tid]["box"]
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, safe_box(recovered_box))
                        tracking_box = recovered_box
                        last_valid_box = recovered_box
                        kalman_box.correct(recovered_box)
                        tracking_status = "RECOVERED"
                    else:
                        print("Impossible de retrouver la cible, on continue en mode projection Kalman")
                        tracking_status = "PERTE"

        tracking_ok, kcf_box = tracker.update(frame) if tracker is not None else (False, None)
        if tracking_ok:
            tracking_box = kcf_box
            last_valid_box = kcf_box
            tracking_status = "OK"
        else:
            tracking_status = "PERTE"
            alerts_tracking_loss += 1
            print(f"[PERTE TRACKING] Frame {frame_count}: KCF perdu !")

        if KALMAN_LISSAGE and tracking_box is not None:
            xk, yk, wk, hk = tracking_box
            kalman_box.correct([xk, yk, wk, hk])
            kalman_pred_box, kalman_pred_vel, kalman_pred_acc = kalman_box.predict()
        else:
            kalman_pred_box, kalman_pred_vel, kalman_pred_acc = None, None, None

        motion_hist.update(frame_count, tracking_box if tracking_box is not None else last_valid_box)
        if last_yolo_box is not None and tracking_box is not None:
            iou_display = iou(last_yolo_box, tracking_box)
        else:
            iou_display = 0.0
        if tracking_box is not None:
            metrics["iou"].append(iou_display)
            metrics["box_w"].append(tracking_box[2])
            metrics["box_h"].append(tracking_box[3])
            metrics["tracker_conf"].append(tracking_confidence)
        if 'global_conf' in locals():
            metrics["global_conf"].append(global_conf)
        metrics["yolo_freq"].append(yolo_freq)

        # Print progression every 25 frames or first
        if frame_count % 25 == 0 or frame_count == 1:
            n_boxes = len(boxes) if boxes is not None else 0
            n_indices = len(indices) if indices is not None else 0
            print(
                f"Frame {frame_count} "
                f"| {n_boxes} boxes avant NMS | {n_indices} boxes après NMS | "
                f"Traitement IA : {(time.perf_counter_ns() - frame_start) / 1e6:.1f} ms"
            )

        # === Affichage sur la vidéo ===
        if SHOW_KCF_BOX and tracking_box is not None:
            x, y, w, h = [int(v) for v in tracking_box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        if SHOW_YOLO_BOX and last_yolo_box is not None:
            x, y, w, h = [int(v) for v in last_yolo_box]
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        if SHOW_INTERP_BOX and last_interp_box is not None:
            x, y, w, h = [int(v) for v in last_interp_box]
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 128, 0), 2)
        if SHOW_KALMAN_BOX and kalman_pred_box is not None:
            xk, yk, wk, hk = [int(v) for v in kalman_pred_box]
            cv2.rectangle(frame, (xk, yk), (xk + wk, yk + hk), (0, 140, 255), 3)
        if SHOW_KALMAN_TRAJ and kalman_pred_box is not None:
            xk, yk, wk, hk = kalman_pred_box
            kalman_center = (int(xk + wk / 2), int(yk + hk / 2))
            kalman_traj.append(kalman_center)
            while len(kalman_traj) > 70:
                kalman_traj.pop(0)
            for i in range(1, len(kalman_traj)):
                alpha = i / len(kalman_traj)
                thickness = int(1 + 3 * (1 - alpha))
                B = int(0 + (220 - 0) * alpha)
                G = int(140 + (220 - 140) * alpha)
                R = int(255 + (220 - 255) * alpha)
                color = (B, G, R)
                cv2.line(frame, kalman_traj[i - 1], kalman_traj[i], color, thickness)

        rect_thickness = 3
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + CROP_SIZE, y_offset + CROP_SIZE),
            (0, 0, 0),
            rect_thickness
        )

        # EXPORT CSV
        if EXPORT_CSV and csv_writer is not None:
            row = [
                frame_count,
                tracking_box if tracking_box is not None else "",
                last_yolo_box if last_yolo_box is not None else "",
                kalman_pred_box if kalman_pred_box is not None else "",
                kalman_pred_vel if kalman_pred_vel is not None else "",
                kalman_pred_acc if kalman_pred_acc is not None else "",
                tracking_status,
                iou_display,
                global_conf if 'global_conf' in locals() else "",
                yolo_delay_frames,
                TRACKER_TYPE,
                int(tracking_ok is False),
                edge_frames
            ]
            csv_writer.writerow(row)

        # EXPORT DEBUG IMAGES
        if EXPORT_DEBUG_IMAGES:
            debug_img_path = os.path.join(OUTPUT_DIR, "debug", f"{basename}_frame_{frame_count:05d}.jpg")
            cv2.imwrite(debug_img_path, frame)

        # ENREGISTREMENT VIDÉO
        if EXPORT_VIDEO and out is not None:
            out.write(frame)

        # LIVE DISPLAY THREAD (retour vidéo direct)
        if display_thread is not None:
            display_thread.submit(frame)

        frame_count += 1
        if tracking_status == "OK" and not replay_applied:
            tracking_only_frames += 1

except KeyboardInterrupt:
    print("\n[INFO] Arrêt manuel détecté (Ctrl+C ou Stop IDE)")
finally:
    cap.release()
    if EXPORT_VIDEO and out is not None:
        out.release()
    yolo_thread.stop()
    yolo_thread.join()
    if EXPORT_CSV and csv_file is not None:
        csv_file.close()
    if display_thread is not None:
        display_thread.stop()
        display_thread.join()
    print("[INFO] Ressources libérées et fichiers correctement fermés.")

    # EXPORT METRICS PLOTS
    if EXPORT_METRICS_PLOTS:
        plt.figure(figsize=(12, 7))
        plt.subplot(221)
        plt.plot(metrics["iou"])
        plt.title("IoU tracker/YOLO")
        plt.subplot(222)
        plt.plot(metrics["box_w"], label="Width")
        plt.plot(metrics["box_h"], label="Height")
        plt.legend()
        plt.title("Box size")
        plt.subplot(223)
        plt.plot(metrics["tracker_conf"])
        plt.title("Tracker confidence")
        plt.subplot(224)
        plt.plot(metrics["yolo_freq"])
        plt.title("YOLO frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{basename}_metrics.png"))
        plt.close()
    if EXPORT_SUMMARY_JSON:
        summary = {
            "video": basename,
            "total_frames": frame_count,
            "yolo_corrections": yolo_corrections,
            "iou_alerts": alerts_iou,
            "tracking_losses": alerts_tracking_loss,
            "frames_only_tracker": tracking_only_frames,
            "panic_mode_count": 0,
        }
        with open(os.path.join(OUTPUT_DIR, f"{basename}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n========== RÉSUMÉ LIVE CAMERA ==========")
    print(f"- Corrections YOLO           : {yolo_corrections}")
    print(f"- Alertes IoU                : {alerts_iou}")
    print(f"- Pertes tracking            : {alerts_tracking_loss}")
    print(f"- Panic mode                 : 0")
    print(f"- Frames uniquement tracking : {tracking_only_frames}")
    print(f"- Log exporté                : {csv_path if EXPORT_CSV else 'non'}")
    print(f"- Résumé JSON                : {basename}_summary.json")
    print(f"Traitement live terminé\n")

print("[INFO] Traitement du flux caméra terminé.")