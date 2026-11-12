# Demo de tracking: cámara -> PoseTracker -> ángulos -> dibujo -> ventana.

import time
from collections import deque

import cv2

from src.retargeting.angles import compute_joint_angles
from src.tracking.pose_tracker import PoseTracker
from src.visualization.drawing import draw_angles, draw_fps, draw_skeleton

tracker = PoseTracker()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

prev_time = time.time()
fps_history = deque(maxlen=30)

while True:
    success, frame = cap.read()
    if not success:
        print("No se pudo leer el frame de la cámara")
        break

    pose = tracker.process(frame)

    if pose is not None:
        # Ángulos calculados con los puntos 3D (en metros)
        angles = compute_joint_angles(pose.world_landmarks)

        # Dibujo con los puntos de imagen (en píxeles)
        draw_skeleton(frame, pose.image_landmarks)
        draw_angles(frame, pose.image_landmarks, angles)

    now = time.time()
    fps_history.append(1 / max(now - prev_time, 1e-6))
    prev_time = now
    draw_fps(frame, sum(fps_history) / len(fps_history))

    cv2.imshow("Demo tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.close()
cap.release()
cv2.destroyAllWindows()