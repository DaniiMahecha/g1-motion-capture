# Demo de tracking: cámara -> PoseTracker -> dibujo -> ventana.
# Aquí solo se "conectan" las piezas; la lógica vive en src/.

import time
from collections import deque  # lista con tamaño máximo para promediar los FPS

import cv2

from src.tracking.pose_tracker import PoseTracker
from src.visualization.drawing import draw_fps, draw_skeleton

# Creamos el tracker (carga el modelo de MediaPipe)
tracker = PoseTracker()

# Abrimos la cámara. 0 = la cámara por defecto
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

prev_time = time.time()
fps_history = deque(maxlen=30)  # últimos 30 valores de FPS, para promediarlos

while True:
    success, frame = cap.read()
    if not success:
        print("No se pudo leer el frame de la cámara")
        break

    # Detectamos el cuerpo. Devuelve un PoseResult, o None si no hay nadie.
    pose = tracker.process(frame)

    # "is not None" = "si se detectó a alguien"
    if pose is not None:
        draw_skeleton(frame, pose.image_landmarks)

    # Cálculo de FPS suavizado
    now = time.time()
    fps_history.append(1 / max(now - prev_time, 1e-6))
    prev_time = now
    draw_fps(frame, sum(fps_history) / len(fps_history))

    cv2.imshow("Demo tracking", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza al terminar
tracker.close()
cap.release()
cv2.destroyAllWindows()