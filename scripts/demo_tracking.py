import time  # para generar los timestamps que MediaPipe exige en modo video

import cv2  # OpenCV: captura y dibujo
import mediapipe as mp  # MediaPipe: aquí lo usamos para crear la imagen de entrada
from mediapipe.tasks import python as mp_python  # opciones base (ruta del modelo)
from mediapipe.tasks.python import vision  # el Pose Landmarker está aquí

# --- Configuración del detector ---------------------------------------------
# BaseOptions le dice a MediaPipe dónde está el archivo del modelo.
base_options = mp_python.BaseOptions(model_asset_path="models/pose_landmarker_lite.task")

# Opciones del detector:
#   running_mode=VIDEO -> procesamos frames de un video/cámara en secuencia.
#                         En este modo MediaPipe aprovecha el frame anterior para
#                         seguir a la persona, así es más estable y rápido.
#   num_poses=1        -> solo nos interesa una persona.
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
)

# Crea el detector con esas opciones.
landmarker = vision.PoseLandmarker.create_from_options(options)

# --- Cámara -----------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

start_time = time.time()  # referencia para calcular los timestamps

while True:
    success, frame = cap.read()
    if not success:
        print("No se pudo leer el frame de la cámara")
        break

    # OpenCV entrega las imágenes en BGR, pero MediaPipe espera RGB.
    # Si no conviertes, los colores se invierten y la detección empeora.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Empaquetamos el array de NumPy en el formato de imagen de MediaPipe.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # En modo VIDEO, cada frame necesita un timestamp en milisegundos que
    # SIEMPRE vaya creciendo. Lo calculamos con el tiempo transcurrido.
    timestamp_ms = int((time.time() - start_time) * 1000)

    # ¡Aquí ocurre la detección!
    # result.pose_landmarks es una lista con una pose por persona detectada.
    # Cada pose es una lista de 33 landmarks con x, y, z y visibility.
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Si detectó a alguien, dibujamos sus puntos.
    if result.pose_landmarks:
        pose = result.pose_landmarks[0]  # la primera (y única) persona

        # Las coordenadas x e y vienen NORMALIZADAS (entre 0 y 1).
        # Para dibujar en píxeles las multiplicamos por el ancho y el alto.
        h, w = frame.shape[:2]

        for landmark in pose:
            cx = int(landmark.x * w)
            cy = int(landmark.y * h)
            # Círculo verde (BGR) de radio 4, relleno (-1)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    cv2.imshow("Demo tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Limpieza ---------------------------------------------------------------
landmarker.close()  # libera los recursos del detector
cap.release()
cv2.destroyAllWindows()