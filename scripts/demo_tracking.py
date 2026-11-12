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


# --- Definición de partes del cuerpo ----------------------------------------
# MediaPipe numera los 33 puntos del 0 al 32. Aquí los agrupamos por parte.
# Cada punto pertenece a UN solo grupo, así no se dibuja dos veces.
# Los colores están en BGR (el orden de OpenCV), no en RGB.
BODY_PARTS = {
    "cabeza":     {"ids": list(range(0, 11)),       "color": (0, 255, 255)},  # amarillo
    "torso":      {"ids": [11, 12, 23, 24],         "color": (0, 255, 0)},    # verde (hombros y caderas)
    "brazo_izq":  {"ids": [13, 15, 17, 19, 21],     "color": (255, 0, 0)},    # azul (codo, muñeca, mano)
    "brazo_der":  {"ids": [14, 16, 18, 20, 22],     "color": (0, 0, 255)},    # rojo
    "pierna_izq": {"ids": [25, 27, 29, 31],         "color": (255, 0, 255)},  # magenta (rodilla, tobillo, pie)
    "pierna_der": {"ids": [26, 28, 30, 32],         "color": (255, 255, 0)},  # cian
}

# Si MediaPipe no está seguro de que un punto es visible (por ejemplo, unas
# piernas fuera de cámara), su "visibility" es baja y mejor no lo dibujamos.
MIN_VISIBILITY = 0.5

# Cada tupla (a, b) es una línea que une el punto a con el punto b.
# Las agrupamos con los mismos nombres de BODY_PARTS para reusar sus colores.
SKELETON = {
    "cabeza":     [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)],
    "torso":      [(11, 12), (11, 23), (12, 24), (23, 24)],  # rectángulo hombros-caderas
    "brazo_izq":  [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)],
    "brazo_der":  [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)],
    "pierna_izq": [(23, 25), (25, 27), (27, 29), (27, 31), (29, 31)],
    "pierna_der": [(24, 26), (26, 28), (28, 30), (28, 32), (30, 32)],
}

# --- Cámara -----------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

start_time = time.time()  # referencia para calcular los timestamps
prev_time = time.time()  # momento del frame anterior, para medir cuánto tarda cada frame

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

    # Si detectó a alguien, dibujamos su esqueleto.
    if result.pose_landmarks:
        pose = result.pose_landmarks[0]  # la primera (y única) persona

        # Las coordenadas x e y vienen NORMALIZADAS (entre 0 y 1).
        # Para dibujar en píxeles las multiplicamos por el ancho y el alto.
        h, w = frame.shape[:2]

        # 1) Líneas: las dibujamos ANTES que los puntos, así los círculos
        # quedan encima y se ven más limpios.
        for part_name, connections in SKELETON.items():
            color = BODY_PARTS[part_name]["color"]  # mismo color que los puntos de esa parte

            for a, b in connections:
                p1, p2 = pose[a], pose[b]

                # Solo dibujamos la línea si AMBOS extremos son visibles;
                # si no, saldrían líneas hacia puntos que MediaPipe está adivinando.
                if p1.visibility < MIN_VISIBILITY or p2.visibility < MIN_VISIBILITY:
                    continue

                # Pasamos de coordenadas normalizadas (0 a 1) a píxeles
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))

                # cv2.line(imagen, punto_inicial, punto_final, color, grosor)
                cv2.line(frame, pt1, pt2, color, 2)

        # 2) Puntos: recorremos cada parte del cuerpo con su lista de puntos y su color
        for part in BODY_PARTS.values():
            for idx in part["ids"]:
                landmark = pose[idx]  # el punto número idx de los 33

                # Ignoramos los puntos que MediaPipe considera poco visibles
                if landmark.visibility < MIN_VISIBILITY:
                    continue

                # x e y vienen normalizados (0 a 1): los pasamos a píxeles
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, part["color"], -1)

        # 3) Pecho: MediaPipe no tiene un punto de "pecho", así que lo calculamos
        # como el punto medio entre el hombro izquierdo (11) y el derecho (12).
        left_sh, right_sh = pose[11], pose[12]
        # Solo lo calculamos si ambos hombros son confiables
        if left_sh.visibility > MIN_VISIBILITY and right_sh.visibility > MIN_VISIBILITY:
            chest_x = int((left_sh.x + right_sh.x) / 2 * w)
            chest_y = int((left_sh.y + right_sh.y) / 2 * h)
            # Círculo blanco más grande para distinguirlo de los demás
            cv2.circle(frame, (chest_x, chest_y), 8, (255, 255, 255), -1)

    # FPS = 1 / (segundos que pasaron desde el frame anterior)
    # Está fuera del "if", así se muestra aunque no haya nadie en cámara.
    now = time.time()
    fps = 1 / max(now - prev_time, 1e-6)  # max() evita dividir entre 0
    prev_time = now

    # cv2.putText(imagen, texto, posición, fuente, tamaño, color, grosor)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Demo tracking", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Limpieza ---------------------------------------------------------------
landmarker.close()  # libera los recursos del detector
cap.release()
cv2.destroyAllWindows()