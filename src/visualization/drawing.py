# Funciones para dibujar el esqueleto sobre la imagen.
# Este archivo SÍ usa OpenCV, porque dibujar es su trabajo.
# Los datos (qué puntos forman cada parte) vienen de body_parts.py.

import cv2

from src.tracking.body_parts import BODY_PARTS, MIN_VISIBILITY, SKELETON

# Color de cada parte, en BGR (el orden que usa OpenCV, no RGB).
# Los nombres deben coincidir con los de BODY_PARTS y SKELETON.
PART_COLORS = {
    "cabeza":     (0, 255, 255),   # amarillo
    "torso":      (0, 255, 0),     # verde
    "brazo_izq":  (255, 0, 0),     # azul
    "brazo_der":  (0, 0, 255),     # rojo
    "pierna_izq": (255, 0, 255),   # magenta
    "pierna_der": (255, 255, 0),   # cian
}


def draw_skeleton(frame, landmarks):
    """Dibuja líneas, puntos y pecho sobre el frame.

    frame     -> la imagen de la cámara (se modifica directamente)
    landmarks -> los 33 puntos de imagen (PoseResult.image_landmarks)
    """
    # Alto y ancho de la imagen en píxeles
    h, w = frame.shape[:2]

    # 1) Líneas (huesos). Van primero para que los puntos queden encima.
    for part_name, connections in SKELETON.items():
        color = PART_COLORS[part_name]

        for a, b in connections:
            p1, p2 = landmarks[a], landmarks[b]

            # Si algún extremo es poco confiable, no dibujamos esa línea
            if p1.visibility < MIN_VISIBILITY or p2.visibility < MIN_VISIBILITY:
                continue

            # x e y vienen entre 0 y 1; los multiplicamos para pasar a píxeles
            pt1 = (int(p1.x * w), int(p1.y * h))
            pt2 = (int(p2.x * w), int(p2.y * h))
            cv2.line(frame, pt1, pt2, color, 2)

    # 2) Puntos (articulaciones)
    for part_name, ids in BODY_PARTS.items():
        color = PART_COLORS[part_name]

        for idx in ids:
            point = landmarks[idx]
            if point.visibility < MIN_VISIBILITY:
                continue

            center = (int(point.x * w), int(point.y * h))
            cv2.circle(frame, center, 5, color, -1)  # -1 = círculo relleno

    # 3) Pecho: MediaPipe no lo trae, así que lo calculamos como el punto
    # medio entre el hombro izquierdo (11) y el derecho (12).
    left_sh, right_sh = landmarks[11], landmarks[12]
    if left_sh.visibility > MIN_VISIBILITY and right_sh.visibility > MIN_VISIBILITY:
        chest = (
            int((left_sh.x + right_sh.x) / 2 * w),
            int((left_sh.y + right_sh.y) / 2 * h),
        )
        cv2.circle(frame, chest, 8, (255, 255, 255), -1)  # blanco y más grande


def draw_fps(frame, fps):
    """Escribe el texto de FPS en la esquina superior izquierda."""
    cv2.putText(
        frame,
        f"FPS: {fps:.0f}",        # :.0f = número sin decimales
        (10, 30),                 # posición (x, y) en píxeles
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,                      # tamaño del texto
        (255, 255, 255),          # color blanco
        2,                        # grosor
    )