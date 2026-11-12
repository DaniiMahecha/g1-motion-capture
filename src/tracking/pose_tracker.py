# Este archivo es el "traductor" entre MediaPipe y el resto del proyecto.
# Recibe una imagen de la cámara y devuelve los puntos del cuerpo.
# Nadie más en el proyecto necesita saber cómo funciona MediaPipe por dentro.

import time
from dataclasses import dataclass  # atajo de Python para crear "cajitas" de datos

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# Una "clase" es un molde para agrupar datos relacionados.
# @dataclass hace que Python arme el molde solo, con solo listar los campos.
@dataclass
class PoseResult:
    """Lo que devuelve el tracker por cada frame donde detecta a una persona."""

    # 33 puntos con x, y normalizados (0 a 1 sobre la imagen). Sirven para DIBUJAR.
    image_landmarks: list

    # 33 puntos en 3D, en metros, con origen en la cadera. Son los que usaremos
    # más adelante para mover el robot G1 en MuJoCo.
    world_landmarks: list


class PoseTracker:
    """Envuelve el Pose Landmarker de MediaPipe en algo fácil de usar."""

    # __init__ se ejecuta una sola vez, al crear el tracker: prepara todo.
    # "self" es el propio tracker; lo que guardas en self lo recuerda mientras exista.
    def __init__(self, model_path="models/pose_landmarker_lite.task"):
        # Le decimos a MediaPipe dónde está el archivo del modelo
        base_options = mp_python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # procesamos video en secuencia
            num_poses=1,                            # solo una persona
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        # Referencias de tiempo para los timestamps que exige el modo VIDEO
        self._start_time = time.time()
        self._last_timestamp_ms = -1

    def process(self, frame):
        """Recibe un frame BGR de OpenCV. Devuelve un PoseResult, o None si no hay nadie."""

        # OpenCV usa BGR y MediaPipe espera RGB, así que convertimos
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp en milisegundos. MediaPipe exige que SIEMPRE aumente.
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            # Si dos frames caen en el mismo milisegundo, forzamos que sea mayor
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        # Aquí ocurre la detección
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        # Si no detectó a nadie, devolvemos None (que significa "nada")
        if not result.pose_landmarks:
            return None

        # [0] = la primera persona detectada (nosotros solo pedimos una)
        return PoseResult(
            image_landmarks=result.pose_landmarks[0],
            world_landmarks=result.pose_world_landmarks[0],
        )

    def close(self):
        """Libera los recursos de MediaPipe. Se llama al terminar el programa."""
        self._landmarker.close()