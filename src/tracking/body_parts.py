# Definición de las partes del cuerpo sobre los 33 landmarks de MediaPipe Pose.
# Este archivo es SOLO DATOS: no importa cv2 ni mediapipe, así lo puede
# reusar cualquier módulo (visualización, retargeting a MuJoCo, etc.).

# Umbral de confianza: por debajo de esto, un punto se considera "no visible"
MIN_VISIBILITY = 0.5

# Qué índices de landmark (0 a 32) pertenecen a cada parte del cuerpo.
# Cada punto pertenece a UN solo grupo.
BODY_PARTS = {
    "cabeza":     list(range(0, 11)),
    "torso":      [11, 12, 23, 24],         # hombros y caderas
    "brazo_izq":  [13, 15, 17, 19, 21],     # codo, muñeca, mano
    "brazo_der":  [14, 16, 18, 20, 22],
    "pierna_izq": [25, 27, 29, 31],         # rodilla, tobillo, pie
    "pierna_der": [26, 28, 30, 32],
}

# Cada tupla (a, b) es un hueso: une el landmark a con el landmark b.
# Usa los mismos nombres de parte que BODY_PARTS.
SKELETON = {
    "cabeza":     [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)],
    "torso":      [(11, 12), (11, 23), (12, 24), (23, 24)],
    "brazo_izq":  [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)],
    "brazo_der":  [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)],
    "pierna_izq": [(23, 25), (25, 27), (27, 29), (27, 31), (29, 31)],
    "pierna_der": [(24, 26), (26, 28), (28, 30), (28, 32), (30, 32)],
}