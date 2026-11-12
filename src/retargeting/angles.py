# Cálculo de ángulos de articulación a partir de los landmarks 3D de MediaPipe.
# Este archivo no dibuja nada ni usa la cámara: solo hace matemática.
# Es la base del retargeting: más adelante estos ángulos moverán el G1 en MuJoCo.

import numpy as np  # librería de matemática con vectores (ya viene con OpenCV)

from src.tracking.body_parts import MIN_VISIBILITY

# Cada articulación se define con 3 puntos (a, b, c), donde b es la articulación:
#   codo   = hombro (a) -> codo (b) -> muñeca (c)
#   rodilla = cadera (a) -> rodilla (b) -> tobillo (c)
# Los números son los índices de landmark de MediaPipe.
JOINT_ANGLES = {
    "codo_izq":    (11, 13, 15),
    "codo_der":    (12, 14, 16),
    "rodilla_izq": (23, 25, 27),
    "rodilla_der": (24, 26, 28),
}


def calculate_angle(a, b, c):
    """Devuelve el ángulo en grados (0 a 180) en el punto b, formado por a-b-c.

    a, b, c -> landmarks de MediaPipe (cada uno con x, y, z)
    """
    # Convertimos cada landmark en un vector de NumPy con sus 3 coordenadas
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    # Dos vectores que salen de la articulación (b): uno hacia a y otro hacia c
    ba = a - b
    bc = c - b

    # Fórmula del ángulo entre dos vectores:
    #   cos(ángulo) = (ba · bc) / (|ba| * |bc|)
    # Se le suma 1e-9 al denominador para no dividir entre cero.
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)

    # np.clip evita que errores de redondeo dejen el valor fuera de [-1, 1],
    # lo que haría fallar a arccos.
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # arccos da el ángulo en radianes; degrees lo pasa a grados
    return float(np.degrees(np.arccos(cos_angle)))


def compute_joint_angles(world_landmarks):
    """Calcula todos los ángulos de JOINT_ANGLES.

    Devuelve un diccionario como {"codo_izq": 172.3, "codo_der": None, ...}
    Si algún punto de la articulación no es visible, el valor es None.
    """
    angles = {}

    for name, (i_a, i_b, i_c) in JOINT_ANGLES.items():
        a, b, c = world_landmarks[i_a], world_landmarks[i_b], world_landmarks[i_c]

        # Si MediaPipe no ve bien alguno de los 3 puntos, el ángulo no es confiable
        if min(a.visibility, b.visibility, c.visibility) < MIN_VISIBILITY:
            angles[name] = None
        else:
            angles[name] = calculate_angle(a, b, c)

    return angles