# Convierte los ángulos de TU cuerpo (MediaPipe) en ángulos del robot G1.
# Esta es la pieza central del retargeting: cada robot tiene su propia escala,
# así que aquí vive la "traducción" entre tu cuerpo y el suyo.

import numpy as np

# --- Calibración del codo -----------------------------------------------------
# Tres puntos de referencia medidos a ojo con el visor:
#   tu codo muy doblado (40°)  -> codo del G1 en -60°
#   tu codo en escuadra (90°)  -> codo del G1 en 0°
#   tu codo estirado (170°)    -> codo del G1 en 60°
# Si algún día el robot se ve raro, estos números son los que hay que ajustar.
ELBOW_HUMAN = [40, 90, 170]   # ángulos de MediaPipe (deben ir de menor a mayor)
ELBOW_ROBOT = [-60, 0, 60]    # ángulos equivalentes del robot

# Qué articulación humana controla qué articulación del robot.
# Izquierda y derecha son las de la PERSONA, igual que en MediaPipe.
JOINT_MAP = {
    "codo_izq": "left_elbow_joint",
    "codo_der": "right_elbow_joint",
}


def human_elbow_to_g1(human_angle):
    """Convierte el ángulo de un codo humano en el ángulo del codo del G1.

    np.interp busca entre qué dos puntos de la tabla cae el valor y calcula
    el intermedio en línea recta. Si el valor queda fuera de la tabla,
    devuelve el extremo más cercano, así nunca pedimos algo imposible.
    """
    return float(np.interp(human_angle, ELBOW_HUMAN, ELBOW_ROBOT))


def retarget(human_angles):
    """Recibe los ángulos humanos y devuelve los ángulos para el robot.

    human_angles -> diccionario como {"codo_izq": 142.0, "codo_der": None, ...}
    Devuelve     -> diccionario como {"left_elbow_joint": 38.5, ...}
                    (solo incluye las articulaciones que se pudieron calcular)
    """
    robot_angles = {}

    for human_name, robot_name in JOINT_MAP.items():
        value = human_angles.get(human_name)

        # Si el ángulo no es confiable (None), no movemos esa articulación:
        # el robot se queda como estaba en vez de dar un salto raro.
        if value is None:
            continue

        robot_angles[robot_name] = human_elbow_to_g1(value)

    return robot_angles