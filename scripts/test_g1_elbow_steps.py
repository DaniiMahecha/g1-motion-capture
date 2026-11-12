# Calibración: mantiene el codo izquierdo del G1 en ángulos fijos, uno por uno.
# Así puedes mirar con calma qué número corresponde a "brazo estirado",
# "codo en 90 grados", etc.

import time

import mujoco.viewer

from src.simulation.g1_robot import G1Robot

robot = G1Robot()

# Ángulos que vamos a probar, en grados (el rango del codo va de -60 a 120)
ANGLES = [-60, -30, 0, 30, 60, 90, 120]

# Segundos que se queda quieto en cada ángulo
HOLD_SECONDS = 4

with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
    start = time.time()
    last_index = -1

    while viewer.is_running():
        # Cuántos segundos han pasado
        t = time.time() - start

        # Qué ángulo de la lista toca ahora. El "%" hace que, al terminar
        # la lista, vuelva a empezar desde el primero.
        index = int(t // HOLD_SECONDS) % len(ANGLES)
        angle = ANGLES[index]

        with viewer.lock():
            robot.set_joint_angle("left_elbow_joint", angle)
            robot.update()

        viewer.sync()

        # Solo imprimimos cuando cambia el ángulo, no en cada frame
        if index != last_index:
            last_index = index
            print(f"Ahora: left_elbow_joint = {angle} grados")

        time.sleep(1 / 60)