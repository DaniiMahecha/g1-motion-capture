# Prueba: mueve el codo izquierdo del G1 de punta a punta del rango.
# Sirve para descubrir qué ángulo del robot significa "brazo estirado"
# y hacia qué lado dobla.

import math
import time

import mujoco.viewer

from src.simulation.g1_robot import G1Robot

robot = G1Robot()

with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
    start = time.time()
    last_print = -1

    while viewer.is_running():
        t = time.time() - start

        # Un seno oscila suave entre -1 y 1. Lo escalamos para que el ángulo
        # vaya de -60 a 120 grados: centro = 30, amplitud = 90.
        # El ciclo completo dura 10 segundos.
        angle = 30 + 90 * math.sin(2 * math.pi * t / 10)

        # lock() evita que el visor lea los datos justo mientras los cambiamos
        with viewer.lock():
            robot.set_joint_angle("left_elbow_joint", angle)
            robot.update()

        viewer.sync()

        # Imprimimos el ángulo cada medio segundo (cada frame sería demasiado)
        if int(t * 2) != last_print:
            last_print = int(t * 2)
            print(f"left_elbow_joint = {angle:6.1f} grados")

        time.sleep(1 / 60)