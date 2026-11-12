# Demo completa: tu cuerpo mueve el G1 en MuJoCo.
# Cámara -> PoseTracker -> ángulos -> suavizado -> retargeting -> G1Robot.

import cv2
import mujoco.viewer

from src.retargeting.angles import compute_joint_angles
from src.retargeting.g1_mapping import retarget
from src.retargeting.smoothing import AngleSmoother
from src.simulation.g1_robot import G1Robot
from src.tracking.pose_tracker import PoseTracker
from src.visualization.drawing import draw_angles, draw_skeleton

# Las tres piezas principales del sistema
tracker = PoseTracker()                # detecta tu cuerpo
robot = G1Robot()                      # el robot simulado
smoother = AngleSmoother(alpha=0.1)    # quita el temblor de los ángulos

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Abrimos el visor de MuJoCo. Mientras esté abierto, repetimos el ciclo.
with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
    while viewer.is_running():
        success, frame = cap.read()
        if not success:
            print("No se pudo leer el frame de la cámara")
            break

        pose = tracker.process(frame)

        if pose is not None:
            # 1) Ángulos de tu cuerpo, calculados con los puntos 3D y suavizados
            human_angles = smoother.update(compute_joint_angles(pose.world_landmarks))

            # 2) Dibujamos tu esqueleto y los ángulos en la ventana de la cámara
            draw_skeleton(frame, pose.image_landmarks)
            draw_angles(frame, pose.image_landmarks, human_angles)

            # 3) Traducimos tus ángulos a ángulos del robot
            robot_angles = retarget(human_angles)

            # 4) Movemos el robot. lock() evita que el visor lea los datos
            # justo mientras los estamos cambiando.
            with viewer.lock():
                for joint_name, degrees in robot_angles.items():
                    robot.set_joint_angle(joint_name, degrees)
                robot.update()

        # Refresca la ventana del robot con el último estado
        viewer.sync()

        # Muestra la cámara y permite salir con 'q'
        cv2.imshow("Camara", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

tracker.close()
cap.release()
cv2.destroyAllWindows()