# Visor del robot G1 en MuJoCo.
# Objetivo de este paso: ver al robot en pantalla y listar sus articulaciones,
# para saber cómo se llaman cuando conectemos tu cuerpo con el robot.

import time

import mujoco  # el motor de física y simulación
import mujoco.viewer  # la ventana 3D para ver el robot
import mujoco_menagerie as mm  # descarga y carga modelos de robots listos

# Carga la escena del G1 (robot + suelo + luces).
# La primera vez descarga los archivos, así que puede tardar un poco.
# "model" describe el robot: sus huesos, articulaciones, límites, etc.
model = mm.load("unitree_g1")

# "data" guarda el ESTADO actual del robot: el ángulo de cada articulación,
# posiciones, velocidades. El modelo no cambia; los datos sí.
data = mujoco.MjData(model)

# Si el modelo trae una pose inicial guardada ("keyframe"), la usamos.
# Si no, se queda en la pose por defecto.
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

# --- Lista de articulaciones ---------------------------------------------
# Cada articulación (joint) es algo que puede girar o moverse: codo, rodilla, etc.
print(f"\nEl modelo tiene {model.njnt} articulaciones:\n")

for i in range(model.njnt):
    # Nombre de la articulación. La base flotante del robot no tiene nombre.
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "(base flotante)"

    # Rango de movimiento permitido, pasado de radianes a grados
    low, high = model.jnt_range[i] * 180 / 3.14159265

    print(f"{i:2d}  {name:35s}  rango: {low:7.1f} a {high:7.1f} grados")

# --- Ventana del visor ----------------------------------------------------
# launch_passive abre la ventana 3D y nos deja el control del programa.
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Mientras la ventana esté abierta...
    while viewer.is_running():
        # mj_forward recalcula las posiciones del robot a partir de sus ángulos,
        # SIN simular física. Por eso el robot no se cae: lo vamos a mover
        # nosotros escribiendo los ángulos directamente en data.qpos.
        mujoco.mj_forward(model, data)

        # sync actualiza la ventana con lo que hay en "data"
        viewer.sync()

        # Pausa corta para no gastar CPU de más (unos 60 cuadros por segundo)
        time.sleep(1 / 60)