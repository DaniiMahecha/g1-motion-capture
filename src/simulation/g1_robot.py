# Envuelve el modelo del G1 en MuJoCo.
# El resto del proyecto solo necesita decir "pon esta articulación en X grados",
# sin saber cómo guarda MuJoCo los datos por dentro.

import mujoco
import mujoco_menagerie as mm
import numpy as np


class G1Robot:
    """El robot G1 simulado: carga el modelo y permite mover sus articulaciones."""

    def __init__(self):
        # "model" = descripción fija del robot (huesos, articulaciones, límites)
        self.model = mm.load("unitree_g1")

        # "data" = estado actual del robot (el ángulo de cada articulación, etc.)
        self.data = mujoco.MjData(self.model)

        # Si el modelo trae una pose inicial guardada, empezamos con ella
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # MuJoCo guarda todos los ángulos en una sola lista larga (data.qpos).
        # Aquí armamos dos "agendas" para encontrar cada articulación por nombre:
        #   _qpos_index -> en qué posición de esa lista está cada articulación
        #   _limits     -> su rango permitido (en radianes)
        self._qpos_index = {}
        self._limits = {}

        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is None:
                continue
            self._qpos_index[name] = self.model.jnt_qposadr[i]
            self._limits[name] = self.model.jnt_range[i].copy()

    def set_joint_angle(self, name, degrees):
        """Pone una articulación en un ángulo (en grados).

        name    -> nombre exacto, por ejemplo "left_elbow_joint"
        degrees -> ángulo deseado. Si se pasa del rango permitido, se recorta.
        """
        low, high = self._limits[name]

        # MuJoCo trabaja en radianes, así que convertimos.
        # np.clip recorta el valor para que nunca salga de los límites del robot.
        radians = np.clip(np.radians(degrees), low, high)

        self.data.qpos[self._qpos_index[name]] = radians

    def update(self):
        """Recalcula las posiciones del robot a partir de los ángulos puestos.

        Hay que llamarla después de cambiar ángulos para que el robot se mueva.
        No simula física (el robot no se cae): solo coloca el cuerpo según los ángulos.
        """
        mujoco.mj_forward(self.model, self.data)