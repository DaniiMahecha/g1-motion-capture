# Suavizado de ángulos para que no tiemblen de un frame a otro.
# Esto importa mucho para MuJoCo: un robot que recibe valores que saltan
# se vería sacudiéndose en vez de moverse con fluidez.


class AngleSmoother:
    """Suaviza un conjunto de ángulos con una media móvil exponencial."""

    def __init__(self, alpha=0.3):
        # alpha entre 0 y 1:
        #   más alto -> responde rápido pero tiembla más
        #   más bajo -> más estable pero con un poco de retraso
        self.alpha = alpha

        # Aquí guardamos el último valor suavizado de cada articulación
        # Ejemplo: {"codo_izq": 142.5, "codo_der": 90.1}
        self._previous = {}

    def update(self, angles):
        """Recibe los ángulos del frame actual y devuelve los suavizados.

        angles -> diccionario como {"codo_izq": 172.3, "codo_der": None, ...}
        """
        smoothed = {}

        for name, value in angles.items():
            # Si el ángulo no es confiable (None), lo dejamos como None y
            # olvidamos el valor anterior. Así, cuando la articulación vuelva
            # a verse, empieza desde el valor real y no "arrastra" uno viejo.
            if value is None:
                smoothed[name] = None
                self._previous.pop(name, None)
                continue

            if name not in self._previous:
                # Primer valor de esta articulación: no hay nada que mezclar
                smoothed[name] = value
            else:
                # Mezcla del valor nuevo con el anterior
                smoothed[name] = self.alpha * value + (1 - self.alpha) * self._previous[name]

            # Guardamos el resultado para el siguiente frame
            self._previous[name] = smoothed[name]

        return smoothed