import numpy as np
from constants import Constants
from state import State
from vector import Vector3D


class CelestialBody:
    """Corps céleste avec propriétés physiques"""

    def __init__(
        self,
        name: str,
        mass: float,
        initial_position: Vector3D,
        initial_velocity: Vector3D,
    ):
        self.name = name
        self.mass = mass
        self.mu = Constants.G * mass  # Paramètre gravitationnel
        self.initial_state = State(initial_position, initial_velocity)

    def get_state_at_time(self, t: float) -> State:
        """
        Retourne l'état du corps à l'instant t
        Pour l'instant, orbite circulaire simplifiée
        """
        # Soleil : fixe à l'origine
        if self.name == "Sun":
            return self.initial_state

        # Terre : orbite circulaire dans le plan (x,y)
        if self.name == "Earth":
            angle = Constants.OMEGA * t
            r = Constants.R_EARTH_ORBIT

            position = Vector3D(r * np.cos(angle), r * np.sin(angle), 0.0)

            velocity = Vector3D(
                -Constants.V_EARTH * np.sin(angle),
                Constants.V_EARTH * np.cos(angle),
                0.0,
            )

            return State(position, velocity)

        return self.initial_state
