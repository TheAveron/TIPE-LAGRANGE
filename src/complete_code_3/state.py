from dataclasses import dataclass

import numpy as np
from vector import Vector3D


@dataclass
class State:
    """État d'un objet : position et vitesse"""

    position: Vector3D
    velocity: Vector3D

    def to_array(self) -> np.ndarray:
        """Convertit en array numpy [x, y, z, vx, vy, vz]"""
        return np.concatenate([self.position.to_array(), self.velocity.to_array()])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Crée un State depuis un array numpy"""
        return cls(
            position=Vector3D.from_array(arr[0:3]),
            velocity=Vector3D.from_array(arr[3:6]),
        )

    def copy(self):
        """Copie profonde de l'état"""
        return State(
            position=Vector3D(self.position.x, self.position.y, self.position.z),
            velocity=Vector3D(self.velocity.x, self.velocity.y, self.velocity.z),
        )
