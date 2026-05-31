"""
body.py — Représentation d'un corps céleste ou d'un spacecraft.
"""

import numpy as np
from numpy.typing import NDArray


class Body:
    """
    Corps physique caractérisé par sa masse, sa position et sa vitesse.

    Les unités sont libres mais doivent être cohérentes dans tout le système :
    - module inertiel  : SI (kg, m, m/s)
    - module CR3BP     : unités non-dimensionnelles

    Parameters
    ----------
    name : str
    mass : np.float64
        Masse [kg ou adim].
    position : NDArray[np.float64], shape (3,)
        Vecteur position initial [m ou adim].
    velocity : NDArray[np.float64], shape (3,)
        Vecteur vitesse initial [m/s ou adim].
    fixed : bool
        Si True, le corps n'est pas intégré (Soleil, Terre dans le CR3BP).
    """

    def __init__(
        self,
        name: str,
        mass: np.float64,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        fixed: bool = False,
    ):
        self.name = name
        self.mass = np.float64(mass)
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.fixed = fixed

        # Historiques accumulés lors de l'intégration
        self._pos_history: list[NDArray[np.float64]] = []
        self._vel_history: list[NDArray[np.float64]] = []

    # État courant

    @property
    def state(self) -> NDArray[np.float64]:
        """Vecteur d'état [x, y, z, vx, vy, vz]."""
        return np.concatenate([self.position, self.velocity], dtype=np.float64)

    @state.setter
    def state(self, s: NDArray[np.float64]):
        self.position = np.array(s[:3], dtype=np.float64, copy=True)
        self.velocity = np.array(s[3:], dtype=np.float64, copy=True)

    # Gestion de l'historique
    def record(self):
        """Enregistre l'état courant dans l'historique."""
        self._pos_history.append(self.position.copy())
        self._vel_history.append(self.velocity.copy())

    def clear_history(self):
        self._pos_history.clear()
        self._vel_history.clear()

    @property
    def pos_history(self) -> NDArray[np.float64]:
        """shape (N, 3)"""
        return np.array(self._pos_history, np.float64)

    @property
    def vel_history(self) -> NDArray[np.float64]:
        """shape (N, 3)"""
        return np.array(self._vel_history, np.float64)

    def __repr__(self):
        return f"Body('{self.name}', mass={self.mass:.3e}, fixed={self.fixed})"
