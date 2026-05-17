"""
body.py — Représentation d'un corps céleste ou d'un spacecraft.
"""

import numpy as np


class Body:
    """
    Corps physique caractérisé par sa masse, sa position et sa vitesse.

    Les unités sont libres mais doivent être cohérentes dans tout le système :
    - module inertiel  : SI (kg, m, m/s)
    - module CR3BP     : unités non-dimensionnelles

    Parameters
    ----------
    name : str
    mass : float
        Masse [kg ou adim].
    position : array_like, shape (3,)
        Vecteur position initial [m ou adim].
    velocity : array_like, shape (3,)
        Vecteur vitesse initial [m/s ou adim].
    fixed : bool
        Si True, le corps n'est pas intégré (Soleil, Terre dans le CR3BP).
    """

    def __init__(
        self,
        name: str,
        mass: float,
        position: np.ndarray,
        velocity: np.ndarray,
        fixed: bool = False,
    ):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.fixed = fixed

        # Historiques accumulés lors de l'intégration
        self._pos_history: list[np.ndarray] = []
        self._vel_history: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # État courant
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Vecteur d'état [x, y, z, vx, vy, vz]."""
        return np.concatenate([self.position, self.velocity])

    @state.setter
    def state(self, s: np.ndarray):
        self.position = s[:3].copy()
        self.velocity = s[3:].copy()

    # ------------------------------------------------------------------
    # Gestion de l'historique
    # ------------------------------------------------------------------

    def record(self):
        """Enregistre l'état courant dans l'historique."""
        self._pos_history.append(self.position.copy())
        self._vel_history.append(self.velocity.copy())

    def clear_history(self):
        self._pos_history.clear()
        self._vel_history.clear()

    @property
    def pos_history(self) -> np.ndarray:
        """shape (N, 3)"""
        return np.array(self._pos_history)

    @property
    def vel_history(self) -> np.ndarray:
        """shape (N, 3)"""
        return np.array(self._vel_history)

    # ------------------------------------------------------------------

    def __repr__(self):
        return f"Body('{self.name}', mass={self.mass:.3e}, fixed={self.fixed})"
