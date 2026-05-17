"""
system.py — Conteneur de corps célestes et historique temporel.

Le System ne contient pas la logique physique (forces, équations de mouvement) :
celle-ci est déléguée aux modules inertial/ et cr3bp/.
Le System est uniquement responsable de :
  - stocker les corps
  - stocker le vecteur temps
  - fournir des accesseurs communs
"""

import numpy as np

from .body import Body


class System:
    """
    Ensemble de corps physiques constituant la simulation.

    Parameters
    ----------
    bodies : list[Body]
        Corps du système (ordre libre, mais conventionnellement :
        corps massifs fixes en premier, spacecraft en dernier).
    """

    def __init__(self, bodies: list[Body]):
        self.bodies = bodies
        self._time_history: list[float] = []

    # Accesseurs

    def __getitem__(self, name: str) -> Body:
        """Retourne un corps par son nom. Lève ValueError si absent."""
        for b in self.bodies:
            if b.name == name:
                return b
        raise ValueError(f"Corps '{name}' introuvable dans le système.")

    @property
    def massive_bodies(self) -> list[Body]:
        """Corps fixes (Soleil, Terre…) — sources de gravité."""
        return [b for b in self.bodies if b.fixed]

    @property
    def free_bodies(self) -> list[Body]:
        """Corps intégrés (spacecraft…)."""
        return [b for b in self.bodies if not b.fixed]

    # Historique temporel

    def record_time(self, t: float):
        self._time_history.append(t)

    def clear_history(self):
        self._time_history.clear()
        for b in self.bodies:
            b.clear_history()

    @property
    def time_history(self) -> np.ndarray:
        return np.array(self._time_history)

    # Diagnostics rapides (extrema pour débogage)

    def print_summary(self, label: str = ""):
        """Affiche quelques valeurs extremum après intégration."""
        tag = f"[{label}] " if label else ""
        print(f"\n{tag}Résumé de simulation")
        print(f"  Pas de temps enregistrés : {len(self._time_history)}")
        for b in self.free_bodies:
            if len(b._pos_history) == 0:
                continue
            pos = b.pos_history
            vel = b.vel_history
            speeds = np.linalg.norm(vel, axis=1)
            print(f"  {b.name}:")
            print(
                f"    |r| min/max : {np.linalg.norm(pos, axis=1).min():.4e} / "
                f"{np.linalg.norm(pos, axis=1).max():.4e}"
            )
            print(f"    |v| min/max : {speeds.min():.4e} / {speeds.max():.4e}")

    def __repr__(self):
        names = [b.name for b in self.bodies]
        return f"System({names})"
