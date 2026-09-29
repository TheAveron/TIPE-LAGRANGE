"""
forces.py — Forces newtoniennes dans le référentiel inertiel J2000.

Unités SI :
  masses     [kg]
  positions  [m]
  vitesses   [m/s]
  forces     [N]
  temps      [s]
"""

import numpy as np
from core.body import Body
from numpy.typing import NDArray

G = np.float64(6.674_30e-11)  # constante gravitationnelle [m³ kg⁻¹ s⁻²]


def gravitational_acceleration(
    pos_sc: NDArray[np.float64],
    bodies: list[Body],
) -> NDArray[np.float64]:
    """
    Accélération gravitationnelle nette sur le spacecraft.

    a = Σ_i  G m_i (r_i - r_sc) / |r_i - r_sc|³

    Parameters
    ----------
    pos_sc : NDArray, shape (3,)   position du spacecraft [m]
    bodies : list of Body             corps massifs (Soleil, Terre…)

    Returns
    -------
    a : NDArray, shape (3,)   [m/s²]
    """
    a = np.zeros(3, dtype=np.float64)
    for body in bodies:
        dr = body.position - pos_sc
        dist = np.linalg.norm(dr)
        if dist < 1.0:
            # Garde-fou numérique : évite la singularité (ne devrait jamais
            # arriver en pratique pour le JWST loin de tous les corps)
            continue
        a += G * body.mass * dr / dist**3
    return a


def mechanical_energy(
    pos_sc: NDArray[np.float64],
    vel_sc: NDArray[np.float64],
    mass_sc: np.float64,
    bodies: list[Body],
) -> np.float64:
    """
    Énergie mécanique spécifique du spacecraft [J/kg].

    E = v²/2  -  Σ_i G m_i / |r_i - r_sc|

    C'est l'énergie par unité de masse ; sa conservation est l'indicateur
    de qualité de l'intégration dans le référentiel inertiel.
    """
    kin = 0.5 * np.dot(vel_sc, vel_sc)
    pot = 0.0
    for body in bodies:
        dr = body.position - pos_sc
        dist = np.linalg.norm(dr)
        pot -= G * body.mass / dist
    return kin + pot
