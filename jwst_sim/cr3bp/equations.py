"""
equations.py — Équations de mouvement du CR3BP non-dimensionnalisé.

Conventions (issues des articles NASA) :
  - Unité de longueur  : distance Soleil–Terre  (l* ≈ 1.496e11 m)
  - Unité de masse     : m_Soleil + m_Terre      (m*)
  - Unité de temps     : t* = sqrt(l*³ / (G m*)) → période = 2π
  - μ = m_Terre / m*  ≈ 3.0404e-6  (valeur JPL des articles)
  - Soleil  en  x = -μ        sur l'axe x
  - Terre   en  x = 1 - μ     sur l'axe x
  - Le repère tourne à ω = 1 (adim)

Pseudo-potentiel :
    U* = (1-μ)/d + μ/r + (x²+y²)/2

Équations de mouvement :
    ẍ - 2ẏ = ∂U*/∂x
    ÿ + 2ẋ = ∂U*/∂y
    z̈      = ∂U*/∂z
"""

import numpy as np

# Paramètre de masse Sun–Earth (valeur JPL, articles AAS 22-623 et 19-806)
MU_SUN_EARTH: float = 3.040423389123456e-6


def distances(x: float, y: float, z: float, mu: float) -> tuple[float, float]:
    """
    Distances adim. du spacecraft au Soleil (d) et à la Terre (r).

    d = ||spacecraft - Soleil||,  Soleil en (-μ, 0, 0)
    r = ||spacecraft - Terre||,   Terre  en (1-μ, 0, 0)
    """
    d = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    return d, r


def pseudo_potential(x: float, y: float, z: float, mu: float) -> float:
    """Pseudo-potentiel U* (scalaire)."""
    d, r = distances(x, y, z, mu)
    return (1 - mu) / d + mu / r + 0.5 * (x**2 + y**2)


def jacobi_constant(state: np.ndarray, mu: float) -> float:
    """
    Constante de Jacobi C = 2U* - v².

    C est une intégrale première du CR3BP : sa conservation est
    l'indicateur principal de la qualité numérique de l'intégration.

    Parameters
    ----------
    state : array (6,)  [x, y, z, vx, vy, vz]  adim.
    mu    : float
    """
    x, y, z, vx, vy, vz = state
    v2 = vx**2 + vy**2 + vz**2
    return 2 * pseudo_potential(x, y, z, mu) - v2


def eom(t: float, state: np.ndarray, mu: float) -> np.ndarray:
    """
    Dérivée du vecteur d'état dans le CR3BP.

    Parameters
    ----------
    t     : float         temps adim. (non utilisé, système autonome)
    state : np.ndarray    [x, y, z, vx, vy, vz]
    mu    : float

    Returns
    -------
    dstate : np.ndarray   [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state
    d, r = distances(x, y, z, mu)

    d3 = d**3
    r3 = r**3

    # Dérivées partielles de U*
    dUx = x - (1 - mu) * (x + mu) / d3 - mu * (x - 1 + mu) / r3
    dUy = y * (1 - (1 - mu) / d3 - mu / r3)
    dUz = z * (-(1 - mu) / d3 - mu / r3)  # terme centrifuge nul en z

    # Termes de Coriolis : -2ẏ sur ax, +2ẋ sur ay
    ax = dUx + 2 * vy
    ay = dUy - 2 * vx
    az = dUz

    return np.array([vx, vy, vz, ax, ay, az])


def eom_factory(mu: float):
    """Retourne une fonction eom(t, state) avec mu fixé (pour l'intégrateur)."""

    def _eom(t: float, state: np.ndarray) -> np.ndarray:
        return eom(t, state, mu)

    return _eom
