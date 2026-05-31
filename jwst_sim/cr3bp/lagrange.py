"""
lagrange.py — Points de Lagrange et état initial via approximation de Richardson.

L2 est le point de Lagrange situé au-delà de la Terre (côté anti-Soleil),
à environ 1.5 million de km de la Terre, soit ~0.01 unités adim.

Approximation de Richardson au 3ème ordre :
  Donne un état initial (x, y, z, vx, vy, vz) sur une orbite halo
  périodique aproximée du CR3BP. Ce n'est pas une orbite exacte, mais
  elle est suffisamment proche pour qu'un intégrateur numérique reste
  dans la région de L2 (la dérive est lente, de l'ordre des termes d'ordre 4).

  Référence : Richardson, D.L. (1980), "Analytic Construction of Periodic
  Orbits about the Collinear Points", Celestial Mechanics, 22, 241-253.

Amplitudes cibles JWST (articles AAS) :
  Ay ≈ 771 000 km  →  Ay_adim ≈ 771 000 / 149 597 870 ≈ 0.00515
  Az ≈ 418 000 km  →  Az_adim ≈ 418 000 / 149 597 870 ≈ 0.00279
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .equations import MU_SUN_EARTH

ZERO = np.float64(0)


# 1. Points de Lagrange colinéaires (L1, L2, L3)
def _gamma_L2(mu: np.float64) -> np.float64:
    """
    Dimensionless distance between the secondary body and L2
    in the circular restricted three-body problem.
    """
    mu = np.float64(mu)

    def eq(g):
        return (
            g**5 + (3 - mu) * g**4 + (3 - 2 * mu) * g**3 - mu * g**2 - 2 * mu * g - mu
        )

    g0 = (mu / 3) ** (1 / 3)

    return np.float64(brentq(eq, g0 * 0.5, g0 * 1.5))


def lagrange_L2(mu: np.float64 = MU_SUN_EARTH) -> NDArray[np.float64]:
    """
    Position de L2 dans le repère tournant, sous forme [x_L2, 0, 0].
    x_L2 = 1 - μ + γ₂  (légèrement au-delà de la Terre)
    """
    gamma = _gamma_L2(mu)
    x_L2 = 1.0 - mu + gamma
    return np.array([x_L2, 0.0, 0.0], dtype=np.float64)


# 2. Approximation de Richardson (3ème ordre) pour orbite halo autour de L2
def richardson_halo_L2(
    Az: np.float64,
    mu: np.float64 = MU_SUN_EARTH,
    northern: bool = True,
    phi: np.float64 = ZERO,
) -> tuple[NDArray[np.float64], np.float64, np.float64]:
    """
    État initial d'une orbite halo autour de L2, approximation de Richardson.

    Parameters
    ----------
    Az : np.float64
        Amplitude hors-plan (adim.).  Ex : 0.00279 pour JWST.
    mu : np.float64
        Paramètre de masse du CR3BP.
    northern : bool
        True  → halo nord  (z > 0 initialement, m = +1)
        False → halo sud   (z < 0 initialement, m = -1)
    phi : np.float64
        Phase initiale sur l'orbite [rad]. phi=0 → départ au plan z=max.

    Returns
    -------
    state : NDArray[np.float64], shape (6,)
        [x, y, z, vx, vy, vz] dans le repère tournant adim.
    T_half : np.float64
        Demi-période approximative de l'orbite halo [adim.].
    """
    m = 1 if northern else -1
    gamma = _gamma_L2(mu)
    c = _cn_coefficients(gamma, mu)

    c2, c3, c4 = c[2], c[3], c[4]

    print("cn coeff", c2, c3, c4)

    Az = Az / gamma

    # Fréquences et coefficients (Richardson 1980, Table 1)

    # Fréquences (valeur propre de la partie in-plane)
    lam, omega_p, omega_v = _eigenvalues(c2)

    # Coefficients k, d1, d2
    k = (1 + 2 * c2 + lam**2) / (2 * lam)

    d1 = 3 * (lam**2) * (k * (6 * (lam**2) - 1) - 2 * lam) / k
    d2 = 8 * (lam**2) * (k * (11 * (lam**2) - 1) - 2 * lam) / k

    # Amplitude Ax en fonction de Az (relation de bifurcation halo)
    # a21..a24, b21..b22 de Richardson
    a21 = (3 * c3 * (k**2 - 2)) / (4 * (1 + 2 * c2))
    a22 = (3 * c3) / (4 * (1 + 2 * c2))
    a23 = -3 * c3 * lam * (3 * lam * (k**3) - 6 * k * (k - lam) + 4) / (4 * k * d1)
    a24 = -3 * c3 * lam * (2 + 3 * k * lam) / (4 * k * d1)

    b21 = -3 * c3 * lam * (3 * k * lam - 4) / (2 * d1)
    b22 = 3 * c3 * lam / d1

    d21 = -c3 / (2 * lam**2)

    a31 = -9 * lam * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2)) / (4 * d2) + (
        9 * lam**2 + 1 - c2
    ) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k**2)) / (2 * d2)
    a32 = -(
        (
            9 * lam * (4 * c3 * (k * a24 - b22) + k * c4) / 4
            + 3 * (9 * lam**2 + 1 - c2) * (c3 * (k * b22 + d21 - 2 * a24) - c4) / 2
        )
        / d2
    )

    b31 = (
        3
        * (
            8 * lam * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k**2))
            + (1 + 2 * c2 + 9 * lam**2)
            * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2))
        )
        / (8 * d2)
    )
    b32 = (
        9 * lam * (3 * c3 * (k * b22 + d21 - 2 * a24) - c4)
        + (3 / 8) * (9 * lam**2 + 1 + 2 * c2) * (4 * c3 * (k * a24 - b22) + k * c4)
    ) / d2

    d31 = (3 / (64 * lam**2)) * (4 * c3 * a24 + c4)
    d32 = (3 / (64 * lam**2)) * (4 * c3 * a23 - d21 + c4 * (1 + k**2 + 3))  # k**2 + 4

    # ----

    temp_denom = 2 * lam * (lam * (1 + k**2) - 2 * k)
    s1 = (
        3 * c3 * (2 * a21 * (k**2 - 2) - a23 * (k**2 + 2) - 2 * k * b21) / 2
        - 3 * (3 * k**4 - 8 * k**2 + 8) / 8
    ) / temp_denom
    s2 = (
        3 * c3 * (2 * a22 * (k**2 - 2) - a24 * (k**2 + 2) + 2 * k * b22 + 5 * d21) / 2
        + 3 * c4 * (12 - k**2) / 8
    ) / temp_denom

    a1 = -3 * c3 * (2 * a21 + a23 + 5 * d21) / 2 - 3 * c4 * (12 - k**2) / 8

    a2 = 3 * (a24 - 2 * a22) / 2 + 9 * c4 / 8

    l1 = a1 + 2 * lam**2 * s1
    l2 = a2 + 2 * lam**2 * s2

    # Empirical initialization for Sun–Earth L2 halo family

    # Fréquence corrigée au 3ème ordre
    omega1 = ZERO  # correction 1er ordre nulle pour halo

    delta = omega_p**2 - omega_v**2
    Ax = np.sqrt(-(delta + Az**2 * l2) / l1, dtype=np.float64)
    nu = 1 + s1 * Ax**2 + s2 * Az**2

    # Demi-période
    T_half = np.float64(np.pi / (omega_p * nu))

    # Coordonnées dans le repère centré sur L2 (repère de Richardson)
    # puis recentrage sur le barycentre
    tau = ZERO

    tau1 = omega_p * tau + phi

    x_L2 = np.float64(lagrange_L2(mu)[0])

    # Développement 3ème ordre de Richardson (coordonnées locales ξ, η, ζ)
    xi = (
        a21 * Ax**2
        + a22 * Az**2
        - Ax * np.cos(tau1, dtype=np.float64)
        + (a23 * Ax**2 - a24 * Az**2) * np.cos(2 * tau1, dtype=np.float64)
        + (a31 * Ax**3 - a32 * Ax * Az**2) * np.cos(3 * tau1, dtype=np.float64)
    )

    eta = (
        k * Ax * np.sin(tau1, dtype=np.float64)
        + (b21 * Ax**2 - b22 * Az**2) * np.sin(2 * tau1, dtype=np.float64)
        + (b31 * Ax**3 - b32 * Ax * Az**2) * np.sin(3 * tau1, dtype=np.float64)
    )

    zeta = m * (
        Az * np.cos(tau1, dtype=np.float64)
        + d21 * Ax * Az * (np.cos(2 * tau1, dtype=np.float64) - 3)
        + (d32 * Ax**2 * Az - d31 * Az**3) * np.cos(3 * tau1, dtype=np.float64)
    )

    # Vitesses (dérivées par rapport à τ = ν·t, donc dτ/dt = ν)
    xi_dot = (
        omega_p
        * nu
        * (
            +Ax * np.sin(tau1, dtype=np.float64)
            - 2 * (a23 * Ax**2 - a24 * Az**2) * np.sin(2 * tau1, dtype=np.float64)
            - 3 * (a31 * Ax**3 - a32 * Ax * Az**2) * np.sin(3 * tau1, dtype=np.float64)
        )
    )

    eta_dot = (
        omega_p
        * nu
        * (
            k * Ax * np.cos(tau1, dtype=np.float64)
            + 2 * (b21 * Ax**2 - b22 * Az**2) * np.cos(2 * tau1, dtype=np.float64)
            + 3 * (b31 * Ax**3 - b32 * Ax * Az**2) * np.cos(3 * tau1, dtype=np.float64)
        )
    )

    zeta_dot = (
        nu
        * omega_p
        * m
        * (
            -Az * np.sin(tau1, dtype=np.float64)
            - 2 * d21 * Ax * Az * np.sin(2 * tau1, dtype=np.float64)
            - 3 * (d32 * Ax**2 * Az - d31 * Az**3) * np.sin(3 * tau1, dtype=np.float64)
        )
    )

    # Passage aux coordonnées CR3BP (barycentre comme origine)
    x = x_L2 + xi * gamma
    y = eta * gamma
    z = zeta * gamma
    vx = xi_dot * gamma
    vy = eta_dot * gamma
    vz = zeta_dot * gamma

    return np.array([x, y, z, vx, vy, vz], dtype=np.float64), T_half, c2


# Fonctions auxiliaires internes
def _cn_coefficients(
    gamma: np.float64, mu: np.float64, n_max: int = 5
) -> dict[int, np.float64]:
    """
    Coefficients c_n du développement du potentiel autour de L2.
    c_n = ((-1)^n / γ³) [ μ + (1-μ) γ^(n+1) / (1+γ)^(n+1) ]
    """
    c: dict[int, np.float64] = {}
    for n in range(2, n_max + 1):
        c[n] = np.float64(
            (-1) ** n
            * (mu + (1 - mu) * (gamma ** (n + 1)) / ((1 + gamma) ** (n + 1)))
            / (gamma ** (3))
        )
    return c


def _eigenvalues(c2: np.float64) -> tuple[np.float64, np.float64, np.float64]:
    """
    Valeur propre réelle positive de la partie in-plane (fréquence λ).
    λ² = (c2 - 2 + sqrt(9c2² - 8c2)) / 2
    """
    disc = 9 * c2**2 - 8 * c2
    lam2 = (c2 - 2 + np.sqrt(disc, dtype=np.float64)) / 2
    omega_p2 = (c2 - 2 - np.sqrt(disc, dtype=np.float64)) / 2
    return (
        np.sqrt(lam2, dtype=np.float64),
        np.sqrt(-omega_p2, dtype=np.float64),
        np.sqrt(c2, dtype=np.float64),
    )
