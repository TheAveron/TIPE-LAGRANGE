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
from scipy.optimize import brentq
from .equations import MU_SUN_EARTH

# ---------------------------------------------------------------------------
# 1. Points de Lagrange colinéaires (L1, L2, L3)
# ---------------------------------------------------------------------------


def _gamma_L2(mu: float) -> float:
    """
    Distance adim. entre la Terre et L2, notée γ₂.
    Solution positive de l'équation quintic de Hill.
    """

    # Équation : γ⁵ - (3-μ)γ⁴ + (3-2μ)γ³ - μγ² + 2μγ - μ = 0  (approx)
    # On utilise la forme classique de la série de Lagrange tronquée :
    #   γ ≈ (μ/3)^(1/3) comme point de départ, puis résolution numérique.
    def eq(g):
        return (
            g**5 - (3 - mu) * g**4 + (3 - 2 * mu) * g**3 - mu * g**2 + 2 * mu * g - mu
        )

    g0 = (mu / 3) ** (1 / 3)
    return float(brentq(eq, g0 * 0.5, g0 * 1.5))  # type: ignore


def lagrange_L2(mu: float = MU_SUN_EARTH) -> np.ndarray:
    """
    Position de L2 dans le repère tournant, sous forme [x_L2, 0, 0].
    x_L2 = 1 - μ + γ₂  (légèrement au-delà de la Terre)
    """
    gamma = _gamma_L2(mu)
    x_L2 = 1.0 - mu + gamma
    return np.array([x_L2, 0.0, 0.0])


# ---------------------------------------------------------------------------
# 2. Approximation de Richardson (3ème ordre) pour orbite halo autour de L2
# ---------------------------------------------------------------------------


def richardson_halo_L2(
    Az: float,
    mu: float = MU_SUN_EARTH,
    northern: bool = True,
    phi: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    État initial d'une orbite halo autour de L2, approximation de Richardson.

    Parameters
    ----------
    Az : float
        Amplitude hors-plan (adim.).  Ex : 0.00279 pour JWST.
    mu : float
        Paramètre de masse du CR3BP.
    northern : bool
        True  → halo nord  (z > 0 initialement, m = +1)
        False → halo sud   (z < 0 initialement, m = -1)
    phi : float
        Phase initiale sur l'orbite [rad]. phi=0 → départ au plan z=max.

    Returns
    -------
    state : np.ndarray, shape (6,)
        [x, y, z, vx, vy, vz] dans le repère tournant adim.
    T_half : float
        Demi-période approximative de l'orbite halo [adim.].
    """
    m = 1 if northern else -1
    gamma = _gamma_L2(mu)
    c = _cn_coefficients(gamma, mu)

    c2, c3, c4 = c[2], c[3], c[4]

    # ------------------------------------------------------------------
    # Fréquences et coefficients (Richardson 1980, Table 1)
    # ------------------------------------------------------------------
    # Fréquence dans le plan  λ  (valeur propre de la partie in-plane)
    lam = _lambda_in_plane(c2)

    # Coefficients k, d1, d2
    k = 2 * lam / (lam**2 + 1 - c2)

    d1 = (3 * lam**2 / k) * (k * (6 * lam**2 - 1) - 2 * lam)
    d2 = (8 * lam**2 / k) * (k * (11 * lam**2 - 1) - 2 * lam)

    # Amplitude Ax en fonction de Az (relation de bifurcation halo)
    # a21..a24, b21..b22 de Richardson
    a21 = (3 * c3 * (k**2 - 2)) / (4 * (1 + 2 * c2))
    a22 = (3 * c3) / (4 * (1 + 2 * c2))
    a23 = (-3 * c3 * lam / (4 * k * d1)) * (3 * k**3 * lam - 6 * k * (k - lam) + 4)
    a24 = (-3 * c3 * lam / (4 * k * d1)) * (2 + 3 * k * lam)
    b21 = (-3 * c3 * lam / (2 * d1)) * (3 * k * lam - 4)
    b22 = 3 * c3 * lam / d1

    d21 = -c3 / (2 * lam**2)

    a31 = (-9 * lam / (4 * d2)) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2))
    a32 = (-1 / (4 * d2)) * (
        9 * lam * (4 * c3 * (k * a24 - b22) + k * c4)
        + 3 * c3**2 * (2 - k**2)
        + 4 * c4 * (k**2 + 2)
    )
    b31 = (3 / (8 * d2)) * (
        8 * lam * (3 * c3 * (k * b21 - lam * a23) - c4 * (2 + 3 * k**2))
        + (9 * lam**2 + 1 + 2 * c2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2))
    )
    b32 = (1 / d2) * (
        9 * lam * (c3 * (k * b22 + lam * a24) - c4)
        + (3 / 8) * (9 * lam**2 + 1 + 2 * c2) * (4 * c3 * (k * a24 - b22) + k * c4)
    )

    # Amplitude in-plane Ax² = -delta2 / delta1  avec delta = f(Az²)
    delta2 = 2 * lam * (lam * (1 + k**2) - 2 * k)
    a1 = -1.5 * c3 * (2 * a21 + a23 + 5 * d21) - 0.375 * c4 * (12 - k**2)
    a2 = 1.5 * c3 * (a24 - 2 * a22) + 1.125 * c4

    # La relation halo : Ax² = -(a1 Az² + delta2) / a2
    # (le signe est correct pour Az petit)
    Ax2 = (a1 * Az**2 + delta2) / a2
    if Ax2 < 0:
        raise ValueError(
            f"Az={Az:.4e} trop grand : Ax² < 0. Réduire Az (max ≈ 0.005 pour JWST)."
        )
    Ax = np.sqrt(Ax2)

    # Fréquence corrigée au 3ème ordre
    omega1 = 0.0  # correction 1er ordre nulle pour halo
    omega2 = (
        (
            (-3 / 2) * c3 * (2 * a21 + a23 + 5 * d21)
            - (3 / 8) * c4 * (12 - k**2)
            + a1 * Ax**2
            + (a2 * Az**2) / Ax**2 * 0  # terme croisé nul ici
        )
        if Ax > 0
        else 0.0
    )

    # Fréquence totale ν = λ + ε²ω₂  (ε ~ Az, approximation)
    nu = lam + omega2 * Az**2

    # Demi-période
    T_half = np.pi / nu

    # ------------------------------------------------------------------
    # Coordonnées dans le repère centré sur L2 (repère de Richardson)
    # puis recentrage sur le barycentre
    # ------------------------------------------------------------------
    tau = phi  # phase initiale

    x_L2 = lagrange_L2(mu)[0]

    # Développement 3ème ordre de Richardson (coordonnées locales ξ, η, ζ)
    xi = (
        a21 * Ax**2
        + a22 * Az**2
        - Ax * np.cos(tau)
        + (a23 * Ax**2 - a24 * Az**2) * np.cos(2 * tau)
        + (a31 * Ax**3 - a32 * Ax * Az**2) * np.cos(3 * tau)
    )

    eta = (
        k * Ax * np.sin(tau)
        + (b21 * Ax**2 - b22 * Az**2) * np.sin(2 * tau)
        + (b31 * Ax**3 - b32 * Ax * Az**2) * np.sin(3 * tau)
    )

    zeta = m * (
        Az * np.cos(tau)
        + d21 * Ax * Az * (np.cos(2 * tau) - 3)
        + (a32 * Ax**2 * Az - a31 * Az**3) * np.cos(3 * tau)
    )

    # Vitesses (dérivées par rapport à τ = ν·t, donc dτ/dt = ν)
    xi_dot = nu * (
        Ax * np.sin(tau)
        - 2 * (a23 * Ax**2 - a24 * Az**2) * np.sin(2 * tau)
        - 3 * (a31 * Ax**3 - a32 * Ax * Az**2) * np.sin(3 * tau)
    )

    eta_dot = nu * (
        k * Ax * np.cos(tau)
        + 2 * (b21 * Ax**2 - b22 * Az**2) * np.cos(2 * tau)
        + 3 * (b31 * Ax**3 - b32 * Ax * Az**2) * np.cos(3 * tau)
    )

    zeta_dot = (
        nu
        * m
        * (
            -Az * np.sin(tau)
            - 2 * d21 * Ax * Az * np.sin(2 * tau)
            - 3 * (a32 * Ax**2 * Az - a31 * Az**3) * np.sin(3 * tau)
        )
    )

    # Passage aux coordonnées CR3BP (barycentre comme origine)
    x = x_L2 + xi
    y = eta
    z = zeta
    vx = xi_dot
    vy = eta_dot
    vz = zeta_dot

    return np.array([x, y, z, vx, vy, vz]), T_half


# ---------------------------------------------------------------------------
# Fonctions auxiliaires internes
# ---------------------------------------------------------------------------


def _cn_coefficients(gamma: float, mu: float, n_max: int = 5) -> dict[int, float]:
    """
    Coefficients c_n du développement du potentiel autour de L2.
    c_n = (1/γ³) [ μ + (-1)^n (1-μ) γ^(n+1) / (1-γ)^(n+1) ]
    """
    c = {}
    for n in range(2, n_max + 1):
        c[n] = (1 / gamma**3) * (
            mu + (-1) ** n * (1 - mu) * gamma ** (n + 1) / (1 - gamma) ** (n + 1)
        )
    return c


def _lambda_in_plane(c2: float) -> float:
    """
    Valeur propre réelle positive de la partie in-plane (fréquence λ).
    λ² = (c2 - 2 + sqrt(9c2² - 8c2)) / 2
    """
    disc = 9 * c2**2 - 8 * c2
    lam2 = (c2 - 2 + np.sqrt(disc)) / 2
    return np.sqrt(lam2)
