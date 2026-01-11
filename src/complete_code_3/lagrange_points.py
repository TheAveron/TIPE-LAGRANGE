import numpy as np
from constants import Constants
from vector import Vector3D


def calculate_lagrange_points() -> dict:
    """
    Calcule les positions des 5 points de Lagrange dans le référentiel tournant
    Pour le système Soleil-Terre avec orbite circulaire

    Returns:
        Dictionnaire avec les positions des points L1 à L5
    """
    # Paramètre de masse μ = M_T / (M_S + M_T)
    mu = Constants.M_EARTH / (Constants.M_SUN + Constants.M_EARTH)
    r = Constants.R_EARTH_ORBIT

    # L1, L2, L3 : approximations au premier ordre
    # Position exacte nécessite résolution numérique d'équation quintique
    # Approximation : L2 est à r * (1 + (mu/3)^(1/3)) du Soleil

    # L1 : entre Soleil et Terre
    r_L1 = r * (1 - (mu / 3) ** (1 / 3))

    # L2 : au-delà de la Terre (où se trouve JWST)
    r_L2 = r * (1 + (mu / 3) ** (1 / 3))

    # L3 : opposé à la Terre
    r_L3 = -r * (1 + 5 * mu / 12)

    # À t=0, la Terre est sur l'axe +x
    lagrange_points = {
        "L1": Vector3D(r_L1, 0, 0),
        "L2": Vector3D(r_L2, 0, 0),
        "L3": Vector3D(r_L3, 0, 0),
        "L4": Vector3D(r * np.cos(np.pi / 3), r * np.sin(np.pi / 3), 0),
        "L5": Vector3D(r * np.cos(-np.pi / 3), r * np.sin(-np.pi / 3), 0),
    }

    return lagrange_points
