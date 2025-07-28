import numpy as np
from constants import M_earth, d_se, x_earth, x_sun, total_mass


def compute_theorical_lagrangian_points():
    """Compute Theorical positions of Lagragian points"""

    # Compute mass ratio
    mu = M_earth / total_mass
    D = d_se  # distance between Sun and Earth

    # Lagrange points (approximate)
    r_L1 = D * (1 - (mu / 3) ** (1 / 3))
    r_L2 = D * (1 + (mu / 3) ** (1 / 3))
    r_L3 = -D * (1 + 5 * mu / 12)

    # L4 and L5 (equilateral triangle, 60� above/below)
    x_L4 = x_earth - D * np.cos(np.pi / 3)
    y_L4 = D * np.sin(np.pi / 3)
    x_L5 = x_earth - D * np.cos(np.pi / 3)
    y_L5 = -D * np.sin(np.pi / 3)

    # Shift L1, L2, L3 from Sun's position
    x_L1 = x_sun + r_L1
    x_L2 = x_sun + r_L2
    x_L3 = x_sun + r_L3

    return x_L1, x_L2, x_L3, (x_L4, y_L4), (x_L5, y_L5)
