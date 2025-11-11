"""Compute L2 position on x-axis using simple scalar root finder.
We look for the colinear L2 point (on Sun-Earth line beyond Earth).
"""

import numpy as np
from constants import a, M_sun, M_earth, x_earth
from dynamics import rotating_frame_acceleration


def analytic_L2_distance_from_earth():
    """
    Approx distance Earth->L2 using the restricted 3-body cubic approximation:
    d = R * (mu/3)^(1/3) where mu = M_earth/(M_sun+M_earth), R = Sun-Earth distance (a)
    This gives the distance from Earth toward L2 (approx 1.5e6 km).
    """
    mu = M_earth / (M_sun + M_earth)
    return a * (mu / 3.0) ** (1.0 / 3.0)


def compute_L2(tol=1e-6, maxiter=60):
    """
    Return x coordinate (km) of L2 along +x direction beyond Earth, in barycentric rotating frame.
    We start with analytic guess and refine by 1D Newton on the x-axis using effective acceleration.
    """
    d_guess = analytic_L2_distance_from_earth()
    x0 = x_earth + d_guess  # starting guess on +x beyond Earth

    def f(x):
        r = np.array([x, 0.0, 0.0])
        v = np.zeros(3)
        # For equilibrium point in rotating frame with v=0, the net acceleration must be zero
        a = rotating_frame_acceleration(r, v)
        return a[0]  # x-component

    # numeric derivative
    x = x0
    for i in range(maxiter):
        fx = f(x)
        h = max(1e-3, abs(x) * 1e-6)
        fpx = (f(x + h) - f(x - h)) / (2 * h)
        if abs(fpx) < 1e-16:
            break
        dx = -fx / fpx
        x += dx
        if abs(dx) < tol:
            break
    return x
