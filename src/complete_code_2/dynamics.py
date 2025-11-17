"""Dynamics module: acceleration in the rotating synodic Sun-Earth frame.
The rotating frame is centered on the Sun-Earth barycenter; primaries are
fixed on the x-axis at x_sun and x_earth.


Equation implemented for the rotating-frame acceleration:
r_ddot = a_grav - 2 * omega x v - omega x (omega x r)
where a_grav = sum of gravitational accelerations from primaries.
"""

import numpy as np
from constants import EPS, G, M_sun, M_earth, x_sun, x_earth, omega_vec
from numba import njit


r_sun = np.array([x_sun, 0.0, 0.0])
r_earth = np.array([x_earth, 0.0, 0.0])


@njit
def grav_acceleration(r):
    """Compute gravitational acceleration from Sun and Earth at position r (km, in rotating frame).
    Returns a vector (km/s^2).
    """
    rs = r - r_sun
    re = r - r_earth

    norm_rs = np.sqrt(rs[0] ** 2 + rs[1] ** 2 + rs[2] ** 2)
    norm_re = np.sqrt(re[0] ** 2 + re[1] ** 2 + re[2] ** 2)

    if norm_re < EPS or np.isnan(norm_re):
        norm_re = 1
    if norm_rs < EPS or np.isnan(norm_rs):
        norm_rs = 1

    a_sun = -G * M_sun * rs / (norm_rs**3)
    a_earth = -G * M_earth * re / (norm_re**3)

    return a_sun + a_earth


@njit
def rotating_frame_acceleration(r, v):
    """Total acceleration in rotating frame (km/s^2) for state (r, v).
    Implements: r_ddot = a_grav - 2*omega x v - omega x (omega x r)
    """
    a_grav = grav_acceleration(r)
    coriolis = -2.0 * np.cross(omega_vec, v)
    centrifugal = -np.cross(omega_vec, np.cross(omega_vec, r))
    return a_grav + coriolis + centrifugal


def inertial_to_rotating(x_list, y_list, z_list, dt, omega):
    """
    Transforme une trajectoire inertielle vers le repère tournant Sun–Earth.
    Retourne x_rot, y_rot, z_rot.
    """
    N = len(x_list)
    x_rot = np.empty(N)
    y_rot = np.empty(N)
    z_rot = np.empty(N)

    for i in range(N):
        t = i * dt
        theta = omega * t  # angle de rotation du repère
        c = np.cos(-theta)
        s = np.sin(-theta)

        x_rot[i] = c * x_list[i] - s * y_list[i]
        y_rot[i] = s * x_list[i] + c * y_list[i]
        z_rot[i] = z_list[i]

    return x_rot, y_rot, z_rot
