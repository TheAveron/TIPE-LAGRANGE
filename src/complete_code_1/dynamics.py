"""Dynamics module: acceleration in the rotating synodic Sun-Earth frame.
The rotating frame is centered on the Sun-Earth barycenter; primaries are
fixed on the x-axis at x_sun and x_earth.


Equation implemented for the rotating-frame acceleration:
r_ddot = a_grav - 2 * omega x v - omega x (omega x r)
where a_grav = sum of gravitational accelerations from primaries.
"""

import numpy as np
from constants import G, M_sun, M_earth, x_sun, x_earth, omega_vec


r_sun = np.array([x_sun, 0.0, 0.0])
r_earth = np.array([x_earth, 0.0, 0.0])


def grav_acceleration(r):
    """Compute gravitational acceleration from Sun and Earth at position r (km, in rotating frame).
    Returns a vector (km/s^2).
    """
    rs = r - r_sun
    re = r - r_earth
    norm_rs = np.linalg.norm(rs)
    norm_re = np.linalg.norm(re)
    a_sun = -G * M_sun * rs / (norm_rs**3 + 1e-30)
    a_earth = -G * M_earth * re / (norm_re**3 + 1e-30)
    return a_sun + a_earth


def rotating_frame_acceleration(r, v):
    """Total acceleration in rotating frame (km/s^2) for state (r, v).
    Implements: r_ddot = a_grav - 2*omega x v - omega x (omega x r)
    """
    a_grav = grav_acceleration(r)
    coriolis = -2.0 * np.cross(omega_vec, v)
    centrifugal = -np.cross(omega_vec, np.cross(omega_vec, r))
    return a_grav + coriolis + centrifugal
