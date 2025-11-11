# frames.py
import numpy as np
from constants import omega_vec, x_sun, x_earth

# frames.py
import numpy as np
from constants import omega_vec, x_sun, x_earth, a, M_sun, M_earth


def rotation_matrix_theta(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0, 0, 1.0]])
    return R


def rotation_matrix_dot_theta(theta, omega):
    d = omega
    Rdot = np.array(
        [
            [-np.sin(theta) * d, np.cos(theta) * d, 0.0],
            [-np.cos(theta) * d, -np.sin(theta) * d, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    return Rdot


def inertial_to_rotating(r_I, v_I, t, input_frame="heliocentric"):
    """
    Convert inertial vectors (r_I, v_I) in km / km/s at epoch t (s) into the rotating synodic frame.
    input_frame: 'heliocentric' or 'barycentric'
      - 'heliocentric': r_I is Sun-centered; we transfer origin to barycenter by subtracting Sun position in inertial frame.
      - 'barycentric': r_I is already expressed in Sun-Earth barycenter frame.
    NOTE: This function assumes orbits are near-circular and uses mean motion omega from constants.
    """
    omega = omega_vec[2]
    theta = omega * t
    R = rotation_matrix_theta(theta)
    Rdot = rotation_matrix_dot_theta(theta, omega)

    if input_frame == "heliocentric":
        # Convert Sun-centered -> barycentric by subtracting Sun position vector in inertial frame.
        # In inertial frame, Sun position relative to barycenter is x_sun on x-axis when theta=0,
        # but to be rigorous you rotate x_sun by the inertial angle theta to get Sun inertial pos:
        # r_sun_inertial = rotation(-theta) @ [x_sun, 0, 0]  (since rotating frame = rotated by +theta)
        # But easier: the barycenter->Sun vector in inertial frame is:
        r_sun_inertial = np.array(
            [x_sun * np.cos(0.0), x_sun * np.sin(0.0), 0.0]
        )  # x_sun on x at t=0
        # Apply rotation of inertial frame relative to rotating frame:
        # rotate r_sun_inertial by -theta to express it in inertial orientation at time t:
        R_inertial_to_rot = (
            R  # mapping inertial -> rotating we use R (common convention above)
        )
        # So barycentric r = r_heliocentric - r_sun_inertial (but r_sun_inertial expressed in same inertial orientation)
        # To be safe assume r_I is heliocentric and Sun is at origin in r_I, so barycentric r_b = r_I - r_sun_inertial(t)
        # Here r_sun_inertial(t) = rotation_matrix(-theta) @ [x_sun0,0,0] but x_sun0 is constant small
        # Simplification: since x_sun magnitude is small (~450 km) compared to AU, error small.
        # We'll subtract rotated x_sun:
        r_sun_inertial_t = np.array(
            [x_sun * np.cos(-theta), x_sun * np.sin(-theta), 0.0]
        )
        r_bary = r_I - r_sun_inertial_t
        v_bary = v_I  # neglecting small correction from Sun motion (acceptable for 1st order)
    else:
        # barycentric input: use directly
        r_bary = r_I
        v_bary = v_I

    # Now rotate to rotating frame
    r_R = R @ r_bary
    v_R = R @ v_bary + Rdot @ r_bary
    return r_R, v_R
