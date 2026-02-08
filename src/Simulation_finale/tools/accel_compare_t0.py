import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.constants import Constants
from src.simulation.coordinates import CoordinateTransformer
from src.simulation.CRTBP_model_dynamics import CRTBP3Body
from src.simulation.dynamics_conf import DynamicsConfig, DynamicsModel
from src.simulation.Ephem_handler import EphemerisManager
from src.simulation.MHF_ephem_dynamics import HighFidelityDynamics


def fmt(v):
    return f"[{v[0]:.6e}, {v[1]:.6e}, {v[2]:.6e}]"


def main():
    t0 = 0.0
    ephem = EphemerisManager()
    transformer = CoordinateTransformer(include_moon=True, ephem_manager=ephem)

    crtbp = CRTBP3Body(DynamicsConfig(model=DynamicsModel.CRTBP), normalized=False)
    hf = HighFidelityDynamics(
        DynamicsConfig(
            model=DynamicsModel.EPHEMERIS, include_moon=True, include_srp=False
        ),
        ephemeris_manager=ephem,
    )

    et0 = ephem.et_from_j2000(t0)
    earth_pos0, earth_vel0 = ephem.get_body_state("EARTH", et0, "SSB", "J2000")

    l2 = transformer.compute_l2_position_rlp()
    A_y = 100e6
    A_z = 50e6

    pos_rlp_0 = np.array([l2[0], A_y, A_z])
    vel_rlp_0 = np.array([0.0, 0.0, -Constants.OMEGA_EARTH * A_y])
    state_rlp = np.concatenate([pos_rlp_0, vel_rlp_0])

    state_ecl0 = transformer.rlp_to_ecliptic(
        state_rlp, t0, earth_position=earth_pos0, earth_velocity=earth_vel0
    )

    print("State RLP:")
    print(" pos:", fmt(state_rlp[:3]))
    print(" vel:", fmt(state_rlp[3:6]))

    # CRTBP accel at t0
    a_crtbp = crtbp.compute_acceleration(t0, state_rlp)
    print(
        "\nCRTBP accel (RLP) at t0:", fmt(a_crtbp), "| norm:", np.linalg.norm(a_crtbp)
    )

    # HF accel transformed
    a_hf_sc = hf._compute_gravitational_acceleration(et0, state_ecl0[:3])

    # barycenter accel mass-weighted
    a_sun = hf._compute_gravitational_acceleration(
        et0, ephem.get_body_state("SUN", et0, "SSB", "J2000")[0]
    )
    a_earth = hf._compute_gravitational_acceleration(et0, earth_pos0)
    a_hf_bary = (Constants.M_SUN * a_sun + Constants.M_EARTH * a_earth) / (
        Constants.M_SUN + Constants.M_EARTH
    )

    # rotate
    sun_pos0, _ = ephem.get_body_state("SUN", et0, "SSB", "J2000")
    sun_to_earth = earth_pos0 - sun_pos0
    x_axis = sun_to_earth / np.linalg.norm(sun_to_earth)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    R = np.array([x_axis, y_axis, z_axis])

    a_rel = a_hf_sc - a_hf_bary
    a_rel_rlp = R @ a_rel

    r_rlp = state_rlp[:3]
    v_rlp = state_rlp[3:6]
    omega = np.array([0.0, 0.0, Constants.OMEGA_EARTH])
    coriolis = -2.0 * np.cross(omega, v_rlp)
    centrifugal = -np.cross(omega, np.cross(omega, r_rlp))

    a_hf_trans = a_rel_rlp + coriolis + centrifugal

    print(
        "\nHF inertial accel at SC (ecl):",
        fmt(a_hf_sc),
        "| norm:",
        np.linalg.norm(a_hf_sc),
    )
    print(
        "HF barycenter accel (inertial):",
        fmt(a_hf_bary),
        "| norm:",
        np.linalg.norm(a_hf_bary),
    )
    print("a_rel (inertial):", fmt(a_rel), "| norm:", np.linalg.norm(a_rel))
    print("a_rel rotated->RLP:", fmt(a_rel_rlp), "| norm:", np.linalg.norm(a_rel_rlp))
    print("coriolis:", fmt(coriolis), "| centrifugal:", fmt(centrifugal))
    print(
        "a_hf_transformed (RLP):",
        fmt(a_hf_trans),
        "| norm:",
        np.linalg.norm(a_hf_trans),
    )

    diff = a_crtbp - a_hf_trans
    print(
        "\nDiff a_crtbp - a_hf_trans (RLP):", fmt(diff), "| norm:", np.linalg.norm(diff)
    )


if __name__ == "__main__":
    main()
