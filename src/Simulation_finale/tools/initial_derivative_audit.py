import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
from scipy.integrate import solve_ivp

from src.simulation.Ephem_handler import EphemerisManager
from src.simulation.coordinates import CoordinateTransformer
from src.simulation.CRTBP_model_dynamics import CRTBP3Body
from src.simulation.MHF_ephem_dynamics import HighFidelityDynamics
from src.simulation.dynamics_conf import DynamicsConfig, DynamicsModel
from src.simulation.constants import Constants


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

    # initial RLP small orbit
    l2 = transformer.compute_l2_position_rlp()
    A_y = 100e6
    A_z = 50e6
    pos_rlp_0 = np.array([l2[0], A_y, A_z])
    vel_rlp_0 = np.array([0.0, 0.0, -Constants.OMEGA_EARTH * A_y])
    state_rlp = np.concatenate([pos_rlp_0, vel_rlp_0])

    # map to ecliptic
    et0 = ephem.et_from_j2000(t0)
    earth_pos0, earth_vel0 = ephem.get_body_state("EARTH", et0, "SSB", "J2000")
    state_ecl0 = transformer.rlp_to_ecliptic(
        state_rlp, t0, earth_position=earth_pos0, earth_velocity=earth_vel0
    )

    # compute derivatives
    state_dot_crtbp = crtbp.equations_of_motion(t0, state_rlp)
    state_dot_hf_ecl = hf.equations_of_motion(t0, state_ecl0)

    # transform HF derivative to RLP-frame derivative
    # 1) transform HF state to RLP to get v_rlp
    state_hf_rlp = transformer.ecliptic_to_rlp(
        state_ecl0, t0, earth_position=earth_pos0, earth_velocity=earth_vel0
    )
    v_rlp = state_hf_rlp[3:6]
    r_rlp = state_hf_rlp[:3]

    # 2) compute inertial HF acceleration at spacecraft and barycenter
    a_hf_sc = hf.compute_acceleration(t0, state_ecl0)

    # barycenter accel: compute acceleration of barycenter SSB (using ephem positions)
    sun_pos, sun_vel = ephem.get_body_state("SUN", et0, "SSB", "J2000")
    mu_ratio = Constants.MU_RATIO_SUN_EARTH
    bary_pos = sun_pos + mu_ratio * (earth_pos0 - sun_pos)
    # Avoid evaluating gravitational acceleration exactly at the Sun (singularity).
    # Use mass-weighted accelerations at the Sun and Earth positions to approximate
    # the barycenter acceleration (safe and consistent with earlier diagnostics).
    a_sun = hf._compute_gravitational_acceleration(t0, sun_pos)
    a_earth = hf._compute_gravitational_acceleration(t0, earth_pos0)
    a_bary = (Constants.M_SUN * a_sun + Constants.M_EARTH * a_earth) / (
        Constants.M_SUN + Constants.M_EARTH
    )

    # rotation
    sun_to_earth = earth_pos0 - sun_pos
    r_se = np.linalg.norm(sun_to_earth)
    x_axis = sun_to_earth / r_se
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    R = np.array([x_axis, y_axis, z_axis])

    a_rel = a_hf_sc - a_bary
    a_rel_rlp = R @ a_rel

    omega = np.array([0.0, 0.0, Constants.OMEGA_EARTH])
    coriolis = -2.0 * np.cross(omega, v_rlp)
    centrifugal = -np.cross(omega, np.cross(omega, r_rlp))
    a_hf_transformed = a_rel_rlp + coriolis + centrifugal

    # Debug prints for intermediate terms
    print("\nDEBUG intermediate acceleration terms:")
    print("a_hf_sc (inertial) =", a_hf_sc, "norm", np.linalg.norm(a_hf_sc))
    print("a_bary (inertial) =", a_bary, "norm", np.linalg.norm(a_bary))
    print("a_rel (inertial) =", a_rel, "norm", np.linalg.norm(a_rel))
    print("a_rel_rlp =", a_rel_rlp, "norm", np.linalg.norm(a_rel_rlp))
    print("coriolis =", coriolis, "norm", np.linalg.norm(coriolis))
    print("centrifugal =", centrifugal, "norm", np.linalg.norm(centrifugal))

    state_dot_hf_rlp = np.concatenate([v_rlp, a_hf_transformed])

    print("\n--- Initial derivative audit at t0 ---")
    print("state_rlp[0:6] =", state_rlp)
    print("state_ecl0[0:6] =", state_ecl0)
    print("\nCRTBP derivative (RLP):")
    print(state_dot_crtbp)
    print("\nHF derivative transformed to RLP:")
    print(state_dot_hf_rlp)
    print("\nDerivative diff (CRTBP - HF_transformed):")
    print(state_dot_crtbp - state_dot_hf_rlp)
    print(
        "norm posdot diff (m/s):",
        np.linalg.norm(state_dot_crtbp[:3] - state_dot_hf_rlp[:3]),
    )
    print(
        "norm veldot diff (m/s^2):",
        np.linalg.norm(state_dot_crtbp[3:] - state_dot_hf_rlp[3:]),
    )

    # Short step propagation (1 s) with both integrators to see immediate change
    dt = 1.0
    sol_crtbp = solve_ivp(
        crtbp.equations_of_motion, (0.0, dt), state_rlp, method="RK45", max_step=0.1
    )
    sol_hf = solve_ivp(
        hf.equations_of_motion, (0.0, dt), state_ecl0, method="RK45", max_step=0.1
    )

    s_crtbp_1 = sol_crtbp.y[:, -1]
    s_hf_ecl_1 = sol_hf.y[:, -1]
    et1 = ephem.et_from_j2000(dt)
    earth_pos1, earth_vel1 = ephem.get_body_state("EARTH", et1, "SSB", "J2000")
    s_hf_rlp_1 = transformer.ecliptic_to_rlp(
        s_hf_ecl_1, dt, earth_position=earth_pos1, earth_velocity=earth_vel1
    )

    print("\n--- After 1 s (RK45, max_step=0.1) ---")
    print("CRTBP state (RLP):", s_crtbp_1)
    print("HF state -> ECL:", s_hf_ecl_1)
    print("HF state -> RLP:", s_hf_rlp_1)
    print("Δpos after 1s (m):", s_crtbp_1[:3] - s_hf_rlp_1[:3])
    print("Δvel after 1s (m/s):", s_crtbp_1[3:] - s_hf_rlp_1[3:])

    # compute CRTBP omega from masses and compare
    GM = Constants.G * (Constants.M_SUN + Constants.M_EARTH)
    omega_kepler = np.sqrt(GM / (Constants.AU**3))
    print("\nCRTBP uses omega =", crtbp.omega)
    print("Constants.OMEGA_EARTH =", Constants.OMEGA_EARTH)
    print("omega Kepler (sqrt(G(M1+M2)/R^3)) =", omega_kepler)
    print(
        "relative difference (crtbp-omega_kepler)/omega_kepler =",
        (crtbp.omega - omega_kepler) / omega_kepler,
    )


if __name__ == "__main__":
    main()
