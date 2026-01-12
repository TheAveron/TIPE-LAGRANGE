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
    t_check = 3600.0  # 1 hour

    ephem = EphemerisManager()
    transformer = CoordinateTransformer(include_moon=True, ephem_manager=ephem)

    # CRTBP
    crtbp = CRTBP3Body(DynamicsConfig(model=DynamicsModel.CRTBP), normalized=False)

    # High fidelity (no SRP)
    hf = HighFidelityDynamics(
        DynamicsConfig(
            model=DynamicsModel.EPHEMERIS, include_moon=True, include_srp=False
        ),
        ephemeris_manager=ephem,
    )

    # Initial RLP state (small orbit around L2)
    l2 = transformer.compute_l2_position_rlp()
    A_y = 100e6
    A_z = 50e6
    pos_rlp_0 = np.array([l2[0], A_y, A_z])
    vel_rlp_0 = np.array([0.0, 0.0, -Constants.OMEGA_EARTH * A_y])
    state_rlp = np.concatenate([pos_rlp_0, vel_rlp_0])

    # Convert to ecliptic for HF
    et0 = ephem.et_from_j2000(t0)
    earth_pos0, earth_vel0 = ephem.get_body_state("EARTH", et0, "SSB", "J2000")
    state_ecl0 = transformer.rlp_to_ecliptic(
        state_rlp, t0, earth_position=earth_pos0, earth_velocity=earth_vel0
    )

    # Propagate short 1 hour with both
    t_span = (t0, t_check)
    sol_crtbp = solve_ivp(
        crtbp.equations_of_motion,
        t_span,
        state_rlp,
        rtol=1e-12,
        atol=1e-12,
        dense_output=True,
    )
    sol_hf = solve_ivp(
        hf.equations_of_motion,
        t_span,
        state_ecl0,
        rtol=1e-12,
        atol=1e-12,
        dense_output=True,
    )

    state_crtbp_t = sol_crtbp.sol(t_check)
    state_hf_ecl_t = sol_hf.sol(t_check)

    # Earth ephemeris at t_check
    et_check = ephem.et_from_j2000(t_check)
    earth_pos_t, earth_vel_t = ephem.get_body_state("EARTH", et_check, "SSB", "J2000")
    sun_pos_t, sun_vel_t = ephem.get_body_state("SUN", et_check, "SSB", "J2000")
    mu_ratio = Constants.MU_RATIO_SUN_EARTH
    barycenter_pos = sun_pos_t + mu_ratio * (earth_pos_t - sun_pos_t)
    barycenter_vel = sun_vel_t + mu_ratio * (earth_vel_t - sun_vel_t)

    print("Sun pos (SSB) [m]:", sun_pos_t)
    print("Earth pos (SSB) [m]:", earth_pos_t)
    print("Barycenter Sun-Earth pos (SSB) [m]:", barycenter_pos)

    state_hf_t_rlp = transformer.ecliptic_to_rlp(
        state_hf_ecl_t, t_check, earth_position=earth_pos_t, earth_velocity=earth_vel_t
    )

    print("State CRTBP (RLP) at t=1h:")
    print(state_crtbp_t)
    print("State HF -> Ecliptic at t=1h:")
    print(state_hf_ecl_t)
    print("State HF converted to RLP at t=1h:")
    print(state_hf_t_rlp)

    pos_diff = state_crtbp_t[:3] - state_hf_t_rlp[:3]
    vel_diff = state_crtbp_t[3:6] - state_hf_t_rlp[3:6]

    print(f"Position diff (m): {pos_diff}")
    print(f"Velocity diff (m/s): {vel_diff}")
    print(f"|Pos diff| (km): {np.linalg.norm(pos_diff)/1e3:.3f}")
    print(f"|Vel diff| (m/s): {np.linalg.norm(vel_diff):.6f}")

    # Compare accelerations
    acc_crtbp = crtbp.compute_acceleration(t_check, state_crtbp_t)
    acc_hf_at_ecl = hf.compute_acceleration(t_check, state_hf_ecl_t)

    print("Acc CRTBP (RLP frame) at t=1h:", acc_crtbp)
    print("Acc HF (ecliptic) at t=1h:", acc_hf_at_ecl)

    # --- Detailed transform of HF acceleration into RLP rotating frame ---
    def rotation_matrix_ecl_to_rlp(earth_pos):
        sun_pos, _ = ephem.get_body_state("SUN", et_check, "SSB", "J2000")
        sun_to_earth = earth_pos - sun_pos
        r_se = np.linalg.norm(sun_to_earth)
        x_axis = sun_to_earth / r_se
        z_axis = np.array([0.0, 0.0, 1.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        R = np.array([x_axis, y_axis, z_axis])
        return R

    # compute barycenter acceleration (inertial) to subtract
    # NOTE: HighFidelityDynamics._compute_gravitational_acceleration expects
    # the time in seconds since J2000, not an ET value. Pass t (seconds).
    acc_bary = hf._compute_gravitational_acceleration(t_check, barycenter_pos)

    # Now compute for multiple times (as in test): 2 hours sampled densely
    t_span_test = (t0, t0 + 2 * 3600.0)
    t_eval = np.linspace(t_span_test[0], t_span_test[1], 201)

    print("\nDetailed acceleration comparison over 2 hours:")
    print(" t(h) | |a_crtbp| (m/s2) | |a_hf_transformed| (m/s2) | diff | notes")
    acc_diffs = []
    a_crtbp_list = []
    a_hftrans_list = []
    for t in t_eval:
        # states
        s_crtbp = sol_crtbp.sol(t)
        s_hf_ecl = sol_hf.sol(t)

        et = ephem.et_from_j2000(t)
        earth_pos_t, earth_vel_t = ephem.get_body_state("EARTH", et, "SSB", "J2000")
        sun_pos_t, sun_vel_t = ephem.get_body_state("SUN", et, "SSB", "J2000")
        mu_ratio = Constants.MU_RATIO_SUN_EARTH
        bary_pos_t = sun_pos_t + mu_ratio * (earth_pos_t - sun_pos_t)

        # Transform HF state to RLP
        s_hf_rlp = transformer.ecliptic_to_rlp(
            s_hf_ecl, t, earth_position=earth_pos_t, earth_velocity=earth_vel_t
        )

        # HF inertial gravitational accel at spacecraft
        a_hf_sc = hf._compute_gravitational_acceleration(et, s_hf_ecl[:3])

        # Compute barycenter acceleration as mass-weighted average of Sun and Earth accelerations
        # Pass physical time 't' (seconds) to the HF acceleration routine
        a_sun = hf._compute_gravitational_acceleration(t, sun_pos_t)
        a_earth = hf._compute_gravitational_acceleration(t, earth_pos_t)
        a_hf_bary = (Constants.M_SUN * a_sun + Constants.M_EARTH * a_earth) / (
            Constants.M_SUN + Constants.M_EARTH
        )

        # Relative accel and rotate to RLP
        a_rel = a_hf_sc - a_hf_bary
        R = rotation_matrix_ecl_to_rlp(earth_pos_t)
        a_rel_rlp = R @ a_rel

        # Rotating-frame terms
        omega = np.array([0.0, 0.0, Constants.OMEGA_EARTH])
        r_rlp = s_hf_rlp[:3]
        v_rlp = s_hf_rlp[3:6]

        coriolis = -2.0 * np.cross(omega, v_rlp)
        centrifugal = -np.cross(omega, np.cross(omega, r_rlp))

        a_hf_transformed = a_rel_rlp + coriolis + centrifugal

        a_crtbp_loc = crtbp.compute_acceleration(t, s_crtbp)

        diff = a_crtbp_loc - a_hf_transformed
        diff_norm = np.linalg.norm(diff)

        acc_diffs.append(diff)
        a_crtbp_list.append(a_crtbp_loc)
        a_hftrans_list.append(a_hf_transformed)

        print(f"\n--- t = {(t-t0)/3600:.2f} h ---")
        print("a_hf_sc (inertial)       :", a_hf_sc, "| norm:", np.linalg.norm(a_hf_sc))
        print(
            "a_hf_bary (inertial)     :",
            a_hf_bary,
            "| norm:",
            np.linalg.norm(a_hf_bary),
        )
        print("a_rel (inertial)         :", a_rel, "| norm:", np.linalg.norm(a_rel))
        print(
            "a_rel rotated -> RLP      :",
            a_rel_rlp,
            "| norm:",
            np.linalg.norm(a_rel_rlp),
        )
        print(
            "coriolis term            :", coriolis, "| norm:", np.linalg.norm(coriolis)
        )
        print(
            "centrifugal term         :",
            centrifugal,
            "| norm:",
            np.linalg.norm(centrifugal),
        )
        print(
            "a_hf_transformed (RLP)   :",
            a_hf_transformed,
            "| norm:",
            np.linalg.norm(a_hf_transformed),
        )
        print(
            "a_crtbp_loc (RLP)        :",
            a_crtbp_loc,
            "| norm:",
            np.linalg.norm(a_crtbp_loc),
        )
        print("diff                     :", diff)

    print("\nDetailed comparison complete.")

    # Integrate acceleration difference to estimate delta-v and delta-r over the interval
    acc_diffs = np.array(acc_diffs)
    # trapezoidal integration over time to get Δv
    dv = np.trapz(acc_diffs, t_eval, axis=0)
    # second integral for approximate Δr (crude)
    vel_integrand = np.cumsum(acc_diffs, axis=0) * (t_eval[1] - t_eval[0])
    dr = np.trapz(vel_integrand, t_eval, axis=0)

    print("\nEstimated Δv from accel-diff (m/s):", dv)
    print("Estimated |Δv| (m/s):", np.linalg.norm(dv))

    # Actual observed Δv from propagation (transform HF final state to RLP)
    t_end = t_span_test[1]
    s_crtbp_end = sol_crtbp.sol(t_end)
    s_hf_end_ecl = sol_hf.sol(t_end)
    et_end = ephem.et_from_j2000(t_end)
    earth_pos_end, earth_vel_end = ephem.get_body_state("EARTH", et_end, "SSB", "J2000")
    s_hf_end_rlp = transformer.ecliptic_to_rlp(
        s_hf_end_ecl, t_end, earth_position=earth_pos_end, earth_velocity=earth_vel_end
    )

    dv_obs = s_crtbp_end[3:6] - s_hf_end_rlp[3:6]
    dr_obs = s_crtbp_end[:3] - s_hf_end_rlp[:3]

    print("Observed Δv (m/s):", dv_obs, "norm:", np.linalg.norm(dv_obs))
    print("Observed Δr (m):", dr_obs, "norm:", np.linalg.norm(dr_obs))


if __name__ == "__main__":
    main()
