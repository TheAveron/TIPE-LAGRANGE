import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.constants import Constants
from src.simulation.coordinates import CoordinateTransformer
from src.simulation.CRTBP_model_dynamics import CRTBP3Body
from src.simulation.dynamics_conf import DynamicsConfig, DynamicsModel
from src.simulation.Ephem_handler import EphemerisManager
from src.simulation.MHF_ephem_dynamics import HighFidelityDynamics


def rotation_matrix_ecl_to_rlp_from_earth(ephem, et, earth_pos):
    sun_pos, _ = ephem.get_body_state("SUN", et, "SSB", "J2000")
    sun_to_earth = earth_pos - sun_pos
    x_axis = sun_to_earth / np.linalg.norm(sun_to_earth)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    R = np.array([x_axis, y_axis, z_axis])
    return R


def compute_grav_acc(mu, r_body, r_sc):
    r_vec = r_body - r_sc
    r = np.linalg.norm(r_vec)
    if r < 1.0:
        return np.zeros(3)
    return mu * r_vec / r**3


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

    # initial state
    l2 = transformer.compute_l2_position_rlp()
    A_y = 100e6
    A_z = 50e6
    pos_rlp_0 = np.array([l2[0], A_y, A_z])
    vel_rlp_0 = np.array([0.0, 0.0, -Constants.OMEGA_EARTH * A_y])
    state_rlp = np.concatenate([pos_rlp_0, vel_rlp_0])

    et0 = ephem.et_from_j2000(t0)
    earth_pos0, earth_vel0 = ephem.get_body_state("EARTH", et0, "SSB", "J2000")
    state_ecl0 = transformer.rlp_to_ecliptic(
        state_rlp, t0, earth_position=earth_pos0, earth_velocity=earth_vel0
    )

    # propagate short 2h as before
    t_span = (t0, t0 + 2 * 3600.0)
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

    t_eval = np.linspace(t_span[0], t_span[1], 201)

    diffs_no_moon = []
    diffs_with_moon = []

    print(
        "Analyzing terms and checking Moon contribution over 2 hours (201 samples)..."
    )

    for t in t_eval:
        et = ephem.et_from_j2000(t)
        earth_pos_t, earth_vel_t = ephem.get_body_state("EARTH", et, "SSB", "J2000")
        sun_pos_t, sun_vel_t = ephem.get_body_state("SUN", et, "SSB", "J2000")
        moon_pos_t, moon_vel_t = ephem.get_body_state("MOON", et, "SSB", "J2000")

        s_crtbp = sol_crtbp.sol(t)
        s_hf_ecl = sol_hf.sol(t)
        s_hf_rlp = transformer.ecliptic_to_rlp(
            s_hf_ecl, t, earth_position=earth_pos_t, earth_velocity=earth_vel_t
        )

        # CRTBP acceleration (already in RLP)
        a_crtbp = crtbp.compute_acceleration(t, s_crtbp)

        # breakdown CRTBP contributions using same formulas
        x, y, z = s_crtbp[0], s_crtbp[1], s_crtbp[2]
        r1 = np.sqrt((x - crtbp.x1) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - crtbp.x2) ** 2 + y**2 + z**2)
        # protections
        r1 = max(r1, 1.0)
        r2 = max(r2, 1.0)
        # contributions
        grav1 = -crtbp.GM_1 * (x - crtbp.x1) / r1**3
        grav2 = -crtbp.GM_2 * (x - crtbp.x2) / r2**3
        # note: those are scalar x-components; compute vector contributions
        r1_vec = np.array([x - crtbp.x1, y, z])
        r2_vec = np.array([x - crtbp.x2, y, z])
        grav1_vec = -crtbp.GM_1 * r1_vec / (r1**3)
        grav2_vec = -crtbp.GM_2 * r2_vec / (r2**3)
        centrifugal = np.array([crtbp.omega**2 * x, crtbp.omega**2 * y, 0.0])
        coriolis = -2.0 * np.cross(np.array([0.0, 0.0, crtbp.omega]), s_crtbp[3:6])

        # HF accel and transformed
        a_hf_sc = hf._compute_gravitational_acceleration(et, s_hf_ecl[:3])
        a_sun = hf._compute_gravitational_acceleration(et, sun_pos_t)
        a_earth = hf._compute_gravitational_acceleration(et, earth_pos_t)
        a_hf_bary = (Constants.M_SUN * a_sun + Constants.M_EARTH * a_earth) / (
            Constants.M_SUN + Constants.M_EARTH
        )
        a_rel = a_hf_sc - a_hf_bary

        R = rotation_matrix_ecl_to_rlp_from_earth(ephem, et, earth_pos_t)
        a_rel_rlp = R @ a_rel
        coriolis_hf = -2.0 * np.cross(
            np.array([0.0, 0.0, Constants.OMEGA_EARTH]), s_hf_rlp[3:6]
        )
        centrifugal_hf = -np.cross(
            np.array([0.0, 0.0, Constants.OMEGA_EARTH]),
            np.cross(np.array([0.0, 0.0, Constants.OMEGA_EARTH]), s_hf_rlp[:3]),
        )
        a_hf_trans = a_rel_rlp + coriolis_hf + centrifugal_hf

        diff = a_crtbp - a_hf_trans
        diffs_no_moon.append(np.linalg.norm(diff))

        # Moon contribution (inertial): a_moon_sc and a_moon_bary
        a_moon_sc = compute_grav_acc(Constants.MU_MOON, moon_pos_t, s_hf_ecl[:3])
        a_moon_bary = compute_grav_acc(
            Constants.MU_MOON,
            moon_pos_t,
            (sun_pos_t + Constants.MU_RATIO_SUN_EARTH * (earth_pos_t - sun_pos_t)),
        )
        # relative moon accel
        a_moon_rel = a_moon_sc - a_moon_bary
        a_moon_rel_rlp = R @ a_moon_rel

        # Add moon_rel_rlp to CRTBP and compute new diff
        diff_with_moon = (a_crtbp + a_moon_rel_rlp) - a_hf_trans
        diffs_with_moon.append(np.linalg.norm(diff_with_moon))

    diffs_no_moon = np.array(diffs_no_moon)
    diffs_with_moon = np.array(diffs_with_moon)

    print("\nResults:")
    print(f"  Mean diff norm (no moon): {np.mean(diffs_no_moon):.6e} m/s^2")
    print(f"  Mean diff norm (with moon): {np.mean(diffs_with_moon):.6e} m/s^2")
    print(f"  Max diff norm (no moon): {np.max(diffs_no_moon):.6e} m/s^2")
    print(f"  Max diff norm (with moon): {np.max(diffs_with_moon):.6e} m/s^2")

    red_mean = (np.mean(diffs_no_moon) - np.mean(diffs_with_moon)) / np.mean(
        diffs_no_moon
    )
    print(f"  Relative reduction in mean diff by adding Moon: {red_mean*100:.3f}%")

    # print sample at t0 and t_end
    print("\nSample at t0:")
    print("  diff no moon:", diffs_no_moon[0], "diff with moon:", diffs_with_moon[0])
    print("Sample at tend:")
    print("  diff no moon:", diffs_no_moon[-1], "diff with moon:", diffs_with_moon[-1])


if __name__ == "__main__":
    main()
