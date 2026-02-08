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


def main():
    t0 = 0.0
    t_end = 10.0  # 10 seconds
    ephem = EphemerisManager()
    transformer = CoordinateTransformer(include_moon=True, ephem_manager=ephem)

    crtbp = CRTBP3Body(DynamicsConfig(model=DynamicsModel.CRTBP), normalized=False)
    hf = HighFidelityDynamics(
        DynamicsConfig(
            model=DynamicsModel.EPHEMERIS, include_moon=True, include_srp=False
        ),
        ephemeris_manager=ephem,
    )

    # initial state (same as tests)
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

    # propagate both for 10 s
    sol_crtbp = solve_ivp(
        crtbp.equations_of_motion,
        (t0, t_end),
        state_rlp,
        t_eval=np.linspace(t0, t_end, 21),
        rtol=1e-12,
        atol=1e-12,
    )
    sol_hf = solve_ivp(
        hf.equations_of_motion,
        (t0, t_end),
        state_ecl0,
        t_eval=np.linspace(t0, t_end, 21),
        rtol=1e-12,
        atol=1e-12,
    )

    print(
        "t(s) | |Δpos|(m) | |Δvel|(m/s) | pos diff components (m) | vel diff components (m/s)"
    )
    for i, t in enumerate(sol_crtbp.t):
        s_crtbp = sol_crtbp.y[:, i]
        s_hf_ecl = sol_hf.y[:, i]
        et = ephem.et_from_j2000(t)
        earth_pos_t, earth_vel_t = ephem.get_body_state("EARTH", et, "SSB", "J2000")
        s_hf_rlp = transformer.ecliptic_to_rlp(
            s_hf_ecl, t, earth_position=earth_pos_t, earth_velocity=earth_vel_t
        )

        dp = s_crtbp[:3] - s_hf_rlp[:3]
        dv = s_crtbp[3:6] - s_hf_rlp[3:6]
        print(
            f"{t:4.1f} | {np.linalg.norm(dp):12.6e} | {np.linalg.norm(dv):12.6e} | {dp} | {dv}"
        )


if __name__ == "__main__":
    main()
