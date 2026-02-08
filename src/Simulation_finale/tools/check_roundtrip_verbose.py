import os
import sys

import numpy as np

# Make project `src` package importable when running scripts from `tools/`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.constants import Constants
from src.simulation.coordinates import CoordinateTransformer
from src.simulation.Ephem_handler import EphemerisManager


def fmt(v):
    return f"[{v[0]:.9e}, {v[1]:.9e}, {v[2]:.9e}]"


def main():
    ephem = EphemerisManager()
    t_test = 0.0
    et = ephem.et_from_j2000(t_test)
    earth_pos, earth_vel = ephem.get_body_state("EARTH", et, "SSB", "J2000")

    transformer = CoordinateTransformer(include_moon=True)

    # L2 theoretical and small orbit as in tests
    l2_pos_rlp = transformer.compute_l2_position_rlp()
    A_y = 100e6
    A_z = 50e6
    pos_rlp_0 = np.array([l2_pos_rlp[0], A_y, A_z])
    omega = Constants.OMEGA_EARTH
    vel_rlp_0 = np.array([0.0, 0.0, -omega * A_y])
    state_rlp = np.concatenate([pos_rlp_0, vel_rlp_0])

    print("State RLP original:")
    print("  pos:", fmt(state_rlp[:3]))
    print("  vel:", fmt(state_rlp[3:6]))

    # Convert to ecliptic using ephemerides
    state_ecl = transformer.rlp_to_ecliptic(
        state_rlp, t_test, earth_position=earth_pos, earth_velocity=earth_vel
    )

    print("\nState Ecliptic:")
    print("  pos:", fmt(state_ecl[:3]))
    print("  vel:", fmt(state_ecl[3:6]))

    # Convert back
    state_rlp_check = transformer.ecliptic_to_rlp(
        state_ecl, t_test, earth_position=earth_pos, earth_velocity=earth_vel
    )

    print("\nState RLP reconverti:")
    print("  pos:", fmt(state_rlp_check[:3]))
    print("  vel:", fmt(state_rlp_check[3:6]))

    diff_pos = state_rlp_check[:3] - state_rlp[:3]
    diff_vel = state_rlp_check[3:6] - state_rlp[3:6]

    print("\nDifférences (reconverti - original):")
    print("  pos diff:", fmt(diff_pos))
    print("  vel diff:", fmt(diff_vel))

    print("\nNormes:")
    print(
        f"  |Δpos| = {np.linalg.norm(diff_pos):.9e} m ({np.linalg.norm(diff_pos)/1e3:.6f} km)"
    )
    print(f"  |Δvel| = {np.linalg.norm(diff_vel):.9e} m/s")

    # Per-component relative (where applicable)
    pos_rel = np.abs(diff_pos) / (np.abs(state_rlp[:3]) + 1e-20)
    vel_rel = np.abs(diff_vel) / (np.abs(state_rlp[3:6]) + 1e-20)
    print("\nRelative differences (abs/abs(original)):")
    print("  pos rel:", fmt(pos_rel))
    print("  vel rel:", fmt(vel_rel))


if __name__ == "__main__":
    main()
