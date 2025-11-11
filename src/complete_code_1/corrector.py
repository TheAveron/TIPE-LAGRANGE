"""Simple station-keeping corrector using a differential approach:
- apply an impulsive dv at current time in a chosen direction (unit vector)
- propagate until the 4th XZ crossing
- evaluate target function (vx at that crossing)
- use secant method to find dv magnitude that zeros the target


This is a pragmatic, numerically straightforward implementation.
"""

import numpy as np
from integrator import propagate_rk4, detect_xz_crossings


def apply_impulse(state, dv_vec):
    """Apply instantaneous velocity impulse dv_vec to state (6,)."""
    new_state = state.copy()
    new_state[3:] += dv_vec
    return new_state


def find_dv_for_vx_zero(
    state0, t0, dt, direction, dv_guess1, dv_guess2, max_iter=10, tol=1e-6
):
    """Find magnitude of dv along 'direction' that makes vx ~= 0 at the 4th XZ crossing.
    direction must be unit vector in inertial/rotating frame coordinates.
    dv_guess1/2 are initial magnitudes (km/s).
    Returns dv_vector (3,), and diagnostics.
    """

    def target(dv_mag):
        dv = dv_mag * direction
        s1 = apply_impulse(state0, dv)
        # propagate sufficiently long to find at least 4 crossings
        tf = t0 + 20 * 24 * 3600.0  # 20 days; conservative
        times, traj = propagate_rk4(s1, t0, tf, dt, save_trajectory=True)
        crossings = detect_xz_crossings(times, traj)
        if len(crossings) < 4:
            # penalize: return large residual
            return 1e3 + (4 - len(crossings)) * 1e2
        # take the 4th crossing state
        t_cross, state_cross = crossings[3]
        vx = state_cross[3]
        return vx

    f1 = target(dv_guess1)
    f2 = target(dv_guess2)
    if abs(f1) < tol:
        return dv_guess1 * direction, {"dv": dv_guess1, "f": f1}
    if abs(f2) < tol:
        return dv_guess2 * direction, {"dv": dv_guess2, "f": f2}

    for i in range(max_iter):
        # secant step
        if (f2 - f1) == 0:
            break
        dv_new = dv_guess2 - f2 * (dv_guess2 - dv_guess1) / (f2 - f1)
        f_new = target(dv_new)
        if abs(f_new) < tol:
            return dv_new * direction, {"dv": dv_new, "f": f_new}
        # shift
        dv_guess1, f1 = dv_guess2, f2
        dv_guess2, f2 = dv_new, f_new
        # if not converged, return best estimate
    return dv_guess2 * direction, {"dv": dv_guess2, "f": f2}
