"""High-level station-keeping routines: compute STM/monodromy, extract stable eigenvector,
and run correction cycles using corrector.find_dv_for_vx_zero.
"""

import numpy as np
from stm import integrate_stm
from integrator import propagate_rk4, detect_xz_crossings
from corrector import find_dv_for_vx_zero, apply_impulse


def compute_stable_eigenvector(state0, t0, tf, dt):
    """Integrate STM from t0 to tf starting from state0 and compute stable eigenvector
    of the monodromy matrix Phi(tf,t0). Returns eigenvector (6,) normalized, and eigenvalue.
    """
    _, Phi = integrate_stm(state0, t0, tf, dt)
    eigvals, eigvecs = np.linalg.eig(Phi)
    # find eigenvalue with magnitude < 1 (stable)
    mags = np.abs(eigvals)
    idx = np.argmin(mags)  # smallest magnitude
    eigval = eigvals[idx]
    eigvec = eigvecs[:, idx]
    # normalize so position part has unit norm
    pos = eigvec[:3]
    if np.linalg.norm(pos) < 1e-12:
        eigvec = eigvec / np.linalg.norm(eigvec)
    else:
        eigvec = eigvec / np.linalg.norm(pos)
    return eigvec.real, eigval.real


def run_station_keeping(initial_state, t0, dt, cycles=3, dv_guess_scale=1e-4):
    """Run a simple station-keeping loop:
    - propagate to get a reference crossing period (time between two successive XZ crossings)
    - compute STM over one period to obtain stable eigenvector
    - for given number of cycles: at each station-keeping event compute dv along stable direction
    using the differential corrector to zero vx at the 4th crossing, apply it, and continue.
    """
    log = []
    # first propagate to get reference period: integrate for up to 10 days to find crossings
    tf_guess = t0 + 40 * 24 * 3600.0
    times, traj = propagate_rk4(initial_state, t0, tf_guess, dt, save_trajectory=True)
    crossings = detect_xz_crossings(times, traj)
    if len(crossings) < 2:
        raise RuntimeError(
            "Not enough crossings to estimate period; increase propagation time"
        )
    # use time between first two crossings as period
    t1 = crossings[0][0]
    t2 = crossings[1][0]
    period = t2 - t1
    print(f"Estimated crossing period: {period/3600.0:.3f} hours")
    # compute stable eigenvector over one period using state at first crossing
    state_at_t1 = crossings[0][1]
    eigvec, eigval = compute_stable_eigenvector(state_at_t1, t1, t1 + period, dt)
    # direction is position part of eigvec
    direction = eigvec[:3]
    direction = direction / np.linalg.norm(direction)
    print("Stable direction (position part, unit):", direction)

    state = initial_state.copy()
    current_time = t0
    total_dv = 0.0
    for cycle in range(cycles):
        print(f"=== Cycle {cycle+1} ===")
        # propagate until we have at least 4 crossings
        tf = current_time + 20 * 24 * 3600.0
        times, traj = propagate_rk4(state, current_time, tf, dt, save_trajectory=True)
        crossings = detect_xz_crossings(times, traj)
        if len(crossings) < 4:
            print("Warning: less than 4 crossings found; extending propagation")
            tf = current_time + 40 * 24 * 3600.0
            times, traj = propagate_rk4(
                state, current_time, tf, dt, save_trajectory=True
            )
            crossings = detect_xz_crossings(times, traj)
            if len(crossings) < 4:
                raise RuntimeError("Failed to get 4 crossings for correction")
        # we'll attempt correction at current_time (impulse now)
        # prepare secant initial guesses
        dv0 = 0.0
        dv1 = dv_guess_scale
        dv_vec, info = find_dv_for_vx_zero(
            state, current_time, dt, direction, dv0, dv1, max_iter=8, tol=1e-5
        )
        print(f'Applying dv magnitude {info["dv"]:.6e} km/s, residual {info["f"]:.3e}')
        total_dv += abs(info["dv"])
        state = apply_impulse(state, dv_vec)
        # advance time a small amount to avoid re-detecting same crossing
        current_time += 60.0
        log.append(
            {
                "cycle": cycle + 1,
                "dv": info["dv"],
                "residual": info["f"],
                "eigval": eigval,
            }
        )
    return state, total_dv, log
