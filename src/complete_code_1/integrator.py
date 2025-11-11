"""RK4 integrator for first-order state y = [r(3), v(3)].
Also contains helpers for propagating and detecting XZ crossings (y~0).
"""

import numpy as np
from dynamics import rotating_frame_acceleration


def state_derivative(state):
    """Given state vector [r(3), v(3)], return derivative [v, a(r,v)]."""
    r = state[:3]
    v = state[3:]
    a = rotating_frame_acceleration(r, v)
    return np.hstack((v, a))


def rk4_step(state, dt):
    """Single RK4 step for state vector (6,)."""
    k1 = state_derivative(state)
    k2 = state_derivative(state + 0.5 * dt * k1)
    k3 = state_derivative(state + 0.5 * dt * k2)
    k4 = state_derivative(state + dt * k3)
    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return new_state


def propagate_rk4(state0, t0, tf, dt, save_trajectory=True):
    """Propagate from t0 to tf with fixed step dt using RK4.
    Returns times array and states array (N,6) if save_trajectory True, else last state.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    nsteps = int(np.ceil((tf - t0) / dt))
    state = state0.copy()
    t = t0

    idx = 1
    if not save_trajectory:
        for i in range(nsteps):
            state = rk4_step(state, dt)
            t += dt

        return t, state

    traj = np.zeros((nsteps + 1, 6))
    times = np.zeros(nsteps + 1)
    traj[0] = state
    times[0] = t

    for i in range(nsteps):
        state = rk4_step(state, dt)
        t += dt
        traj[idx] = state
        times[idx] = t
        idx += 1

    return times, traj


def detect_xz_crossings(times, traj):
    """Detect indices where trajectory crosses y=0 plane (XZ plane).
    We detect sign changes of y coordinate and return the indices of crossing points
    using linear interpolation between steps for better accuracy.
    Returns a list of tuples: (t_cross, state_cross)
    """
    crossings = []
    y = traj[:, 1]
    for i in range(len(y) - 1):
        if y[i] == 0:
            crossings.append((times[i], traj[i].copy()))
        elif y[i] * y[i + 1] < 0:
            # linear interpolation for t and state
            alpha = abs(y[i]) / (abs(y[i]) + abs(y[i + 1]))
            t_cross = times[i] + alpha * (times[i + 1] - times[i])
            state_cross = traj[i] * (1 - alpha) + traj[i + 1] * alpha
            crossings.append((t_cross, state_cross))
    return crossings
