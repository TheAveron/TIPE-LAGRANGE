"""State Transition Matrix integration (variational equations) for the rotating-frame dynamics.
Integrates augmented state [r, v, Phi(6x6)] with RK4 using the same dynamics as in dynamics.py
"""

import numpy as np
from dynamics import (
    rotating_frame_acceleration,
    r_sun,
    r_earth,
    omega_vec,
    G,
    M_sun,
    M_earth,
)


def cross_product_matrix(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def compute_dAdr(r):
    """Compute partial derivative of acceleration w.r.t position (3x3).
    Includes gravitational contributions from Sun and Earth and centrifugal matrix.
    """
    # gravitational part from Sun
    rs = r - r_sun
    re = r - r_earth
    rsn = np.linalg.norm(rs)
    ren = np.linalg.norm(re)
    I3 = np.eye(3)
    # avoid singularities
    if rsn < 1e-8 or ren < 1e-8:
        raise ValueError("Close approach to primary in compute_dAdr")

    def grav_term(rvec, M):
        rnorm = np.linalg.norm(rvec)
        rrT = np.outer(rvec, rvec)
        return -G * M * (I3 / rnorm**3 - 3.0 * rrT / rnorm**5)

    A_grav = grav_term(rs, M_sun) + grav_term(re, M_earth)
    # centrifugal contribution: - omega x (omega x r) = omega^2 * [x, y, 0]
    omega = omega_vec[2]
    A_cent = np.diag([omega**2, omega**2, 0.0])
    return A_grav + A_cent


def compute_dAdv():
    """Partial derivative of acceleration w.r.t velocity (3x3).
    Only Coriolis term depends on v: -2 * omega x v -> derivative is -2 * cross(omega).
    """
    Omega = cross_product_matrix(omega_vec)
    return -2.0 * Omega


def augmented_derivative(aug_state):
    """aug_state: length 6 + 36 flattened (r(3), v(3), Phi flatten 36)
    returns derivative of same length
    """
    r = aug_state[:3]
    v = aug_state[3:6]
    Phi = aug_state[6:].reshape((6, 6))
    a = rotating_frame_acceleration(r, v)
    # Build Jacobian F = [[0, I],[dAdr, dAdv]]
    dAdr = compute_dAdr(r)
    dAdv = compute_dAdv()
    F = np.zeros((6, 6))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 0:3] = dAdr
    F[3:6, 3:6] = dAdv
    # Phi_dot = F * Phi
    Phi_dot = F.dot(Phi)
    deriv = np.zeros_like(aug_state)
    deriv[:3] = v
    deriv[3:6] = a
    deriv[6:] = Phi_dot.reshape(36)
    return deriv


def integrate_stm(state0, t0, tf, dt):
    """Integrate augmented state using RK4. state0 is 6-vector [r,v].
    Returns final Phi (6x6) and trajectory of states if needed.
    """
    nsteps = int(np.ceil((tf - t0) / dt))
    # initial augmented state
    Phi0 = np.eye(6)
    aug = np.zeros(6 + 36)
    aug[:6] = state0
    aug[6:] = Phi0.reshape(36)
    t = t0
    for _ in range(nsteps):
        # RK4 on augmented
        k1 = augmented_derivative(aug)
        k2 = augmented_derivative(aug + 0.5 * dt * k1)
        k3 = augmented_derivative(aug + 0.5 * dt * k2)
        k4 = augmented_derivative(aug + dt * k3)
        aug = aug + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    Phi_final = aug[6:].reshape((6, 6))
    state_final = aug[:6]
    return state_final, Phi_final
