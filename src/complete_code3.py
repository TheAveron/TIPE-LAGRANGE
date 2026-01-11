"""
High-precision restricted 3-body / 3-body integrator
---------------------------------------------------
This script implements a careful, physically-consistent integration of a
massless particle in the gravitational field of Sun + Earth using a
barycentric, inertial N-body formulation for the primaries and a
symplectic integrator (4th-order Yoshida composition of velocity-Verlet)
for long-term accuracy.

Key design choices for precision:
 - Work in SI-consistent units (kilometres for distance, seconds for time,
   kilograms for mass) to stay compatible with your previous code while
   keeping magnitudes readable.
 - Use barycentric initial conditions for Sun and Earth (center-of-mass at
   origin, velocities such that the two-body motion is circular).
 - Integrate ALL massive bodies (Sun, Earth) and the massless particle in
   the same integrator so there are no inconsistent fixed primaries.
 - Use a symplectic integrator (velocity-Verlet) composed to 4th order
   (Yoshida) for much better energy behaviour over long times.
 - Provide utilities to transform states to a rotating frame with the
   correct angular velocity (computed from the two-body motion) for
   diagnostics or plotting.

Limitations & notes:
 - The code intentionally avoids external dependencies beyond numpy and matplotlib.
 - For highest precision one could use a high-order adaptive integrator
   (e.g. IAS15), but that requires external libraries; symplectic integrator
   is preferred for long-term qualitative stability of orbits near Lagrange
   points.
 - Radiation pressure, J2, solar mass-loss, relativistic corrections,
   and non-gravitational forces are NOT included. Add them if needed.

Author: ChatGPT (GPT-5 Thinking mini)
"""

from math import sqrt

import numpy as np

# Physical constants and units
G = 6.67430e-20  # km^3 / kg / s^2 (as in user's original code)

# Masses
M_sun = 1.98847e30  # kg
M_earth = 5.97219e24  # kg

# Mean Sun-Earth distance
AU = 149597870.7  # km

# Derived quantities for two-body
mu_total = M_sun + M_earth

# Place primaries initially on x-axis so that barycenter is at origin.
# Let Earth be at +a * (M_sun / mu_total) and Sun at -a * (M_earth / mu_total)
# so that r_earth - r_sun = a. We choose a = AU.

a = AU
x_earth = a * (M_sun / mu_total)
x_sun = -a * (M_earth / mu_total)

# Angular rate of circular two-body motion (rad/s)
omega = np.sqrt(G * mu_total / a**3)
omega_vec = np.array([0.0, 0.0, omega])

# Helper functions


def accel_pairwise(positions, masses):
    """Compute accelerations on each body due to all other bodies.

    positions: (N,3) array
    masses: length-N array

    returns accelerations array shape (N,3)
    """
    N = len(masses)
    a = np.zeros_like(positions)
    for i in range(N):
        pi = positions[i]
        ai = np.zeros(3)
        for j in range(N):
            if i == j:
                continue
            r = pi - positions[j]
            r2 = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
            # Softening is intentionally NOT used; if you need to avoid
            # singularities for extremely close encounters, add softening.
            ai += -G * masses[j] * r / (r2 * np.sqrt(r2))
        a[i] = ai
    return a


def compute_two_body_circular_velocities(r1, r2, m1, m2):
    """Given positions r1,r2 and masses, compute velocity vectors for a
    circular two-body orbit about the barycenter in the xy-plane.

    The velocities are perpendicular to the line connecting the bodies and
    have magnitudes such that centripetal = gravitational.
    Returns v1, v2 (each 3-vector).
    """
    mu = m1 + m2
    rel = r2 - r1
    R = np.linalg.norm(rel)
    # angular speed for circular motion with separation R
    w = np.sqrt(G * mu / R**3)
    # direction perpendicular (choose +y for earth positive x)
    # unit vector along rel
    ex = rel / R
    ey = np.array([-ex[1], ex[0], 0.0])  # rotate by +90deg in xy-plane
    v1 = w * np.cross([0, 0, 1.0], r1)  # alternative formula
    v2 = w * np.cross([0, 0, 1.0], r2)
    # ensure velocities give the correct opposite directions
    return v1, v2


# Symplectic integrator: velocity-Verlet (kick-drift-kick)
# Compose it to 4th order using Yoshida coefficients


def velocity_verlet_step(positions, velocities, masses, dt):
    """Single velocity-Verlet step for all bodies.

    positions, velocities: shape (N,3)
    masses: (N,)
    dt: scalar

    returns updated positions, velocities
    """
    a = accel_pairwise(positions, masses)
    # half-kick
    v_half = velocities + 0.5 * dt * a
    # drift
    r_new = positions + dt * v_half
    # compute new accel
    a_new = accel_pairwise(r_new, masses)
    # half-kick
    v_new = v_half + 0.5 * dt * a_new
    return r_new, v_new


# Yoshida 4th-order composition coefficients (coefficients s1,s2,s3)
# See H. Yoshida, "Construction of higher order symplectic integrators"
YOSHIDA_COEFFS = {
    "w0": -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0)),
    "w1": 1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
}


def yoshida4_step(positions, velocities, masses, dt):
    """One Yoshida 4th-order composed symplectic step.

    It performs three velocity-verlet substeps with specially chosen
    coefficients to achieve global 4th order accuracy while remaining
    symplectic.
    """
    w0 = YOSHIDA_COEFFS["w0"]
    w1 = YOSHIDA_COEFFS["w1"]
    # sequence: w1, w0, w1
    r, v = positions.copy(), velocities.copy()
    r, v = velocity_verlet_substep(r, v, masses, w1 * dt)
    r, v = velocity_verlet_substep(r, v, masses, w0 * dt)
    r, v = velocity_verlet_substep(r, v, masses, w1 * dt)
    return r, v


def velocity_verlet_substep(positions, velocities, masses, dt):
    # simple alias to keep intent clear
    return velocity_verlet_step(positions, velocities, masses, dt)


def integrate_yoshida(positions, velocities, masses, dt, t_max, store_interval=1):
    """Integrate the system using Yoshida 4th-order symplectic integrator.

    positions, velocities: numpy arrays shape (N,3)
    masses: array length N
    dt: timestep (s)
    t_max: total time (s)
    store_interval: store every `store_interval` steps

    returns times, pos_hist, vel_hist
    """
    nsteps = int(np.ceil(t_max / dt))
    nstore = (nsteps // store_interval) + 1

    pos_hist = np.zeros((nstore, positions.shape[0], 3))
    vel_hist = np.zeros_like(pos_hist)
    times = np.zeros(nstore)

    idx = 0
    pos_hist[idx] = positions.copy()
    vel_hist[idx] = velocities.copy()
    times[idx] = 0.0

    t = 0.0
    for step in range(1, nsteps + 1):
        positions, velocities = yoshida4_step(positions, velocities, masses, dt)
        t += dt
        if step % store_interval == 0:
            idx += 1
            pos_hist[idx] = positions.copy()
            vel_hist[idx] = velocities.copy()
            times[idx] = t
    return times, pos_hist, vel_hist


# Utilities to work with rotating frame (Sun-Earth rotating with mean motion)


def to_rotating_frame(positions, t0, omega=omega):
    """Rotate array of positions (N,3) by angle theta = omega * t0 around z-axis.

    This rotates inertial -> rotating by applying R(-theta) so that a point fixed
    in rotation frame stays fixed.
    """
    th = omega * t0
    c = np.cos(-th)
    s = np.sin(-th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return positions @ R.T


def from_rotating_frame(positions, t0, omega=omega):
    th = omega * t0
    c = np.cos(th)
    s = np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return positions @ R.T


# Example driver: set initial conditions for Sun, Earth (circular two-body), plus particle near L2


def example_simulation():
    # Bodies: [Sun, Earth, Particle]
    masses = np.array([M_sun, M_earth, 0.0])

    r_sun = np.array([x_sun, 0.0, 0.0])
    r_earth = np.array([x_earth, 0.0, 0.0])

    # Compute circular velocities for Sun and Earth about barycenter
    v_sun, v_earth = compute_two_body_circular_velocities(
        r_sun, r_earth, M_sun, M_earth
    )

    # Particle initial near L2: we'll compute L2 using a 1D Newton solve in the rotating frame
    x_L2_physical = compute_L2_physical_distance()
    # Place particle a few thousand km beyond L2 along +x and a small transverse kick
    r_particle = np.array([x_L2_physical + 4000.0, -7000.0, -4000.0])
    v_particle = np.array([0.0, -0.0172, 0.002])

    positions = np.vstack([r_sun, r_earth, r_particle])
    velocities = np.vstack([v_sun, v_earth, v_particle])

    # Integrate
    dt = 600.0  # 10 min timesteps
    tmax = 3600.0 * 24.0 * 365.25 * 1.0  # 1 year as example

    times, pos_hist, vel_hist = integrate_yoshida(
        positions, velocities, masses, dt, tmax, store_interval=10
    )

    return times, pos_hist, vel_hist


# Compute L2 approximately in physical units using CR3BP cubic approximation then refine


def analytic_L2_distance_from_earth_physical():
    mu = M_earth / (M_sun + M_earth)
    return a * (mu / 3.0) ** (1.0 / 3.0)


def compute_L2_physical_distance(tol=1e-8, maxiter=80):
    # Newton on x-axis in rotating frame where primaries are at x_sun, x_earth
    d_guess = analytic_L2_distance_from_earth_physical()
    x = x_earth + d_guess

    def f(xp):
        r = np.array([xp, 0.0, 0.0])
        v = np.zeros(3)
        # rotating_frame_acceleration uses r_earth inferred from global x_earth; we'll
        # reimplement here in pure inertial->rotating check: compute total accel in rotating frame
        # compute gravitational accel in inertial from sun and earth
        rs = r - np.array([x_sun, 0.0, 0.0])
        re = r - np.array([x_earth, 0.0, 0.0])
        norm_rs = np.linalg.norm(rs)
        norm_re = np.linalg.norm(re)
        a_sun = -G * M_sun * rs / (norm_rs**3)
        a_earth = -G * M_earth * re / (norm_re**3)
        a_grav = a_sun + a_earth
        # centrifugal term in rotating frame: omega^2 * x (x-component)
        cent = omega**2 * xp
        # Coriolis is zero because v=0
        return a_grav[0] + cent

    for i in range(maxiter):
        fx = f(x)
        h = max(1e-3, abs(x) * 1e-8)
        fpx = (f(x + h) - f(x - h)) / (2.0 * h)
        if abs(fpx) < 1e-16:
            break
        dx = -fx / fpx
        x += dx
        if abs(dx) < tol:
            break
    return x


if __name__ == "__main__":
    # simple run
    times, pos_hist, vel_hist = example_simulation()
    # Write a short summary printout
    print("Simulation finished. Stored steps:", len(times))
    # Example: print final particle state
    final_pos = pos_hist[-1, 2]
    final_vel = vel_hist[-1, 2]
    print("Final particle position (km):", final_pos)
    print("Final particle velocity (km/s):", final_vel)
