"""Example usage: compute L2, place a spacecraft near L2, propagate and (optionally) run a simple correction.


Save the above file sections into separate .py files in the same folder and run this script.
"""

# example_main.py
import numpy as np
from constants import DEFAULT_DT
from l2 import compute_L2
from station_keeping import run_station_keeping
from plotting import plot_trajectory_3d, plot_projections, plot_dv_log
from integrator import propagate_rk4

from frames import inertial_to_rotating


def checks(state0):
    from l2 import compute_L2, analytic_L2_distance_from_earth
    from dynamics import grav_acceleration
    from constants import x_earth, x_sun, a

    print("=== SANITY CHECKS ===")
    print("Earth position x_earth =", x_earth)
    x_L2 = compute_L2()
    print("Computed L2 x coordinate:", x_L2)
    print("Analytic Earth->L2 distance (approx):", analytic_L2_distance_from_earth())
    # distance Earth->L2
    print("Distance Earth->L2 (km):", abs(x_L2 - x_earth))
    # check gravitational accel direction at initial point
    r0 = state0[:3]
    a0 = grav_acceleration(r0)
    print("Grav accel at r0:", a0, "norm", np.linalg.norm(a0))
    # distances to primaries
    dist_sun = np.linalg.norm(r0 - np.array([x_sun, 0, 0]))
    dist_earth = np.linalg.norm(r0 - np.array([x_earth, 0, 0]))
    print("Distance to Sun (AU):", dist_sun / a)
    print("Distance to Earth (km):", dist_earth)


def example_run_full():
    print("Chargement des données réelles JWST...")

    # Données NASA (2022-01-24 au moment de la mise en orbite L2)
    r_I = np.array([1.499073e8, 3.97968e5, -1.83603e4])
    v_I = np.array([-1.23229, -29.26776, 0.119171])

    # temps initial t0 = 0
    t0 = 0.0
    dt = DEFAULT_DT

    # conversion inertiel \u2192 rotatif
    r0, v0 = inertial_to_rotating(r_I, v_I, t0)
    state0 = np.hstack((r0, v0))

    checks(state0)

    print("Propagation 20 jours réels...")
    tf = 20 * 24 * 3600
    times, traj = propagate_rk4(state0, t0, tf, dt, save_trajectory=True)

    plot_trajectory_3d(times, traj)
    plot_projections(times, traj)


if __name__ == "__main__":

    example_run_full()


# End of multi-file package
