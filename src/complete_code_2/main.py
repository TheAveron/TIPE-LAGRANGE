import numpy as np
from integrator import integrate_particle_rk4
from l2 import compute_L2
from plotting import plot_projections, plot_trajectory_3d
from profiler import profile_with_memory

x_L2 = compute_L2()


@profile_with_memory
def main():
    r0 = np.array([x_L2 + 4e3, -7e3, -4e3])
    v0 = np.array([0, -0.0172, 0.002])

    r_list, _ = integrate_particle_rk4(r0, v0, dt=3600, t_max=1e9)

    print("Affichage ...")
    plot_trajectory_3d(r_list)
    # plot_projections(r_list)


if __name__ == "__main__":
    main()
