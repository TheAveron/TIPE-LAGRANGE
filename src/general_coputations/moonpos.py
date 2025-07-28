from constants import x_earth, d_em, x_sun, M_sun, M_earth, M_moon, omega
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from acceleration import compute_grav_accel_vec
from therocal_poisitions import compute_theorical_lagrangian_points


def compute_accel_with_moon(X, Y, x_moon, y_moon):
    """
    Compute total acceleration (Sun, Earth, Moon, centrifugal) at each grid point.
    """
    a_sun = compute_grav_accel_vec(X, Y, x_sun, 0, M_sun)
    a_earth = compute_grav_accel_vec(X, Y, x_earth, 0, M_earth)
    a_moon = compute_grav_accel_vec(X, Y, x_moon, y_moon, M_moon)

    a_centrifugal_x = omega**2 * X
    a_centrifugal_y = omega**2 * Y

    a_total_x = a_sun[0] + a_earth[0] + a_moon[0] + a_centrifugal_x
    a_total_y = a_sun[1] + a_earth[1] + a_moon[1] + a_centrifugal_y

    a_norm = np.sqrt(a_total_x**2 + a_total_y**2)
    return a_total_x, a_total_y, a_norm


def moon_orbit_positions(n_positions=8):
    angles = np.linspace(0, 2 * np.pi, n_positions, endpoint=False)
    x_moons = x_earth + d_em * np.cos(angles)
    y_moons = 0 + d_em * np.sin(angles)
    return list(zip(x_moons, y_moons))


def plot_moon_phase_effects(grid_size=1000, xy_lim=1.5e11):
    x = np.linspace(x_earth - xy_lim, x_earth + xy_lim, grid_size)
    y = np.linspace(-xy_lim, xy_lim, grid_size)
    X, Y = np.meshgrid(x, y)

    moon_pos = moon_orbit_positions(16)

    x_L1, x_L2, x_L3, (x_L4, y_L4), (x_L5, y_L5) = compute_theorical_lagrangian_points()

    for i, (x_m, y_m) in enumerate(moon_pos):
        a_x, a_y, a_norm = compute_accel_with_moon(X, Y, x_m, y_m)

        fig, ax = plt.subplots(figsize=(8, 8))

        plt.contourf(X, Y, np.log10(a_norm), levels=1000, cmap="coolwarm")
        plt.colorbar(label="log10(|acceleration|) [m/s²]")

        ## Placing Sun and Earth
        plt.scatter([x_earth, x_m], [0, y_m], color=["blue", "green"], s=80)
        # plt.text(x_sun, 0, "Sun", color="yellow", ha="right")
        plt.text(x_earth, 0, "Earth", color="blue", ha="right")
        plt.text(x_m, y_m, "Moon", color="green", ha="right")

        ## Placing Lagragian points
        plt.scatter([x_L1, x_L2], [0, 0], color="grey", s=40, label="Lagrange Points")

        plt.text(x_L1, 0, "L1", color="grey", ha="center", va="bottom")
        plt.text(x_L2, 0, "L2", color="grey", ha="center", va="bottom")

        plt.title(
            f"Net Acceleration Field (Sun-Earth-Moon, Rotating Barycentric Frame) - Moon phase {i+1} / 7"
        )
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid(False)
        plt.tight_layout()
        ax.set_aspect("equal", adjustable="box")
        # plt.show()

        plt.savefig(f"plots/acceleration_near_earth/fig{i}.png")
