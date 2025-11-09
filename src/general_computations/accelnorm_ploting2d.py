import matplotlib.pyplot as plt
from numpy import int64, log10, min, max, ndarray, float64
from typing import Any
from constants import x_earth, x_sun, x_moon
from therocal_poisitions import compute_theorical_lagrangian_points
from acceleration import compute_acceleration_norm, compute_acceleration_vect
from integrator import integrate_particle
import numpy as np


def simulate_particle(X, Y, Z, a_norm: np.ndarray, a_total_vec, lagrange_points):
    x_L2 = lagrange_points[1]
    x0 , y0, z0= x_L2 + 4.0e8, -1.0e5, -1.0e8  # -100 000 km in z (m)

    T_orbit = 168.0 * 86400.0  # secondes (~168 days)
    r0 = 4.0e8  # metres (400000 km)
    v0_norm = (
        2.0 * np.pi * r0 / T_orbit
    )  # vitesse tangentiel (~173 m/s pour r0=4e8)

    dx = x0 - x_L2
    dy = y0
    norm_xy = np.sqrt(dx * dx + dy * dy)
    if norm_xy < 1e-12:
        norm_xy = 1.0
    ux_t = -dy / norm_xy  # rotaion de 90 deg pour l'approximation tangentielle
    uy_t = dx / norm_xy
    uz_t = 0.0

    vx0 = v0_norm * ux_t
    vy0 = v0_norm * uy_t
    vz0 = v0_norm * uz_t

    Nx, Ny, Nz, _ = a_total_vec.shape
    x_vals = np.linspace(X.min(), X.max(), Nx)
    y_vals = np.linspace(Y.min(), Y.max(), Ny)
    z_vals = np.linspace(Z.min(), Z.max(), Nz)

    t_max = 3.0 * T_orbit # simulation de 3 orbites
    nsteps = 200000 # Nombre de points

    x_list, y_list, z_list = integrate_particle(
        x0,
        y0,
        z0,
        vx0,
        vy0,
        vz0,
        x_vals,
        y_vals,
        z_vals,
        a_total_vec,
        x_L2,
        nsteps=nsteps,
        t_max=t_max,
    )

    del a_norm, a_total_vec
    plot_traj(X, Y, x_list, y_list, z_list, lagrange_points)


def plot_traj(X, Y, x_list, y_list, z_list, lagrange_points):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    ## Projection des points de Lagrange
    proj_x, proj_y, labels = project_lagrange_points(lagrange_points, plane="xy")
    ax.scatter(proj_x, proj_y, np.zeros_like(proj_x), color="grey", s=40)

    ## Projection de la Terre (utilise lagrange_points[0] comme approximation)
    ax.scatter([x_earth], [0.0],[0.0], color=["blue"], s=80)

    ax.plot(x_list, y_list, z_list, color="red", label="Particule")
    ax.scatter(
        [x_list[0]],
        [y_list[0]],
        [z_list[0]],
        color="red",
        s=60,
        marker="x",
        label="Départ",
    )

    margin = 1e9
    ax.set_xlim(lagrange_points[0] - margin / 4, lagrange_points[1] + margin)
    ax.set_ylim(-margin, margin)
    ax.set_zlim(-margin, margin)

    plt.title("Trajectoire simulée (approx. JWST L2 halo & SK)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    plt.grid(False)
    plt.tight_layout()
    # note: set_aspect('equal') n'est pas supporté proprement sur Axes3D
    plt.legend(loc="upper right")
    plt.show()


### Plot ###
def main(type=None):
    lagrange_points = x_L1, x_L2, x_L3, (x_L4, y_L4), (x_L5, y_L5) = (
        compute_theorical_lagrangian_points()
    )

    X, Y, Z, a_norm = compute_acceleration_norm(100, 2e11)
    a_total_vec = compute_acceleration_vect(X, Y, Z)

    print("fin de calcul du champ d'accélérations")
    simulate_particle(X, Y, Z, a_norm, a_total_vec, lagrange_points)

    # graphical_representation_xy(X, Y, a_norm, a_total_vec, lagrange_points)
    # graphical_representation_xz(X, Z, a_norm, a_total_vec, lagrange_points)
    # graphical_representation_yz(Y, Z, a_norm, a_total_vec, lagrange_points)

    """match type:
        case "normal_xy":
            graphical_representation_xy(X, Y, a_norm, a_total_vec, lagrange_points)

        case "normal_xz":
            graphical_representation_xz(X, Z, a_norm, a_total_vec, lagrange_points)

        case "normal_yz":
            graphical_representation_yz(Y, Z, a_norm, a_total_vec, lagrange_points)

        case "moon_phases":
            plot_moon_phase_effects(500, 2e9)
    """


def project_lagrange_points(lagrange_points, plane):
    """
    Projette les points de Lagrange sur un plan choisi : 'xy', 'yz' ou 'xz'.

    Paramètres
    ----------
    lagrange_points : tuple
        (x_L1, x_L2, x_L3, (x_L4, y_L4), (x_L5, y_L5))
    plane : str
        'xy', 'yz' ou 'xz'

    Retour
    ------
    coords : (list, list)
        Deux listes (X_proj, Y_proj) pour scatter/plot
    labels : list
        Noms correspondants ['L1', ..., 'L5']
    """
    x_L1, x_L2, x_L3, (x_L4, y_L4), (x_L5, y_L5) = lagrange_points

    # On suppose que les points sont tous dans le plan z=0
    points = {
        "L1": (x_L1, 0, 0),
        "L2": (x_L2, 0, 0),
        "L3": (x_L3, 0, 0),
        "L4": (x_L4, y_L4, 0),
        "L5": (x_L5, y_L5, 0),
    }

    proj_x, proj_y, labels = [], [], []

    for label, (x, y, z) in points.items():
        if plane == "xy":
            proj_x.append(x)
            proj_y.append(y)
        elif plane == "yz":
            proj_x.append(y)
            proj_y.append(z)
        elif plane == "xz":
            proj_x.append(x)
            proj_y.append(z)
        else:
            raise ValueError("Plane must be 'xy', 'yz' or 'xz'")
        labels.append(label)

    return proj_x, proj_y, labels


def graphical_representation_xy(X, Y, a_norm: np.ndarray, a_total_vec, lagrange_points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    idz = a_norm.shape[2] // 2
    a_norm_xy = a_norm[:, :, idz]

    plt.contourf(
        X[:, :, idz], Y[:, :, idz], log10(a_norm_xy), levels=1000, cmap="coolwarm"
    )
    plt.colorbar(label="log10(|acceleration|) [m/s²]")

    ## Projection des points de Lagrange
    proj_x, proj_y, labels = project_lagrange_points(lagrange_points, plane="xy")
    plt.scatter(proj_x, proj_y, color="grey", s=40)
    for lx, ly, name in zip(proj_x, proj_y, labels):
        plt.text(lx, ly, name, color="grey", ha="center", va="bottom")

    ## Projection de la Terre
    plt.scatter([x_earth], [0], color=["blue"], s=80)
    plt.text(x_earth, 0, "Earth", color="blue", ha="right")

    plt.title("Champs d'accélération (Soleil-Terre) - Plan XY")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(False)
    plt.tight_layout()
    ax.set_aspect("equal", adjustable="box")

    plt.show()


def graphical_representation_xz(X, Z, a_norm: np.ndarray, a_total_vec, lagrange_points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    idy = a_norm.shape[1] // 2
    a_norm_xz = a_norm[:, idy, :]

    plt.contourf(
        X[:, idy, :], Z[:, idy, :], log10(a_norm_xz), levels=1000, cmap="coolwarm"
    )

    # plt.colorbar(label="log10(|a|) [m/s²]")

    """# Projection des points de Lagrange
    proj_x, proj_z, labels = project_lagrange_points(lagrange_points, plane="xz")
    plt.scatter(proj_x, proj_z, color="grey", s=40)
    for lx, lz, name in zip(proj_x, proj_z, labels):
        plt.text(lx, lz, name, color="grey", ha="center", va="bottom")"""

    plt.title("Champs d'accélération (Soleil-Terre) - Plan XZ")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.grid(False)
    plt.tight_layout()
    ax.set_aspect("equal", adjustable="box")

    plt.show()


def graphical_representation_yz(Y, Z, a_norm, a_total_vec, lagrange_points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    idx = a_norm.shape[0] // 2
    a_norm_yz = a_norm[idx, :, :]

    plt.contourf(
        Y[idx, :, :], Z[idx, :, :], log10(a_norm_yz), levels=1000, cmap="coolwarm"
    )

    plt.colorbar(label="log10(|a|) [m/s²]")

    # Projection des points de Lagrange
    proj_y, proj_z, labels = project_lagrange_points(lagrange_points, plane="yz")
    plt.scatter(proj_y, proj_z, color="grey", s=40)
    for lx, lz, name in zip(proj_y, proj_z, labels):
        plt.text(lx, lz, name, color="grey", ha="center", va="bottom")

    plt.title("Champs d'accélération (Soleil-Terre) - Plan YZ")
    plt.xlabel("y [m]")
    plt.ylabel("z [m]")
    plt.grid(False)
    plt.tight_layout()
    ax.set_aspect("equal", adjustable="box")

    plt.show()


if __name__ == "__main__":
    main()

    """ Pour afficher les vecteurs
    step = 1
    plt.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        a_total_vec[0, ::step, ::step],
        a_total_vec[1, ::step, ::step],
        color="white",
        scale=2e-3,
        width=0.002,
    )
    """
