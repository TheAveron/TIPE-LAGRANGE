import matplotlib.pyplot as plt
from numpy import int64, log10, min, max, ndarray, float64
from typing import Any
from constants import x_earth, x_sun, x_moon
from therocal_poisitions import compute_theorical_lagrangian_points
from acceleration import compute_acceleration_norm, compute_acceleration_vect
from integrator import (
    integrate_particle,
    nondim_params,
    propagate_monodromy,
    stable_eigenvector_from_monodromy,
)
import numpy as np


# --- simulate_particle: compute stable eigenvector (CR3BP) before integration ---
def simulate_particle(X, Y, Z, a_norm: np.ndarray, a_total_vec, lagrange_points):
    # setup initial conditions (dimensional)
    x_L2 = lagrange_points[1]
    x0 = x_L2 + 4.0e8
    y0 = 1.0e5
    z0 = -1.0e8

    # nondim params
    mu, L_unit, T_unit = nondim_params()

    # convert initial state to nondim rotating frame coordinates
    # NOTE: We approximate that L2 lies on x-axis at nondim x ~ 1 + small (we assume lagrange_points[1] corresponds to dimensional L2)
    # For CR3BP reference orbit, typical initial condition is near L2 in rotating frame:
    x0_n = (
        x0 - (1.0 - mu) * L_unit
    ) / L_unit  # crude transform: place barycenter at origin — user must ensure lagrange_points consistent
    y0_n = y0 / L_unit
    z0_n = z0 / L_unit

    # estimate initial velocities in rotating frame ~ 0 for starting guess (improvement: use known halo initial state)
    vx0_n = 0.0
    vy0_n = 0.0
    vz0_n = 0.0

    state0_n = np.array([x0_n, y0_n, z0_n, vx0_n, vy0_n, vz0_n])

    # choose period nondim: use CR3BP approximation of LPO period near L2
    # Dimensional T_orbit ~ 168 days -> nondim T = T_orbit / T_unit
    T_orbit_dim = 168.0 * 86400.0
    T_orbit_n = T_orbit_dim / T_unit

    # propagate monodromy (coarse nsteps sufficient for eigenvector)
    Phi = propagate_monodromy(state0_n, mu, T_orbit_n, nsteps=4000)

    stable_pos_n = stable_eigenvector_from_monodromy(
        Phi
    )  # in nondim rotating frame (position components)
    # convert stable_pos_n to dimensional vector in meters (still in rotating-frame basis)
    stable_pos_dim = stable_pos_n * L_unit  # vector in m in RLP axes

    # To use this direction in inertial-integrator, we need an approximate rotation from rotating frame to inertial.
    # Approximate rotation angle = theta = omega * t0 where omega = 2*pi / T_orbit_dim ; assume t0 ~ 0
    omega = 2.0 * np.pi / T_orbit_dim
    theta = 0.0  # assume initial epoch t=0; improvement: use actual epoch/time
    c = np.cos(theta)
    s = np.sin(theta)
    # rotation about z: R_z(theta)
    # rotating-frame vector -> inertial approx: [x_i] = R_z(theta) @ [x_r]
    stable_inertial = np.zeros(3)
    stable_inertial[0] = c * stable_pos_dim[0] - s * stable_pos_dim[1]
    stable_inertial[1] = s * stable_pos_dim[0] + c * stable_pos_dim[1]
    stable_inertial[2] = stable_pos_dim[2]

    # normalize to unit vector for dv application
    sn = np.linalg.norm(stable_inertial)
    if sn < 1e-12:
        stable_inertial = np.array([1.0, 0.0, 0.0])
        sn = 1.0
    stable_unit = stable_inertial / sn

    # build grids for interpolator (same as before)
    Nx, Ny, Nz, _ = a_total_vec.shape
    x_vals = np.linspace(X.min(), X.max(), Nx)
    y_vals = np.linspace(Y.min(), Y.max(), Ny)
    z_vals = np.linspace(Z.min(), Z.max(), Nz)

    # initial velocity estimate for the halo motion (same as earlier heuristic)
    T_orbit = T_orbit_dim
    r0 = 4.0e8
    v0_norm = 2.0 * np.pi * r0 / T_orbit
    # tangential approx in XY
    dx = x0 - x_L2
    dy = y0
    norm_xy = np.sqrt(dx * dx + dy * dy)
    if norm_xy < 1e-12:
        norm_xy = 1.0
    ux_t = -dy / norm_xy
    uy_t = dx / norm_xy
    uz_t = 0.0
    vx0 = v0_norm * ux_t
    vy0 = v0_norm * uy_t
    vz0 = v0_norm * uz_t

    # Call the njit integrator, passing stable_dir components
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
        stable_unit[0],
        stable_unit[1],
        stable_unit[2],
        nsteps=200000,
        t_max=3.0 * T_orbit_dim,
    )

    del a_norm, a_total_vec
    plot_traj(X, Y, x_list, y_list, z_list, lagrange_points)


def plot_traj(X, Y, x_list, y_list, z_list, lagrange_points):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    ## Projection des points de Lagrange
    proj_x, proj_y, labels = project_lagrange_points(lagrange_points, plane="xy")
    ax.scatter(proj_x, proj_y, np.zeros_like(proj_x), color="grey", s=40)  # type: ignore

    ## Projection de la Terre (utilise lagrange_points[0] comme approximation)
    ax.scatter([x_earth], [0.0], [0.0], color=["blue"], s=80)  # type: ignore

    ax.plot(x_list, y_list, z_list, color="red", label="Particule")
    ax.scatter(
        [x_list[0]],
        [y_list[0]],
        [z_list[0]],
        color="red",
        s=60,  # type: ignore
        marker="x",
        label="Départ",
    )

    margin = 1e9
    ax.set_xlim(lagrange_points[0] - margin / 4, lagrange_points[1] + margin)
    ax.set_ylim(-margin, margin)
    ax.set_zlim(-margin, margin)  # type: ignore

    plt.title("Trajectoire simulée (approx. JWST L2 halo & SK)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")  # type: ignore

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

    X, Y, Z, a_norm = compute_acceleration_norm(100, 2e11)  # type: ignore
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
