# full_jwst_l2_sim.py
# D�pendances: numpy, math, numba, matplotlib
import numpy as np
import math
from numba import njit
import matplotlib.pyplot as plt

from constants import x_earth

# ----------------- Physical constants & normalization -----------------
G = 6.67430e-11
M_sun = 1.98847e30
M_earth = 5.97219e24

x_L2 = x_earth + 1.5e9


def nondim_params():
    """
    Retourne mu, L_unit, T_unit pour nondim CR3BP (Sun-Earth).
    Hypoth�se: positions handed as Sun-Earth barycentric (origin at barycenter).
    """
    mu = M_earth / (M_sun + M_earth)
    L = x_L2
    T = math.sqrt(L**3 / (G * (M_sun + M_earth)))
    return mu, L, T


# ----------------- CR3BP dynamics & STM propagation (non-numba) -----------
def cr3bp_state_deriv_rot(state, mu):
    """D�riv�e d'�tat (rotating frame nondim). state = [x,y,z,vx,vy,vz]"""
    x, y, z, vx, vy, vz = state
    d1 = math.sqrt((x + mu) ** 2 + y**2 + z**2)
    d2 = math.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)

    # partials Ux Uy Uz
    Ux = x - (1.0 - mu) * (x + mu) / (d1**3) - mu * (x - 1.0 + mu) / (d2**3)
    Uy = y - (1.0 - mu) * y / (d1**3) - mu * y / (d2**3)
    Uz = -(1.0 - mu) * z / (d1**3) - mu * z / (d2**3)

    ax = 2.0 * vy + Ux
    ay = -2.0 * vx + Uy
    az = Uz
    return np.array([vx, vy, vz, ax, ay, az], dtype=float)


def jacobian_A_cr3bp(state, mu):
    """Jacobian 6x6 for CR3BP rotating frame (nondim)."""
    x, y, z, vx, vy, vz = state
    d1 = math.sqrt((x + mu) ** 2 + y**2 + z**2)
    d2 = math.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)

    rx1 = x + mu
    ry1 = y
    rz1 = z
    rx2 = x - 1.0 + mu
    ry2 = y
    rz2 = z

    r1_3 = d1**3
    r1_5 = d1**5
    r2_3 = d2**3
    r2_5 = d2**5

    Uxx = (
        1.0
        - (1.0 - mu) * (1.0 / r1_3 - 3.0 * rx1 * rx1 / r1_5)
        - mu * (1.0 / r2_3 - 3.0 * rx2 * rx2 / r2_5)
    )
    Uyy = (
        1.0
        - (1.0 - mu) * (1.0 / r1_3 - 3.0 * ry1 * ry1 / r1_5)
        - mu * (1.0 / r2_3 - 3.0 * ry2 * ry2 / r2_5)
    )
    Uzz = -(1.0 - mu) * (1.0 / r1_3 - 3.0 * rz1 * rz1 / r1_5) - mu * (
        1.0 / r2_3 - 3.0 * rz2 * rz2 / r2_5
    )

    Uxy = -(1.0 - mu) * (-3.0 * rx1 * ry1 / r1_5) - mu * (-3.0 * rx2 * ry2 / r2_5)
    Uxz = -(1.0 - mu) * (-3.0 * rx1 * rz1 / r1_5) - mu * (-3.0 * rx2 * rz2 / r2_5)
    Uyz = -(1.0 - mu) * (-3.0 * ry1 * rz1 / r1_5) - mu * (-3.0 * ry2 * rz2 / r2_5)

    A = np.zeros((6, 6))
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0

    A[3, 0] = Uxx
    A[3, 1] = Uxy
    A[3, 2] = Uxz
    A[3, 4] = 2.0
    A[4, 0] = Uxy
    A[4, 1] = Uyy
    A[4, 2] = Uyz
    A[4, 3] = -2.0
    A[5, 0] = Uxz
    A[5, 1] = Uyz
    A[5, 2] = Uzz
    return A


def aug_deriv(aug, mu):
    """
    Derivative of augmented vector [state(6), Phi_flat(36)].
    """
    state = aug[:6]
    Phi_flat = aug[6:]
    Phi = Phi_flat.reshape((6, 6))
    state_dot = cr3bp_state_deriv_rot(state, mu)
    A = jacobian_A_cr3bp(state, mu)
    Phi_dot = A.dot(Phi)
    out = np.zeros_like(aug)
    out[:6] = state_dot
    out[6:] = Phi_dot.flatten()
    return out


def propagate_monodromy(state0, mu, period_nondim, nsteps=4000):
    """
    Propagate augmented system for one period nondim using RK4 -> returns Phi_final.
    state0 in nondim rotating frame.
    """
    dt = period_nondim / nsteps
    Phi = np.eye(6)
    aug = np.zeros(6 + 36)
    aug[:6] = state0.copy()
    aug[6:] = Phi.flatten()

    for _ in range(nsteps):
        k1 = aug_deriv(aug, mu)
        k2 = aug_deriv(aug + 0.5 * dt * k1, mu)
        k3 = aug_deriv(aug + 0.5 * dt * k2, mu)
        k4 = aug_deriv(aug + dt * k3, mu)
        aug = aug + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    Phi_final = aug[6:].reshape((6, 6))
    return Phi_final


def stable_eigenvector_from_monodromy(Phi):
    """
    Retourne composante position (3) de l'eigenvector stable (valeur propre |lambda|<1).
    """
    vals, vecs = np.linalg.eig(Phi)
    # choose eigen with modulus < 1 and smallest modulus (stable)
    idx = None
    min_mod = 1e9
    for i in range(len(vals)):
        mod = abs(vals[i])
        if mod < min_mod:
            min_mod = mod
            idx = i
    v = vecs[:, idx]
    vr = np.real(v)
    pos = vr[:3]
    norm = np.linalg.norm(pos)
    if norm == 0:
        return np.array([1e-12, 0.0, 0.0])
    return pos / norm


# ----------------- trilinear interpolation (numba) -----------------------
@njit(fastmath=True)
def trilinear_interpolation(x, y, z, x_vals, y_vals, z_vals, field):
    """
    Interpolation trilineaire du champ "field" (shape: Nx,Ny,Nz,3).
    Retourne (ax, ay, az).
    """
    Nx, Ny, Nz, _ = field.shape
    dx_grid = (x_vals[-1] - x_vals[0]) / (Nx - 1)
    dy_grid = (y_vals[-1] - y_vals[0]) / (Ny - 1)
    dz_grid = (z_vals[-1] - z_vals[0]) / (Nz - 1)

    # Indices des coins \u2014 clamp
    i = int((x - x_vals[0]) / dx_grid)
    j = int((y - y_vals[0]) / dy_grid)
    k = int((z - z_vals[0]) / dz_grid)

    if i < 0:
        i = 0
    if i > Nx - 2:
        i = Nx - 2
    if j < 0:
        j = 0
    if j > Ny - 2:
        j = Ny - 2
    if k < 0:
        k = 0
    if k > Nz - 2:
        k = Nz - 2

    tx = (x - x_vals[i]) / dx_grid
    ty = (y - y_vals[j]) / dy_grid
    tz = (z - z_vals[k]) / dz_grid

    ax = 0.0
    ay = 0.0
    az = 0.0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                w = (
                    ((1.0 - tx) if di == 0 else tx)
                    * ((1.0 - ty) if dj == 0 else ty)
                    * ((1.0 - tz) if dk == 0 else tz)
                )
                f0 = field[i + di, j + dj, k + dk, 0]
                f1 = field[i + di, j + dj, k + dk, 1]
                f2 = field[i + di, j + dj, k + dk, 2]
                ax += w * f0
                ay += w * f1
                az += w * f2
    return ax, ay, az


# ----------------- Integrator with SK impulses (numba) --------------------
@njit(fastmath=True)
def integrate_particle(
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
    pos_L2,
    stable_dir_x,
    stable_dir_y,
    stable_dir_z,
    nsteps=200000,
    t_max=3.0 * 168.0 * 86400.0,
    save_every=10,
    dv_sk=0.3,
):
    """
    RK4 integrator. Applies station-keeping impulses (dv_sk) every 21 days along stable_dir.
    save_every reduces memory (store 1 point every save_every steps).
    """
    dt = t_max / nsteps
    period_time_sk = 21.0 * 86400.0
    period_steps = max(1, int(period_time_sk / dt))

    x = x0
    y = y0
    z = z0
    vx = vx0
    vy = vy0
    vz = vz0

    # number of stored points
    stored = nsteps // save_every + 1
    xs = np.empty(stored)
    ys = np.empty(stored)
    zs = np.empty(stored)
    idx_store = 0
    xs[idx_store] = x
    ys[idx_store] = y
    zs[idx_store] = z
    idx_store += 1

    demi_temps = 0.5 * dt
    sixieme_temps = dt / 6.0

    for n in range(nsteps):
        # RK4
        ax1, ay1, az1 = trilinear_interpolation(
            x, y, z, x_vals, y_vals, z_vals, a_total_vec
        )
        kx1, ky1, kz1 = vx, vy, vz

        ax2, ay2, az2 = trilinear_interpolation(
            x + demi_temps * kx1,
            y + demi_temps * ky1,
            z + demi_temps * kz1,
            x_vals,
            y_vals,
            z_vals,
            a_total_vec,
        )
        kx2, ky2, kz2 = (
            vx + demi_temps * ax1,
            vy + demi_temps * ay1,
            vz + demi_temps * az1,
        )

        ax3, ay3, az3 = trilinear_interpolation(
            x + demi_temps * kx2,
            y + demi_temps * ky2,
            z + demi_temps * kz2,
            x_vals,
            y_vals,
            z_vals,
            a_total_vec,
        )
        kx3, ky3, kz3 = (
            vx + demi_temps * ax2,
            vy + demi_temps * ay2,
            vz + demi_temps * az2,
        )

        ax4, ay4, az4 = trilinear_interpolation(
            x + dt * kx3,
            y + dt * ky3,
            z + dt * kz3,
            x_vals,
            y_vals,
            z_vals,
            a_total_vec,
        )
        kx4, ky4, kz4 = vx + dt * ax3, vy + dt * ay3, vz + dt * az3

        x += sixieme_temps * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4)
        y += sixieme_temps * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4)
        z += sixieme_temps * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4)

        vx += sixieme_temps * (ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4)
        vy += sixieme_temps * (ay1 + 2.0 * ay2 + 2.0 * ay3 + ay4)
        vz += sixieme_temps * (az1 + 2.0 * az2 + 2.0 * az3 + az4)

        # station-keeping impulse
        if (n + 1) % period_steps == 0:
            vx += dv_sk * stable_dir_x
            vy += dv_sk * stable_dir_y
            vz += dv_sk * stable_dir_z

        # store every save_every steps
        if (n + 1) % save_every == 0:
            xs[idx_store] = x
            ys[idx_store] = y
            zs[idx_store] = z
            idx_store += 1

    return xs[:idx_store], ys[:idx_store], zs[:idx_store]


# ----------------- Orchestrator: build stable_dir and run -----------------
def simulate_particle_full(X, Y, Z, a_total_vec, lagrange_points):
    """
    X,Y,Z not used here except to build grid vectors consistent with a_total_vec shape.
    lagrange_points expected to be array-like with L1, L2 positions in same frame (dimensional).
    """
    # ---- realistic JWST initial state (Sun-Earth barycentric approx, dimensional) ----
    # These values are approximate and chosen to represent a JWST-like halo initial state.
    x0 = (
        lagrange_points[1] + 4.0e8
    )  # 400 000 km outward from L2 (m)  --> ajuste selon halo souhaité
    y0 = 1.0e5  # 100 km
    z0 = -1.0e9
    vx0 = 0.0  # m/s
    vy0 = -17.2  # m/s
    vz0 = -2.0  # m/s

    # ---- nondim params and conversion ----
    mu, L_unit, T_unit = nondim_params()
    # convert state to nondim rotating frame: assume barycenter origin => divide by L_unit
    state_n = np.zeros(6, dtype=float)
    state_n[0] = x0 / L_unit
    state_n[1] = y0 / L_unit
    state_n[2] = z0 / L_unit
    state_n[3] = (vx0) / (L_unit / T_unit)
    state_n[4] = (vy0) / (L_unit / T_unit)
    state_n[5] = (vz0) / (L_unit / T_unit)

    # nondim orbital period approx (use 168 days)
    T_orbit_dim = 168.0 * 86400.0
    T_orbit_n = T_orbit_dim / T_unit

    # propagate monodromy (nsteps moderately large for decent accuracy)
    print("Propagating monodromy (this can take some time)...")
    Phi = propagate_monodromy(state_n, mu, T_orbit_n, nsteps=3000)
    stable_pos_n = stable_eigenvector_from_monodromy(
        Phi
    )  # pos components nondim (unit)

    # convert stable_pos_n to dimensional (meters) in rotating frame
    stable_pos_dim = stable_pos_n * L_unit

    # approximate rotation RLP->inertial: we use simple rotation about z by theta=omega*t0 (t0=0 here)
    # omega (rad/s) ~ 2*pi / T_orbit_dim (mean)
    omega = 2.0 * math.pi / T_orbit_dim
    theta = 0.0  # if you have epoch time, use theta = omega * epoch_seconds
    c = math.cos(theta)
    s = math.sin(theta)
    # rotating -> inertial approx
    stable_inertial = np.zeros(3, dtype=float)
    stable_inertial[0] = c * stable_pos_dim[0] - s * stable_pos_dim[1]
    stable_inertial[1] = s * stable_pos_dim[0] + c * stable_pos_dim[1]
    stable_inertial[2] = stable_pos_dim[2]

    # normalize unit direction to apply delta-v
    nrm = np.linalg.norm(stable_inertial)
    if nrm == 0:
        stable_unit = np.array([1e-12, 0.0, 0.0])
    else:
        stable_unit = stable_inertial / nrm

    print("Stable direction (inertial, unit):", stable_unit)

    # build x_vals,y_vals,z_vals compatible with a_total_vec grid shape
    Nx, Ny, Nz, _ = a_total_vec.shape
    x_vals = np.linspace(X.min(), X.max(), Nx)
    y_vals = np.linspace(Y.min(), Y.max(), Ny)
    z_vals = np.linspace(Z.min(), Z.max(), Nz)

    # run integrator: pass stable_dir components
    xs, ys, zs = integrate_particle(
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
        save_every=50,
        dv_sk=0.3,
    )
    # plot
    plot_traj(xs, ys, zs, lagrange_points)


# ----------------- simple plotting -----------------
def plot_traj(x_list, y_list, z_list, lagrange_points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    # Lagrange points projection
    ax.scatter([x_earth], [0.0], [0.0], color="blue", s=80, label="Earth (approx)")  # type: ignore
    ax.scatter([lagrange_points[1]], [0.0], [0.0], color="grey", s=60, label="L2 (approx)")  # type: ignore
    ax.plot(x_list, y_list, z_list, color="red", label="Trajectory")
    ax.scatter(
        [x_list[0]],
        [y_list[0]],
        [z_list[0]],
        color="red",
        marker="x",
        s=80,  # type: ignore
        label="Start",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")  # type: ignore

    margin = 1e9
    ax.set_xlim(x_earth - margin, lagrange_points[1] + margin)
    # ax.set_ylim(-margin, margin)
    # ax.set_zlim(-margin, margin)  # type: ignore

    plt.title("Simulated JWST-like halo & SK (approx)")
    plt.legend()
    plt.show()


# ----------------- Example usage (user must provide a_total_vec grid and X,Y,Z mesh) -----------
if __name__ == "__main__":
    # == Example placeholder: build a_total_vec with pure Sun+Earth point-mass gravity for demonstration ==
    # build grid extents consistent with typical problem scale:
    Nx, Ny, Nz = 20, 20, 10
    x_vals = np.linspace(1.48e11, 1.52e11, Nx)  # around 1.5e11 m
    y_vals = np.linspace(-1.0e9, 1.0e9, Ny)
    z_vals = np.linspace(-1.0e9, 1.0e9, Nz)
    Xg = np.zeros((Nx, Ny, Nz))
    Yg = np.zeros_like(Xg)
    Zg = np.zeros_like(Xg)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                Xg[i, j, k] = x_vals[i]
                Yg[i, j, k] = y_vals[j]
                Zg[i, j, k] = z_vals[k]

    # naive gravitational acceleration field due to Sun and Earth-Moon barycenter (for illustration)
    a_total_vec = np.zeros((Nx, Ny, Nz, 3))
    # Earth location (approx barycenter at origin, so Earth at ~ +mu*L etc) but we'll place Earth at 1.0 location for plotting simplicity
    # For posterity, use simple two-body accelerations centered at Sun and Earth positions
    sun_pos = np.array(
        [0.0, 0.0, 0.0]
    )  # we assume barycenter at origin for this toy grid => actually not perfect mapping
    earth_pos = np.array([x_earth, 0.0, 0.0])  # approx

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r = np.array([Xg[i, j, k], Yg[i, j, k], Zg[i, j, k]])
                # acceleration due to Sun
                r_sun = r - sun_pos
                rs = np.linalg.norm(r_sun)
                a_sun = -G * M_sun * r_sun / (rs**3)
                # acceleration due to Earth
                r_earth = r - earth_pos
                re = np.linalg.norm(r_earth)
                a_earth = -G * M_earth * r_earth / (re**3)
                a_total = a_sun + a_earth
                a_total_vec[i, j, k, 0] = a_total[0]
                a_total_vec[i, j, k, 1] = a_total[1]
                a_total_vec[i, j, k, 2] = a_total[2]

    # Lagrange points placeholders (dimensional)
    lagrange_points = [
        x_earth - 1.5e9,
        x_L2,
    ]  # simple: L1 placeholder at 0.0, L2 at ~1.5e11 (for demo)

    # run simulation
    simulate_particle_full(Xg, Yg, Zg, a_total_vec, lagrange_points)
