# --- fonctions CR3BP / STM (hors Numba) ---
import numpy as np
import math
from numba import njit
from constants import G, M_sun, M_earth, d_se

# --- fonctions CR3BP / STM (hors Numba) ---
import numpy as np
import math

# constantes physiques
G = 6.67430e-11
M_sun = 1.98847e30
M_earth = 5.97219e24
# distance Sun - Earth-Moon barycenter (meters) - du doc
D_SE = 149.5e9  # 149.5e6 km -> 149.5e9 m


def nondim_params():
    """
    retourne mu (mass parameter), et temps/longueur de normalisation
    pour le système Sun-Earth.
    """
    mu = M_earth / (M_sun + M_earth)  # mass parameter
    L = D_SE  # length unit
    T = math.sqrt(L**3 / (G * (M_sun + M_earth)))  # time unit
    return mu, L, T


def cr3bp_state_deriv_rot(state, mu):
    """
    état dans le cadre rotatif (nondim) : state = [x,y,z, vx, vy, vz]
    dérivées temporelles en nondim time.
    """
    x, y, z, vx, vy, vz = state
    # distances to primaries in nondim units:
    d1 = math.sqrt((x + mu) ** 2 + y**2 + z**2)
    d2 = math.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)

    # potential partials (U_x, U_y, U_z) in nondim CR3BP rotating frame
    Ux = x - (1.0 - mu) * (x + mu) / (d1**3) - mu * (x - 1.0 + mu) / (d2**3)
    Uy = y - (1.0 - mu) * y / (d1**3) - mu * y / (d2**3)
    Uz = -(1.0 - mu) * z / (d1**3) - mu * z / (d2**3)

    ax = 2.0 * vy + Ux
    ay = -2.0 * vx + Uy
    az = Uz

    return np.array([vx, vy, vz, ax, ay, az], dtype=float)


def jacobian_A_cr3bp(state, mu):
    """
    Compute the 6x6 Jacobian matrix A(t) for CR3BP at `state` in rotating frame (nondim).
    A is structured as in the paper (top-right identity, bottom-left Hessian of potential + coriolis entries).
    """
    x, y, z, vx, vy, vz = state
    d1 = math.sqrt((x + mu) ** 2 + y**2 + z**2)
    d2 = math.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)

    # second derivatives of potential U*
    # compute common terms
    # For brevity: compute Uxx, Uyy, Uzz, Uxy, Uxz, Uyz
    # careful with signs
    # contributions from primary1 (1-mu) at (-mu,0,0)
    rx1 = x + mu
    ry1 = y
    rz1 = z
    r1_5 = d1**5
    r1_3 = d1**3

    # contributions from primary2 (mu) at (1-mu,0,0)
    rx2 = x - 1.0 + mu
    ry2 = y
    rz2 = z
    r2_5 = d2**5
    r2_3 = d2**3

    Uxx = (
        1.0
        - (1.0 - mu) * ((r1_3 - 3.0 * rx1 * rx1 / d1**2) / r1_3)
        - mu * ((r2_3 - 3.0 * rx2 * rx2 / d2**2) / r2_3)
    )
    # but above expression simplifies poorly; compute explicitly using standard formula

    # safer explicit expressions (from standard CR3BP Hessian)
    Uxx = (
        1.0
        - (1.0 - mu) * ((1.0 / r1_3) - 3.0 * rx1 * rx1 / r1_5)
        - mu * ((1.0 / r2_3) - 3.0 * rx2 * rx2 / r2_5)
    )

    Uyy = (
        1.0
        - (1.0 - mu) * ((1.0 / r1_3) - 3.0 * ry1 * ry1 / r1_5)
        - mu * ((1.0 / r2_3) - 3.0 * ry2 * ry2 / r2_5)
    )

    Uzz = -(1.0 - mu) * ((1.0 / r1_3) - 3.0 * rz1 * rz1 / r1_5) - mu * (
        (1.0 / r2_3) - 3.0 * rz2 * rz2 / r2_5
    )

    Uxy = -(1.0 - mu) * (-3.0 * rx1 * ry1 / r1_5) - mu * (-3.0 * rx2 * ry2 / r2_5)
    Uxz = -(1.0 - mu) * (-3.0 * rx1 * rz1 / r1_5) - mu * (-3.0 * rx2 * rz2 / r2_5)
    Uyz = -(1.0 - mu) * (-3.0 * ry1 * rz1 / r1_5) - mu * (-3.0 * ry2 * rz2 / r2_5)

    A = np.zeros((6, 6))
    # top-right identity
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    # bottom-left: partials of accelerations (Uxx etc) and coriolis terms
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


def propagate_monodromy(state0, mu, period_nondim, nsteps):
    """
    Propagate the combined state + STM for one period in nondim CR3BP rotating frame.
    state0: 6-vector initial state in rotating frame (nondim)
    returns STM(t0+period) as 6x6 matrix (monodromy)
    """
    dt = period_nondim / nsteps
    # Initial STM = identity (6x6)
    Phi = np.eye(6)
    # augmented state: [state(6), Phi.flatten(36)]
    aug = np.zeros(6 + 36)
    aug[:6] = state0.copy()
    aug[6:] = Phi.flatten()

    for _ in range(nsteps):
        # simple RK4 on augmented system
        k1 = aug_deriv(aug, mu)
        k2 = aug_deriv(aug + 0.5 * dt * k1, mu)
        k3 = aug_deriv(aug + 0.5 * dt * k2, mu)
        k4 = aug_deriv(aug + dt * k3, mu)
        aug = aug + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    Phi_final = aug[6:].reshape((6, 6))
    return Phi_final


def aug_deriv(aug, mu):
    """
    derivative of the augmented state: aug = [state(6), Phi_flat(36)]
    returns derivative of same shape.
    """
    state = aug[:6]
    Phi_flat = aug[6:]
    Phi = Phi_flat.reshape((6, 6))
    # state derivative
    state_dot = cr3bp_state_deriv_rot(state, mu)
    # STM derivative: A(t) @ Phi
    A = jacobian_A_cr3bp(state, mu)
    Phi_dot = A.dot(Phi)
    out = np.zeros_like(aug)
    out[:6] = state_dot
    out[6:] = Phi_dot.flatten()
    return out


def stable_eigenvector_from_monodromy(Phi):
    """
    Given monodromy Phi (6x6), compute eigenvectors and return the stable eigenvector (real part)
    associated to eigenvalue with |lambda| < 1. Return normalized position components (3-vector).
    """
    vals, vecs = np.linalg.eig(Phi)
    # find eigen with modulus < 1 (stable)
    idx = -1
    min_mod = 1e9
    for i in range(len(vals)):
        mod = abs(vals[i])
        # choose eigenvalue with mod < 1 and minimal mod
        if mod < min_mod:
            min_mod = mod
            idx = i
    v = vecs[:, idx]
    # eigenvector may be complex -> take real part
    v_real = np.real(v)
    pos_comp = v_real[:3]
    nrm = np.linalg.norm(pos_comp)
    if nrm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return pos_comp / nrm


# --- End of CR3BP / monodromy helper functions ---


# --- Modifications dans simulate_particle et integrate_particle ---


# integrate_particle signature changed to accept stable_dir (3-vector)


@njit(fastmath=True)
def trilinear_interpolation(
    x,
    y,
    z,
    x_vals,
    y_vals,
    z_vals,
    field,
):
    """
    Interpolation trilineaire du champ "field" (shape: Nx,Ny,Nz,3).
    Retourne (ax, ay, az).
    """
    Nx, Ny, Nz, _ = field.shape
    dx_grid = (x_vals[-1] - x_vals[0]) / (Nx - 1)
    dy_grid = (y_vals[-1] - y_vals[0]) / (Ny - 1)
    dz_grid = (z_vals[-1] - z_vals[0]) / (Nz - 1)

    # Indices des coins
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

    # Poids relatifs (tx, ty, tz dans [0,1])
    tx = (x - x_vals[i]) / dx_grid
    ty = (y - y_vals[j]) / dy_grid
    tz = (z - z_vals[k]) / dz_grid

    ax = ay = az = 0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                w = (
                    ((1.0 - tx) if di == 0 else tx)
                    * ((1.0 - ty) if dj == 0 else ty)
                    * ((1.0 - tz) if dk == 0 else tz)
                )
                f = field[i + di, j + dj, k + dk]
                ax += w * f[0]
                ay += w * f[1]
                az += w * f[2]

    return ax, ay, az


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
    nsteps=5000,
    t_max=100.0,
):
    """
    Intègre une particule dans le champ d'accélération avec RK4.
    Applique des impulsions SK alignées avec 'stable_dir' (déjà fourni, inertial-approx).
    """
    dt = t_max / nsteps

    x = x0
    y = y0
    z = z0
    vx = vx0
    vy = vy0
    vz = vz0

    x_list = np.empty(nsteps + 1)
    y_list = np.empty(nsteps + 1)
    z_list = np.empty(nsteps + 1)
    x_list[0], y_list[0], z_list[0] = x, y, z

    demi_temps = 0.5 * dt
    sixieme_temps = dt / 6.0

    for n in range(nsteps):
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

        x_list[n + 1], y_list[n + 1], z_list[n + 1] = x, y, z

    return x_list, y_list, z_list
