from encodings.punycode import T
import numpy as np
from numba import njit


@njit(fastmath=True)
def trilinear_interpolation(
    x, y, z, x_vals, y_vals, z_vals, field, dt, n, pos_L2, t_max
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
    pos_L2,
    nsteps=5000,
    t_max=100.0,
):
    """
    Intègre une particule dans le champ d'accélération avec RK4.
    Station-keeping modélisé par impulsions (delta-v) tous les 21 jours,
    appliquées comme de petites impulsions instantanées sur la vitesse.
    """

    dt = t_max / nsteps

    # --- parametres de Station-keeping ---
    period_time_sk = 21.0 * 86400.0  # 21 days in seconds (SK cadence)
    period_steps = max(1, int(period_time_sk / dt))

    # Realistic JWST SK magnitude: ~0.1 - 0.5 m/s per SK ; default 0.3 m/s
    dv_sk = 0.3  # m/s, adjustable

    # Heuristic weights to combine "position" and "vx-reduction" directions
    # weight_position: importance of pushing along the position component (stable eigenvector approx)
    # weight_vx: importance of reducing the x-velocity (aiming for zero x-velocity at crossing)
    weight_position = 0.5
    weight_vx = 0.5

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

    # For crossing detection (XZ plane is y==0)
    y_prev = y
    crossing_count = 0

    for n in range(nsteps):
        # RK4 : on calcule accélération plusieurs fois
        ax1, ay1, az1 = trilinear_interpolation(
            x, y, z, x_vals, y_vals, z_vals, a_total_vec, dt, n, pos_L2, t_max
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
            dt,
            n,
            pos_L2,
            t_max,
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
            dt,
            n,
            pos_L2,
            t_max,
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
            dt,
            n,
            pos_L2,
            t_max,
        )
        kx4, ky4, kz4 = vx + dt * ax3, vy + dt * ay3, vz + dt * az3

        # Mise à jour (RK4)
        x += sixieme_temps * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4)
        y += sixieme_temps * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4)
        z += sixieme_temps * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4)

        vx += sixieme_temps * (ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4)
        vy += sixieme_temps * (ay1 + 2.0 * ay2 + 2.0 * ay3 + ay4)
        vz += sixieme_temps * (az1 + 2.0 * az2 + 2.0 * az3 + az4)

        # --- Crossing detection (XZ plane: y == 0) ---
        # Detect sign change from previous step -> crossing
        if (y_prev <= 0.0 and y > 0.0) or (y_prev >= 0.0 and y < 0.0):
            crossing_count += 1
        y_prev = y

        # --- Station-keeping impulse every period_steps (≈21 days) ---
        if (n + 1) % period_steps == 0:
            # Position-based direction (approximation of position component of stable eigenvector)
            dx_pos = pos_L2 - x
            dy_pos = -y
            dz_pos = -z
            norm_pos = np.sqrt(dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos)
            if norm_pos < 1e-12:
                norm_pos = 1.0
            ux_pos = dx_pos / norm_pos
            uy_pos = dy_pos / norm_pos
            uz_pos = dz_pos / norm_pos

            # Velocity reduction direction: attempt to reduce x-velocity (toward zero x-velocity)
            # take projection of current velocity and create a direction that reduces vx preferentially
            # we use -sign(vx) along x and small components along y,z to avoid purely axis-aligned burns
            vx_sign = 1.0
            if vx > 0.0:
                vx_sign = 1.0
            else:
                vx_sign = -1.0
            # direction to reduce vx is roughly (-sign(vx), 0, 0) in inertial frame
            ux_v = -vx_sign
            uy_v = 0.0
            uz_v = 0.0
            # normalize combined direction (weighted)
            comb_x = weight_position * ux_pos + weight_vx * ux_v
            comb_y = weight_position * uy_pos + weight_vx * uy_v
            comb_z = weight_position * uz_pos + weight_vx * uz_v
            comb_norm = np.sqrt(comb_x * comb_x + comb_y * comb_y + comb_z * comb_z)
            if comb_norm < 1e-12:
                comb_norm = 1.0
            ux = comb_x / comb_norm
            uy = comb_y / comb_norm
            uz = comb_z / comb_norm

            # Apply delta-v impulse along composed direction
            vx += dv_sk * ux
            vy += dv_sk * uy
            vz += dv_sk * uz

        x_list[n + 1], y_list[n + 1], z_list[n + 1] = x, y, z

    return x_list, y_list, z_list
