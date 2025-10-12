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
    dx = (x_vals[-1] - x_vals[0]) / (Nx - 1)
    dy = (y_vals[-1] - y_vals[0]) / (Ny - 1)
    dz = (z_vals[-1] - z_vals[0]) / (Nz - 1)

    # Indices des coins
    i = int((x - x_vals[0]) / dx)
    j = int((y - y_vals[0]) / dy)
    k = int((z - z_vals[0]) / dz)

    if i < 0 or i >= Nx - 1 or j < 0 or j >= Ny - 1 or k < 0 or k >= Nz - 1:
        return 0, 0, 0  # hors grille → pas d'accélération

    # Poids relatifs
    tx = (x - x_vals[i]) / dx
    ty = (y - y_vals[j]) / dy
    tz = (z - z_vals[k]) / dz

    ax = ay = az = 0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                w = (
                    ((1 - tx) if di == 0 else tx)
                    * ((1 - ty) if dj == 0 else ty)
                    * ((1 - tz) if dk == 0 else tz)
                )
                f = field[i + di, j + dj, k + dk]
                ax += w * f[0]
                ay += w * f[1]
                az += w * f[2]

    if n > 0 and (t_max / (n * dt)) % 1_900_000 == 0:
        a_correction = 300
        alpha = np.arctan2(y, pos_L2 - x)
        cor_x = a_correction * np.cos(alpha)
        cor_y = a_correction * np.sin(alpha)
        return ax + cor_x, ay - cor_y, az

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
    Intègre une particule dans le champ d'accélération avec RK4 (rapide avec Numba).
    """
    dt = t_max / nsteps

    x, y, z = x0, y0, z0
    vx, vy, vz = vx0, vy0, vz0

    x_list = np.empty(nsteps + 1)
    y_list = np.empty(nsteps + 1)
    z_list = np.empty(nsteps + 1)
    x_list[0], y_list[0], z_list[0] = x, y, z

    demi_temps = 0.5 * dt
    sixieme_temps = dt / 6

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
        x += sixieme_temps * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
        y += sixieme_temps * (ky1 + 2 * ky2 + 2 * ky3 + ky4)
        z += sixieme_temps * (kz1 + 2 * kz2 + 2 * kz3 + kz4)

        vx += sixieme_temps * (ax1 + 2 * ax2 + 2 * ax3 + ax4)
        vy += sixieme_temps * (ay1 + 2 * ay2 + 2 * ay3 + ay4)
        vz += sixieme_temps * (az1 + 2 * az2 + 2 * az3 + az4)

        x_list[n + 1], y_list[n + 1], z_list[n + 1] = x, y, z

    return x_list, y_list, z_list
