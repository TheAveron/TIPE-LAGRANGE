import numpy as np
from numpy import ndarray, dtype
from typing import Any
from constants import G, C, M_earth, M_sun, x_earth, x_sun, omega, M_moon, x_moon


from numba import njit, prange


@njit(fastmath=True, parallel=True)
def compute_grav_accel_vec_numba(X, Y, Z, x_body, y_body, z_body, M_body):
    """
    Compute gravitational acceleration of a body at all grid points.
    Returns (ax, ay, az) arrays of same shape as X.
    """
    g_body = G * M_body

    g_body3 = 3 * g_body
    c_carré = C**2

    gc = g_body3 / c_carré

    Nx, Ny, Nz = X.shape
    ax = np.empty((Nx, Ny, Nz))
    ay = np.empty((Nx, Ny, Nz))
    az = np.empty((Nx, Ny, Nz))

    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                dx = X[i, j, k] - x_body
                dy = Y[i, j, k] - y_body
                dz = Z[i, j, k] - z_body
                r2 = dx * dx + dy * dy + dz * dz + 1e-20
                sq_r = np.sqrt(r2)
                inv_r3 = 1 / (r2 * sq_r)
                correction = 1 + gc / sq_r

                sous_produit = -inv_r3 * correction * g_body

                ax[i, j, k] = dx * sous_produit
                ay[i, j, k] = dy * sous_produit
                az[i, j, k] = dz * sous_produit

    return (ax, ay, az)


def compute_grav_accel_vec(X, Y, Z, x_body, y_body, M_body, z_body=0):
    """
    Vectorized computation of gravitational acceleration due to a body.
    Returns a 2*N*N array (ax, ay).
    """
    dx = X - x_body
    dy = Y - y_body
    dz = Z - z_body
    r2 = dx**2 + dy**2 + dz**2
    r3 = r2 * np.sqrt(r2 + 1e-20)
    ax = -G * M_body * dx / r3
    ay = -G * M_body * dy / r3
    az = -G * M_body * dz / r3

    correction = 1 + (3 * G * M_body) / (np.sqrt(r2 + 1e-20) * C**2)
    print(ax[0])

    return np.array([ax * correction, ay * correction, az * correction])


def compute_total_accel(X, Y, Z, include_norm=True):
    """
    Computes total acceleration vector field in the rotating frame.
    Returns either:
    - a 2*N*N array (vector field) if include_norm=False,
    - (ax, ay, a_norm) components if include_norm=True.
    """
    # Gravitational acceleration (Sun and Earth)
    a_sun = np.array(compute_grav_accel_vec_numba(X, Y, Z, x_sun, 0, 0, M_sun))
    a_earth = np.array(compute_grav_accel_vec_numba(X, Y, Z, x_earth, 0, 0, M_earth))
    a_moon = 0  # compute_grav_accel_vec(X, Y, x_moon, 0, M_moon)

    # Centrifugal acceleration
    a_centrifugal = omega**2 * np.stack([X, Y, Z], axis=0)

    # Total acceleration
    a_total = a_sun + a_earth + a_centrifugal + a_moon
    a_norm = np.linalg.norm(a_total, axis=0)

    if include_norm:
        return a_total[0], a_total[1], a_total[2], a_norm
    else:
        return a_total / (a_norm * 10000)


def compute_acceleration_norm(x,y,z) -> tuple[
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    Any,
]:
    """
    Creates grid and computes norm of the total acceleration.
    """
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    _, _, _, a_norm = compute_total_accel(X, Y, Z, include_norm=True)
    return X, Y, Z, a_norm


def compute_acceleration_vect(X, Y, Z):
    """
    Returns 2D acceleration vector field (ax, ay).
    """
    return compute_total_accel(X, Y, Z, include_norm=False)
