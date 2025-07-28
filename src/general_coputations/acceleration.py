import numpy as np
from constants import G, C, M_earth, M_sun, x_earth, x_sun, omega, M_moon, x_moon


def compute_grav_accel_vec(X, Y, x_body, y_body, M_body):
    """
    Vectorized computation of gravitational acceleration due to a body.
    Returns a 2*N*N array (ax, ay).
    """
    dx = X - x_body
    dy = Y - y_body
    r2 = dx**2 + dy**2
    r3 = r2 * np.sqrt(r2 + 1e-20)
    ax = -G * M_body * dx / r3
    ay = -G * M_body * dy / r3

    correction = 1 + (3 * G * M_body) / (np.sqrt(r2 + 1e-20) * C**2)

    return np.array([ax, ay])


def compute_total_accel(X, Y, include_norm=True):
    """
    Computes total acceleration vector field in the rotating frame.
    Returns either:
    - a 2*N*N array (vector field) if include_norm=False,
    - (ax, ay, a_norm) components if include_norm=True.
    """
    # Gravitational acceleration (Sun and Earth)
    a_sun = compute_grav_accel_vec(X, Y, x_sun, 0, M_body=M_sun)
    a_earth = compute_grav_accel_vec(X, Y, x_earth, 0, M_earth)
    a_moon = compute_grav_accel_vec(X, Y, x_moon, 0, M_moon)

    # Centrifugal acceleration
    a_centrifugal = omega**2 * np.stack([X, Y], axis=0)

    # Total acceleration
    a_total = a_sun + a_earth + a_centrifugal + a_moon
    a_norm = np.linalg.norm(a_total, axis=0)

    if include_norm:
        return a_total[0], a_total[1], a_norm
    else:
        return a_total / (a_norm * 10000)


def compute_acceleration_norm(grid_size=1000, xy_lim=2e6):
    """
    Creates grid and computes norm of the total acceleration.
    """
    x = np.linspace(-xy_lim, +xy_lim, 500)  # np.linspace(-xy_lim, xy_lim, grid_size)
    y = np.linspace(-xy_lim, xy_lim, 500)  # np.linspace(-xy_lim, xy_lim, grid_size)
    X, Y = np.meshgrid(x, y)

    ax, ay, a_norm = compute_total_accel(X, Y, include_norm=True)
    return X, Y, a_norm


def compute_acceleration_vect(X, Y):
    """
    Returns 2D acceleration vector field (ax, ay).
    """
    return compute_total_accel(X, Y, include_norm=False)
