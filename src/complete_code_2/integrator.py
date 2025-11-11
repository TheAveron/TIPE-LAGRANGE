import numpy as np
from tqdm import tqdm

from dynamics import rotating_frame_acceleration


def trilinear_interpolation(x, y, z, x_vals, y_vals, z_vals, field):
    """
    Interpolation trilineaire du champ "field" (shape: Nx, Ny, Nz, 3).
    Retourne (ax, ay, az) correspondant à l'interpolation en (x, y, z).

    Arguments:
    - x, y, z : Coordonnées où l'interpolation doit être effectuée.
    - x_vals, y_vals, z_vals : Listes des valeurs de la grille sur chaque axe.
    - field : Champ de données à interpoler (Nx, Ny, Nz, 3).

    Retourne :
    - ax, ay, az : Valeurs interpolées dans le champ aux coordonnées (x, y, z).
    """
    Nx, Ny, Nz, _ = field.shape
    dx_grid = (x_vals[-1] - x_vals[0]) / (Nx - 1)
    dy_grid = (y_vals[-1] - y_vals[0]) / (Ny - 1)
    dz_grid = (z_vals[-1] - z_vals[0]) / (Nz - 1)

    # Indices des coins (contrôlés pour éviter d'être hors de la grille)
    i = max(0, min(int((x - x_vals[0]) / dx_grid), Nx - 2))
    j = max(0, min(int((y - y_vals[0]) / dy_grid), Ny - 2))
    k = max(0, min(int((z - z_vals[0]) / dz_grid), Nz - 2))

    # Poids relatifs (tx, ty, tz dans [0, 1])
    tx = (x - x_vals[i]) / dx_grid
    ty = (y - y_vals[j]) / dy_grid
    tz = (z - z_vals[k]) / dz_grid

    # Pré-calcul des poids pour éviter des calculs redondants
    tx1, tx2 = 1.0 - tx, tx
    ty1, ty2 = 1.0 - ty, ty
    tz1, tz2 = 1.0 - tz, tz

    # Initialisation des résultats
    ax = ay = az = 0.0

    # Calcul de l'interpolation trilineaire
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                # Poids pour chaque coin
                w = (
                    (tx1 if di == 0 else tx2)
                    * (ty1 if dj == 0 else ty2)
                    * (tz1 if dk == 0 else tz2)
                )
                f = field[i + di, j + dj, k + dk]
                ax += w * f[0]
                ay += w * f[1]
                az += w * f[2]

    return ax, ay, az


import numpy as np


def integrate_particle_rk4(r0, v0, dt, t_max=100.0):
    """
    Intègre une particule dans le champ gravitationnel de la Terre et du Soleil, avec les effets du référentiel tournant.
    Utilise la méthode Runge-Kutta d'ordre 4 (RK4) pour l'intégration.

    Arguments:
    - r0 : position initiale (vecteur 3D, en km)
    - v0 : vitesse initiale (vecteur 3D, en km/s)
    - omega_vec : vecteur de la vitesse angulaire du référentiel tournant
    - G : constante gravitationnelle
    - M_sun, M_earth : masses du Soleil et de la Terre
    - x_sun, x_earth : positions du Soleil et de la Terre dans le référentiel
    - nsteps : nombre d'étapes d'intégration
    - t_max : temps total de simulation

    Retourne:
    - r_list : liste des positions à chaque étape
    - v_list : liste des vitesses à chaque étape
    """

    # Initialisation des paramètres
    nsteps = round(t_max / dt) + 1
    print(nsteps)
    demi_temps = 0.5 * dt
    sixieme_temps = dt / 6.0

    # Positions et vitesses initiales
    r = r0
    v = v0

    # Listes pour stocker les positions et vitesses
    r_list = np.zeros((nsteps + 1, 3))
    v_list = np.zeros((nsteps + 1, 3))
    r_list[0], v_list[0] = r, v

    # Intégration de la trajectoire par Runge-Kutta (RK4)
    for n in tqdm(range(nsteps)):
        # Calcul des accélérations à différents points selon la méthode RK4
        a1 = rotating_frame_acceleration(r, v)
        k1 = v

        a2 = rotating_frame_acceleration(r + demi_temps * k1, v + demi_temps * a1)
        k2 = v + demi_temps * a1

        a3 = rotating_frame_acceleration(r + demi_temps * k2, v + demi_temps * a2)
        k3 = v + demi_temps * a2

        a4 = rotating_frame_acceleration(r + dt * k3, v + dt * a3)
        k4 = v + dt * a3

        # Mise à jour des positions et vitesses avec les poids RK4
        r += sixieme_temps * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        v += sixieme_temps * (a1 + 2.0 * a2 + 2.0 * a3 + a4)

        # Enregistrement des positions et vitesses
        r_list[n + 1], v_list[n + 1] = r, v

    return r_list, v_list
