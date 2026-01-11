from abc import abstractmethod

import numpy as np


def acceleration(pos, v) -> float:
    pass


def integrate_particle_rk4(pos0, v0, dt, t_max=100.0):
    """
    Intègre la position d'une particule soumise à une acceleration donnée.
    Utilise la méthode Runge-Kutta d'ordre 4 (RK4) pour l'intégration.

    Arguments:
    - r0 : position initiale (vecteur 3D, en km)
    - v0 : vitesse initiale (vecteur 3D, en km/s)
    - nsteps : nombre d'étapes d'intégration
    - t_max : temps total de simulation

    Retourne:
    - pos_list : liste des positions à chaque étape
    - v_list : liste des vitesses à chaque étape
    """

    # Initialisation des paramètres
    nsteps = round(t_max / dt) + 1

    demi_temps = 0.5 * dt
    sixieme_temps = dt / 6.0

    # Positions et vitesses initiales
    pos = pos0
    v = v0

    # Listes pour stocker les positions et vitesses
    pos_list = np.zeros((nsteps + 1, 3))
    v_list = np.zeros((nsteps + 1, 3))
    pos_list[0], v_list[0] = pos, v

    # Intégration de la trajectoire par Runge-Kutta (RK4)
    for n in range(nsteps):
        # Calcul des accélérations à différents points selon la méthode RK4
        a1 = acceleration(pos, v)
        k1 = v

        a2 = acceleration(pos + demi_temps * k1, v + demi_temps * a1)
        k2 = v + demi_temps * a1

        a3 = acceleration(pos + demi_temps * k2, v + demi_temps * a2)
        k3 = v + demi_temps * a2

        a4 = acceleration(pos + dt * k3, v + dt * a3)
        k4 = v + dt * a3

        # Mise à jour des positions et vitesses avec les poids RK4
        pos += sixieme_temps * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        v += sixieme_temps * (a1 + 2.0 * a2 + 2.0 * a3 + a4)

        # Enregistrement des positions et vitesses
        pos_list[n + 1], v_list[n + 1] = pos, v

    return pos_list, v_list
