from typing import Callable

import numpy as np

from .coordinates import PositionVector, StateVector


def integrate_particle_rk4(
    fun: Callable[[PositionVector, PositionVector], float],
    state0: StateVector,
    dt: float,
    t_max=100.0,
):
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
    nsteps = round(t_max / dt) + 1

    demi_temps = 0.5 * dt
    sixieme_temps = dt / 6.0

    pos = state0[:3].copy()
    v = state0[3:].copy()

    state_list = np.zeros((nsteps + 1, 6))

    state_list[0] = state0.copy()

    for n in range(nsteps):
        a1 = fun(pos, v)
        k1 = v

        a2 = fun(pos + demi_temps * k1, v + demi_temps * a1)
        k2 = v + demi_temps * a1

        a3 = fun(pos + demi_temps * k2, v + demi_temps * a2)
        k3 = v + demi_temps * a2

        a4 = fun(pos + dt * k3, v + dt * a3)
        k4 = v + dt * a3

        pos += sixieme_temps * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        v += sixieme_temps * (a1 + 2.0 * a2 + 2.0 * a3 + a4)

        state_list[n + 1, :3] = pos
        state_list[n + 1, 3:] = v

    return state_list
