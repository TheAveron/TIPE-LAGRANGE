"""
integrator.py — Intégrateur RK4 générique à pas fixe.

Choix RK4 :
  - Ordre 4 : erreur locale en O(h^5), erreur globale en O(h^4).
  - Pas fixe : simplicité, débogage facile, contrôle explicite du coût.
  - Suffisant pour ~3-4 révolutions quasi-halo (~2 ans) avec h ~ 1 h.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def rk4_step(
    f: Callable[[np.float64, NDArray[np.float64]], NDArray[np.float64]],
    t: np.float64,
    y: NDArray[np.float64],
    h: np.float64,
) -> NDArray[np.float64]:
    """
    Un pas RK4 classique.

    Parameters
    ----------
    f : callable(t, y) -> dy/dt
        Fonction dérivée.
    t : np.float64
        Temps courant.
    y : NDArray[np.float64]
        État courant.
    h : np.float64
        Pas de temps.

    Returns
    -------
    y_next : NDArray[np.float64]
        État après un pas.
    """

    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)

    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    f: Callable[[np.float64, NDArray[np.float64]], NDArray[np.float64]],
    y0: NDArray[np.float64],
    t0: np.float64,
    t_end: np.float64,
    h: np.float64,
    callback: Callable[[np.float64, NDArray[np.float64]], None] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Intégration RK4 sur [t0, t_end].

    Parameters
    ----------
    f : callable(t, y) -> dy/dt
    y0 : NDArray[np.float64]
        État initial.
    t0, t_end : np.float64
        Bornes de l'intégration.
    h : np.float64
        Pas de temps fixe.
    callback : callable(t, y), optional
        Appelé à chaque pas (ex. pour enregistrer des diagnostics).

    Returns
    -------
    t_arr : NDArray[np.float64], shape (N,)
    y_arr : NDArray[np.float64], shape (N, len(y0))
    """

    n_steps = int(np.ceil((t_end - t0) / h))
    # Dernier pas potentiellement raccourci
    times = np.empty(n_steps + 1, dtype=np.float64)
    states = np.empty((n_steps + 1, len(y0)), dtype=np.float64)

    t = t0
    y = y0.copy()
    times[0] = t
    states[0] = y

    if callback is not None:
        callback(t, y)

    for i in range(1, n_steps + 1):
        h_eff = min(h, t_end - t)  # raccourcit le dernier pas si nécessaire
        y = rk4_step(f, t, y, h_eff)
        t += h_eff
        times[i] = t
        states[i] = y

        if callback is not None:
            callback(t, y)

    return times, states
