from typing import List, Tuple

import numpy as np
from celestial_body import CelestialBody
from constants import Constants
from integrator import rk4_step
from lagrange_points import calculate_lagrange_points
from stabilisator import compute_monodromy_matrix
from state import State
from vector import Vector3D


class Simulator:
    """Simulateur principal pour la propagation de trajectoires"""

    def __init__(self):
        """Initialise le simulateur avec le système Soleil-Terre"""
        self.sun = CelestialBody(
            name="Sun",
            mass=Constants.M_SUN,
            initial_position=Vector3D(0, 0, 0),
            initial_velocity=Vector3D(0, 0, 0),
        )

        self.earth = CelestialBody(
            name="Earth",
            mass=Constants.M_EARTH,
            initial_position=Vector3D(Constants.R_EARTH_ORBIT, 0, 0),
            initial_velocity=Vector3D(0, Constants.V_EARTH, 0),
        )

        self.bodies = [self.sun, self.earth]

        self.lagrange_points = calculate_lagrange_points()

        self.history: List[Tuple[float, State]] = []

    def propagate(self, initial_state: State, t_start: float, t_end: float, dt: float):
        """
        Propage la trajectoire d'un satellite, et l'enregistre dans l'historique

        Args:
            initial_state: État initial du satellite
            t_start: Temps de début [s]
            t_end: Temps de fin [s]
            dt: Pas de temps [s]

        Returns:
            Liste de tuples (temps, état)
        """
        history = []
        state = initial_state.copy()
        t = t_start

        history.append((t, state.copy()))

        while t < t_end:
            state = rk4_step(state, t, dt, self.bodies)
            t += dt
            history.append((t, state.copy()))

        self.history = history

    def compute_orbital_period_estimate(self, state: State, t: float) -> float:
        """
        Estime la période orbitale approximative basée sur l'énergie
        Pour les orbites autour de L2, période ~ 1 an

        Returns:
            Période estimée [s]
        """
        # Pour les orbites de halo autour de L2, la période est proche
        # de la période orbitale terrestre
        return Constants.T_YEAR

    def compute_monodromy_for_orbit(
        self, initial_state: State, t_start: float, n_periods: int = 1
    ) -> np.ndarray:
        """
        Calcule la matrice de monodromie pour l'orbite actuelle

        Args:
            initial_state: État initial sur l'orbite
            t_start: Temps de départ
            n_periods: Nombre de périodes à intégrer

        Returns:
            Matrice de monodromie
        """
        period = self.compute_orbital_period_estimate(initial_state, t_start)
        total_time = period * n_periods

        print(f"Calcul de la matrice de monodromie sur {n_periods} période(s)...")
        print(f"Période estimée : {period / (24*3600):.2f} jours")

        M = compute_monodromy_matrix(
            initial_state, t_start, total_time, self.bodies, n_steps=200
        )

        return M

    def get_energy(self, state: State, t: float) -> float:
        """
        Calcule l'énergie mécanique totale du satellite
        E = E_cinétique + E_potentielle

        Args:
            state: État du satellite
            t: Temps [s]

        Returns:
            Énergie totale [J/kg] (énergie spécifique)
        """
        # Énergie cinétique spécifique
        v = state.velocity.norm()
        E_kin = 0.5 * v**2

        # Énergie potentielle spécifique
        E_pot = 0.0
        for body in self.bodies:
            body_state = body.get_state_at_time(t)
            r = (state.position - body_state.position).norm()
            if r > 0:
                E_pot -= body.mu / r

        return E_kin + E_pot
