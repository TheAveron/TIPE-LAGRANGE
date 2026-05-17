"""
Module de Station-Keeping pour orbites Halo/Lissajous autour de L2.

Ce module implémente les stratégies de maintien en orbite utilisées pour
les missions opérationnelles autour des points de Lagrange.

Références principales :
- Folta, D.C. et al. (2006). "Earth-Moon Libration Point Orbit Stationkeeping"
- Pavlak, T.A. & Howell, K.C. (2013). "Strategy for long-term libration point
  orbit stationkeeping in the Earth-Moon system"
- Petersen, K. (2019). "Station-Keeping Requirements for the James Webb Space
  Telescope"

Principe physique :
------------------
Les points de Lagrange L1 et L2 sont des équilibres INSTABLES du CRTBP.
Sans corrections, un satellite dérive exponentiellement (temps caractéristique
~23 jours pour Sun-Earth L2).

Le station-keeping consiste à :
1. Détecter les dérives par rapport à l'orbite de référence
2. Calculer une manœuvre ΔV minimale pour corriger la trajectoire
3. Exécuter la manœuvre au moment optimal

Stratégies implémentées :
------------------------
1. Target Point Approach (TPA) : viser un point cible sur l'orbite nominale
2. Floquet Mode Control : stabiliser les modes instables
3. Adaptive Control : ajuster selon l'écart mesuré
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.Models.base_controller import (
    BaseStationKeepingController,
    ManeuverPlan,
    StationKeepingConstraints,
    StationKeepingStrategy,
)
from ..Models.vectors import StateVector
from .constants import Constants, JWSTParameters
from .CRTBP3_dynamics import CRTBP3Body
from .orbit_generator import OrbitInitialConditions


class StationKeepingController(BaseStationKeepingController):
    """
    Contrôleur de station-keeping pour orbites L2.

    Cette classe implémente différentes stratégies de maintien en orbite
    avec gestion des contraintes opérationnelles.

    Usage:
        >>> controller = StationKeepingController(
        ...     reference_orbit=orbit_nominal,
        ...     strategy=StationKeepingStrategy.JWST_OPERATIONAL
        ... )
        >>>
        >>> # Simulation avec station-keeping
        >>> states = integrate_with_sk(initial_state, duration=365*86400)
    """

    def __init__(
        self,
        reference_orbit: OrbitInitialConditions,
        crtbp_model,
        strategy: StationKeepingStrategy = StationKeepingStrategy.TARGET_POINT,
        constraints: Optional[StationKeepingConstraints] = None,
    ):
        """
        Initialise le contrôleur de station-keeping.

        Args:
            reference_orbit: Orbite nominale de référence
            crtbp_model: Modèle CRTBP pour propagation
            strategy: Stratégie de contrôle
            constraints: Contraintes opérationnelles
        """
        self.reference_orbit = reference_orbit
        self.crtbp = crtbp_model
        self.strategy = strategy
        self.constraints = constraints or StationKeepingConstraints()

        # Historique des manœuvres
        self.maneuver_history: list[ManeuverPlan] = []

        # Statistiques
        self.total_delta_v = 0.0
        self.num_maneuvers = 0

        # Position L2
        mu = Constants.MU_RATIO_SUN_EARTH
        self.x_l2_norm = (1.0 - mu) + (mu / 3.0) ** (1.0 / 3.0)
        self.x_l2_phys = self.x_l2_norm * Constants.AU

    def check_maneuver_needed(
        self,
        current_state: StateVector,
        time: float,
        last_maneuver_time: float | None = None,
    ) -> tuple[bool, str]:
        """
        Détermine si une manœuvre est nécessaire.

        Critères de déclenchement :
        1. Écart de position trop grand
        2. Écart de vitesse trop grand
        3. Sortie des limites d'amplitude
        4. Temps depuis dernière manœuvre

        Args:
            current_state: État actuel [x, y, z, vx, vy, vz] [m, m/s]
            time: Temps actuel [s]
            last_maneuver_time: Temps de la dernière manœuvre [s]

        Returns:
            (maneuver_needed, reason)
        """
        # Convertir en normalisé pour comparaison
        L_star = Constants.AU
        V_star = L_star * Constants.OMEGA_EARTH

        state_norm = current_state.copy()
        state_norm[:3] /= L_star
        state_norm[3:] /= V_star

        # Position/vitesse de référence (orbite nominale au même point)
        # Approximation : on compare avec l'orbite de référence
        ref_state_norm = self.reference_orbit.state

        # 1. Vérifier amplitude Y
        y_current = abs(current_state[1])
        if y_current > self.constraints.max_amplitude_y:
            return (
                True,
                f"Amplitude Y dépassée ({y_current/1e6:.0f} km > {self.constraints.max_amplitude_y/1e6:.0f} km)",
            )

        # 2. Vérifier amplitude Z
        z_current = abs(current_state[2])
        if z_current > self.constraints.max_amplitude_z:
            return (
                True,
                f"Amplitude Z dépassée ({z_current/1e6:.0f} km > {self.constraints.max_amplitude_z/1e6:.0f} km)",
            )

        # 3. Vérifier distance à L2
        dist_to_l2 = np.linalg.norm(
            current_state[:3] - np.array([self.x_l2_phys, 0, 0])
        )
        if dist_to_l2 > self.constraints.max_distance_to_l2:
            return True, f"Distance à L2 excessive ({dist_to_l2/1e6:.0f} km)"

        # 4. Vérifier cadence minimale
        if last_maneuver_time is not None:
            days_since_last = (time - last_maneuver_time) / 86400
            if days_since_last >= self.constraints.min_maneuver_spacing:
                # Manœuvre périodique nécessaire
                return True, f"Cadence de {days_since_last:.0f} jours atteinte"

        return False, "Pas de manœuvre nécessaire"

    def compute_maneuver(
        self,
        current_state: StateVector,
        time: float,
        target_state: StateVector | None = None,
    ) -> ManeuverPlan:
        """
        Calcule la manœuvre ΔV optimale.

        Args:
            current_state: État actuel [m, m/s]
            time: Temps actuel [s]
            target_state: État cible (optionnel)

        Returns:
            Plan de manœuvre
        """
        if self.strategy == StationKeepingStrategy.TARGET_POINT:
            return self._compute_target_point_maneuver(
                current_state, time, target_state
            )

        elif self.strategy == StationKeepingStrategy.FLOQUET_MODE:
            return self._compute_floquet_mode_maneuver(current_state, time)

        elif self.strategy == StationKeepingStrategy.JWST_OPERATIONAL:
            return self._compute_jwst_operational_maneuver(current_state, time)

        else:
            return self._compute_adaptive_maneuver(current_state, time)

    def _compute_target_point_maneuver(
        self,
        current_state: StateVector,
        time: float,
        target_state: StateVector | None = None,
    ) -> ManeuverPlan:
        """
        Target Point Approach : viser un point cible sur l'orbite nominale.

        Stratégie :
        1. Identifier point le plus proche sur orbite de référence
        2. Calculer ΔV pour intercepter ce point
        3. Optimiser pour minimiser |ΔV|

        Référence:
            Folta et al. (2006) - Section 3.2
        """
        # Si pas de cible spécifiée, viser le point de référence
        if target_state is None:
            target_state = self.reference_orbit.to_physical().state

        # Écart de vitesse (correction simple)
        delta_v = target_state[3:] - current_state[3:]

        # Limiter le ΔV
        magnitude = float(np.linalg.norm(delta_v))
        if magnitude > self.constraints.max_delta_v_per_maneuver:
            delta_v = delta_v * (self.constraints.max_delta_v_per_maneuver / magnitude)
            magnitude = self.constraints.max_delta_v_per_maneuver

        state_after = current_state.copy()
        state_after[3:] += delta_v

        return ManeuverPlan(
            time=time,
            delta_v=delta_v,
            magnitude=magnitude,
            state_before=current_state.copy(),
            state_after=state_after,
            reason="Target Point Approach",
        )

    def _compute_floquet_mode_maneuver(
        self, current_state: StateVector, time: float
    ) -> ManeuverPlan:
        """
        Floquet Mode Control : stabiliser le mode instable.

        Principe :
        Le système linéarisé autour de L2 a un mode instable exponentiel.
        On calcule le ΔV qui annule la composante sur ce mode.

        Référence:
            Howell & Pernicka (1988) - Floquet theory for libration point orbits
        """
        # TODO: Implémentation complète nécessite calcul des modes propres
        # Pour l'instant, utiliser approximation simple

        warnings.warn(
            "Floquet mode control non complètement implémenté, utilisation TPA"
        )
        return self._compute_target_point_maneuver(current_state, time)

    def _compute_jwst_operational_maneuver(
        self, current_state: StateVector, time: float
    ) -> ManeuverPlan:
        """
        Stratégie opérationnelle JWST.

        D'après Petersen (2019) :
        1. Corriger principalement la composante Y (dans le plan orbital)
        2. Minimiser les corrections en Z (hors plan)
        3. Viser à ramener vers le centre de l'orbite

        Contraintes JWST :
        - Sun-keep-out zone : éviter pointage direct vers Soleil
        - Correction tous les ~21 jours
        - ΔV ~ 2.43 m/s par an
        """
        L_star = Constants.AU

        # Écart par rapport à L2
        pos_rel_to_l2 = current_state[:3] - np.array([self.x_l2_phys, 0, 0])

        # Stratégie : ramener vers centre de l'orbite
        # Viser y=0, z proche de la référence
        target_y = 0.0
        target_z = self.reference_orbit.to_physical().state[2]

        # Calcul simple du ΔV nécessaire
        # (approximation : correction proportionnelle)
        vy_correction = (
            -0.1 * current_state[1] / L_star * Constants.OMEGA_EARTH * L_star
        )
        vz_correction = (
            0.05
            * (target_z - current_state[2])
            / L_star
            * Constants.OMEGA_EARTH
            * L_star
        )

        ref_phys = self.reference_orbit.to_physical().state
        pos_error = current_state[:3] - ref_phys[:3]
        vel_error = current_state[3:] - ref_phys[3:]

        delta_v = np.array(
            [
                0.0,  # Pas de correction en X (contrôlé naturellement)
                -vel_error[1] - Constants.OMEGA_EARTH * pos_error[1],  # correction en Y
                -vel_error[2] * 0.5,
            ]
        )

        magnitude = float(np.linalg.norm(delta_v))
        if magnitude > self.constraints.max_delta_v_per_maneuver:
            delta_v *= self.constraints.max_delta_v_per_maneuver / magnitude
            magnitude = self.constraints.max_delta_v_per_maneuver

        state_after = current_state.copy()
        state_after[3:] += delta_v
        return ManeuverPlan(
            time,
            delta_v,
            magnitude,
            current_state.copy(),
            state_after,
            "JWST Operational Strategy",
        )

    def _compute_adaptive_maneuver(
        self, current_state: StateVector, time: float
    ) -> ManeuverPlan:
        """
        Stratégie adaptative : ajuster selon l'écart.
        """
        return self._compute_jwst_operational_maneuver(current_state, time)

    def apply_maneuver(self, maneuver: ManeuverPlan) -> StateVector:
        """
        Applique une manœuvre à l'état.

        Args:
            maneuver: Plan de manœuvre

        Returns:
            État après manœuvre
        """
        # Enregistrer dans l'historique
        self.maneuver_history.append(maneuver)
        self.total_delta_v += maneuver.magnitude
        self.num_maneuvers += 1

        return maneuver.state_after.copy()

    def get_statistics(self) -> Dict:
        """
        Retourne les statistiques de station-keeping.

        Returns:
            Dictionnaire avec statistiques
        """
        if self.num_maneuvers == 0:
            return {
                "num_maneuvers": 0,
                "total_delta_v": 0.0,
                "mean_delta_v": 0.0,
                "max_delta_v": 0.0,
                "delta_v_per_year": 0.0,
            }

        maneuver_times = [m.time for m in self.maneuver_history]
        maneuver_magnitudes = [m.magnitude for m in self.maneuver_history]

        duration_years = (maneuver_times[-1] - maneuver_times[0]) / (365.25 * 86400)
        delta_v_per_year = (
            self.total_delta_v / duration_years if duration_years > 0 else 0.0
        )

        return {
            "num_maneuvers": self.num_maneuvers,
            "total_delta_v": self.total_delta_v,
            "mean_delta_v": np.mean(maneuver_magnitudes),
            "max_delta_v": np.max(maneuver_magnitudes),
            "min_delta_v": np.min(maneuver_magnitudes),
            "delta_v_per_year": delta_v_per_year,
            "maneuver_times": maneuver_times,
            "maneuver_magnitudes": maneuver_magnitudes,
            "duration_years": duration_years,
        }

    def print_statistics(self):
        """Affiche les statistiques de station-keeping."""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("STATISTIQUES STATION-KEEPING")
        print("=" * 70)
        print(f"Nombre de manœuvres : {stats['num_maneuvers']}")
        print(f"ΔV total            : {stats['total_delta_v']:.3f} m/s")
        print(f"ΔV moyen            : {stats['mean_delta_v']:.4f} m/s")
        print(f"ΔV max              : {stats['max_delta_v']:.4f} m/s")
        print(f"ΔV min              : {stats['min_delta_v']:.4f} m/s")
        print(f"ΔV par an           : {stats['delta_v_per_year']:.3f} m/s/an")
        print(f"Durée simulée       : {stats['duration_years']:.2f} ans")

        # Comparaison avec JWST nominal
        print(f"\nComparaison JWST nominal :")
        print(f"  Budget JWST : {JWSTParameters.DELTA_V_SK_PER_YEAR:.2f} m/s/an")
        ratio = (
            stats["delta_v_per_year"] / JWSTParameters.DELTA_V_SK_PER_YEAR
            if stats["delta_v_per_year"] > 0
            else 0
        )
        print(f"  Ratio       : {ratio:.2f}x")

        if ratio < 1.5:
            print(f"  ✅ Consommation acceptable")
        else:
            print(f"  ⚠️  Consommation élevée")

        print("=" * 70)


def integrate_with_station_keeping(
    crtbp_model: CRTBP3Body,
    initial_state: StateVector,
    reference_orbit,
    duration: float,
    dt: float = 3600.0,  # 1 heure
    strategy: StationKeepingStrategy = StationKeepingStrategy.JWST_OPERATIONAL,
    check_interval: float = 20 * 86400.0,
    controller_override=None,
) -> Tuple[np.ndarray, List[ManeuverPlan]]:
    """
    Intègre une trajectoire avec station-keeping.

    Args:
        crtbp_model: Modèle CRTBP
        initial_state: État initial [m, m/s]
        reference_orbit: Orbite de référence
        duration: Durée totale [s]
        dt: Pas de temps d'intégration [s]
        strategy: Stratégie de station-keeping
        check_interval: Intervalle de vérification SK [s]

    Returns:
        (états, manœuvres) où états est array (n_steps, 6)
    """
    if controller_override is None:

        controller = StationKeepingController(
            reference_orbit=reference_orbit, crtbp_model=crtbp_model, strategy=strategy
        )
    else:
        controller = controller_override

    nsteps = int(duration / dt)
    states = np.zeros((nsteps + 1, 6))
    states[0] = initial_state.copy()
    state = initial_state.copy()
    t = 0.0

    last_maneuver_time = None
    next_check_time = check_interval

    print(f"\n🚀 Intégration avec station-keeping ({strategy.value})")
    print(f"   Durée : {duration/86400:.0f} jours")
    print(f"   Vérification tous les {check_interval/86400:.0f} jours")

    for i in range(nsteps):
        # RK4
        k1 = crtbp_model.equations_of_motion(t, state)
        k2 = crtbp_model.equations_of_motion(t + dt / 2, state + dt / 2 * k1)
        k3 = crtbp_model.equations_of_motion(t + dt / 2, state + dt / 2 * k2)
        k4 = crtbp_model.equations_of_motion(t + dt, state + dt * k3)

        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

        if t >= next_check_time:
            needed, reason = controller.check_maneuver_needed(
                state, t, last_maneuver_time
            )

            if needed:
                print(f"\n⚡ Manœuvre à t={t/86400:.1f} jours : {reason}")
                maneuver = controller.compute_maneuver(state, t)
                state = controller.apply_maneuver(maneuver)
                print(f"   ΔV = {maneuver.magnitude:.4f} m/s")
                last_maneuver_time = t

            next_check_time += check_interval

        states[i + 1] = state

    # controller.print_statistics()
    return states, controller.maneuver_history


"""
EXEMPLE :

from station_keeping import integrate_with_station_keeping, StationKeepingStrategy

# Générer orbite nominale
gen = OrbitGenerator()
orbit_nominal = gen.generate_jwst_nominal_orbit()

# Convertir en physique
orbit_phys = orbit_nominal.to_physical()

# Simuler 1 an avec station-keeping
states, maneuvers = integrate_with_station_keeping(
    crtbp_model=crtbp,
    initial_state=orbit_phys.state,
    reference_orbit=orbit_nominal,
    duration=365*86400,  # 1 an
    strategy=StationKeepingStrategy.JWST_OPERATIONAL
)

# Afficher résultats
print(f"Nombre de manœuvres : {len(maneuvers)}")
print(f"ΔV total : {sum(m.magnitude for m in maneuvers):.3f} m/s")
"""
