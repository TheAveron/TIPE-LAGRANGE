# Dans dynamics_conf.py ou un nouveau fichier base_controller.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from src.Models.vectors import StateVector
import numpy as np

from src.Simulations.constants import JWSTParameters


class StationKeepingStrategy(Enum):
    """Stratégies de station-keeping disponibles."""

    TARGET_POINT = "target_point"  # Viser un point sur orbite nominale
    FLOQUET_MODE = "floquet_mode"  # Contrôle par modes propres
    ADAPTIVE = "adaptive"  # Adaptatif selon écart
    JWST_OPERATIONAL = "jwst_operational"  # Stratégie opérationnelle JWST


@dataclass
class ManeuverPlan:
    """
    Plan de manœuvre de station-keeping.

    Attributs:
        time: Temps de la manœuvre [s] depuis epoch
        delta_v: Vecteur ΔV [vx, vy, vz] [m/s]
        magnitude: Magnitude du ΔV [m/s]
        state_before: État avant manœuvre [m, m/s]
        state_after: État après manœuvre [m, m/s]
        reason: Raison de la manœuvre
        cost_estimate: Coût estimé en propergol [kg] (optionnel)
    """

    time: float
    delta_v: np.ndarray
    magnitude: float
    state_before: np.ndarray
    state_after: np.ndarray
    reason: str
    cost_estimate: float | None = None

    def __str__(self) -> str:
        return (
            f"Manœuvre SK à t={self.time/86400:.1f} jours:\n"
            f"  ΔV = [{self.delta_v[0]:.4f}, {self.delta_v[1]:.4f}, {self.delta_v[2]:.4f}] m/s\n"
            f"  |ΔV| = {self.magnitude:.4f} m/s\n"
            f"  Raison: {self.reason}"
        )


@dataclass
class StationKeepingConstraints:
    """
    Contraintes pour le station-keeping.

    Définit les limites acceptables de l'orbite avant correction.
    """

    # Amplitudes maximales [m]
    max_amplitude_y: float = JWSTParameters.ORBIT_Y_MAX
    max_amplitude_z: float = JWSTParameters.ORBIT_Z_MAX

    # Distance maximale à L2 [m]
    max_distance_to_l2: float = 2.0e9  # 2 million km

    # Écart maximal en position [m]
    max_position_error: float = 200e6  # 200,000 km

    # Écart maximal en vitesse [m/s]
    max_velocity_error: float = 10.0  # 10 m/s

    # Cadence minimale entre manœuvres [jours]
    min_maneuver_spacing: float = JWSTParameters.SK_CADENCE_DAYS

    # ΔV maximal par manœuvre [m/s]
    max_delta_v_per_maneuver: float = 2000.0


class BaseStationKeepingController(ABC):
    maneuver_history: list[ManeuverPlan]

    @abstractmethod
    def check_maneuver_needed(
        self, current_state: StateVector, time: float, last_maneuver_time: float | None
    ) -> tuple[bool, str]:
        pass

    @abstractmethod
    def compute_maneuver(self, current_state: StateVector, time: float) -> ManeuverPlan:
        pass

    def apply_maneuver(self, maneuver: ManeuverPlan) -> StateVector:
        self.maneuver_history.append(maneuver)
        return maneuver.state_after.copy()
