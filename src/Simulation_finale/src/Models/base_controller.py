# Dans dynamics_conf.py ou un nouveau fichier base_controller.py
from abc import ABC, abstractmethod

from src.Models.vectors import StateVector
from src.Simulations.station_keeping import ManeuverPlan


class BaseStationKeepingController(ABC):
    maneuver_history: list[ManeuverPlan]

    @abstractmethod
    def check_maneuver_needed(
        self, state: StateVector, time: float, last_maneuver_time: float | None
    ) -> tuple[bool, str]:
        pass

    @abstractmethod
    def compute_maneuver(self, state: StateVector, time: float) -> ManeuverPlan:
        pass

    def apply_maneuver(self, maneuver: ManeuverPlan) -> StateVector:
        self.maneuver_history.append(maneuver)
        return maneuver.state_after.copy()
