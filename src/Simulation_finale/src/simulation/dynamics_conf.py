from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .constants import JWSTParameters
from .vectors import PositionVector, StateVector


class DynamicsModel(Enum):
    """Énumération des modèles dynamiques disponibles."""

    CRTBP = "crtbp"
    EPHEMERIS = "ephemeris"
    EPHEMERIS_SRP = "ephemeris_srp"
    FULL = "full"  # Éphémérides + SRP + Relativité


@dataclass
class DynamicsConfig:
    """
    Configuration pour le modèle dynamique.

    Attributs:
        model: Type de modèle à utiliser
        include_moon: Inclure la Lune séparément (vs système Terre-Lune)
        include_srp: Inclure la pression de radiation solaire
        include_relativity: Inclure les effets relativistes
        spacecraft_area_to_mass: Ratio aire/masse [m²/kg]
        reflectivity_coeff: Coefficient de réflectivité [0-1]
    """

    model: DynamicsModel = DynamicsModel.CRTBP
    include_moon: bool = False
    include_srp: bool = False
    include_relativity: bool = False
    spacecraft_area_to_mass: float = JWSTParameters.AREA_TO_MASS_RATIO
    reflectivity_coeff: float = JWSTParameters.REFLECTIVITY_COEFF

    def __post_init__(self):
        """Validation de la configuration."""
        if self.model == DynamicsModel.FULL:
            self.include_srp = True
            self.include_relativity = True

        if self.include_srp and self.spacecraft_area_to_mass <= 0:
            raise ValueError("A/M ratio doit être > 0 si SRP activé")


# ========== CLASSE ABSTRAITE DE BASE ==========


class BaseDynamics(ABC):
    """
    Classe abstraite définissant l'interface pour tous les modèles dynamiques.

    Tous les modèles doivent implémenter:
    - equations_of_motion: calcul de la dérivée d'état
    - compute_acceleration: calcul de l'accélération seule

    Convention:
        État = [x, y, z, vx, vy, vz] en unités SI (m, m/s)
        Temps en secondes depuis J2000.0
    """

    def __init__(self, config: DynamicsConfig):
        """
        Initialise le modèle dynamique.

        Args:
            config: Configuration du modèle
        """
        self.config = config

    @abstractmethod
    def equations_of_motion(self, t: float, state: StateVector) -> StateVector:
        """
        Calcule la dérivée temporelle de l'état.

        Args:
            t: Temps [s] depuis J2000.0
            state: État [x, y, z, vx, vy, vz]

        Returns:
            Dérivée d'état [vx, vy, vz, ax, ay, az]
        """
        pass

    @abstractmethod
    def compute_acceleration(self, t: float, state: StateVector) -> PositionVector:
        """
        Calcule uniquement l'accélération.

        Args:
            t: Temps [s]
            state: État [x, y, z, vx, vy, vz]

        Returns:
            Accélération [ax, ay, az] [m/s²]
        """
        pass

    def __call__(self, t: float, state: StateVector) -> StateVector:
        """
        Permet d'utiliser l'objet comme une fonction.
        Utile pour les intégrateurs scipy.
        """
        return self.equations_of_motion(t, state)
