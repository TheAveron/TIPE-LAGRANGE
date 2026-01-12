"""
Constants physiques et paramètres pour la simulation JWST L2.

Organisation:
- Constants: constantes physiques fondamentales et système Soleil-Terre
- JWSTParameters: paramètres spécifiques à la mission JWST
- NumericalConstants: paramètres pour les calculs numériques

Sources principales:
- JPL DE440/DE441 Ephemeris
- IAU 2015 Nominal Values
- CODATA 2018
- Documents JWST mission (voir README)
"""

from math import pi
from typing import Final


class Constants:
    """
    Constantes physiques fondamentales et paramètres du système Soleil-Terre.

    Toutes les valeurs sont en unités SI sauf indication contraire.
    """

    # ========== CONSTANTES FONDAMENTALES ==========

    G: Final[float] = 6.67430e-11  # Constante gravitationnelle [m³ kg⁻¹ s⁻²]
    C: Final[float] = 299792458.0  # Vitesse de la lumière [m/s]

    # Paramètres post-Newtoniens (General Relativity)
    BETA_PPN: Final[float] = 1.0
    GAMMA_PPN: Final[float] = 1.0

    # ========== MASSES [kg] ==========

    M_SUN: Final[float] = 1.988416e30
    M_EARTH: Final[float] = 5.9724e24
    M_MOON: Final[float] = 7.346e22
    M_EARTH_MOON: Final[float] = M_EARTH + M_MOON

    # ========== DISTANCES [m] ==========

    AU: Final[float] = 149597870700.0
    R_EARTH_ORBIT: Final[float] = AU
    A_EARTH: Final[float] = 1.00000011 * AU
    E_EARTH: Final[float] = 0.0167086  # Excentricité

    R_EARTH_MOON: Final[float] = 384400000.0
    R_SOI_EARTH: Final[float] = 924644727.0  # Sphère d'influence terrestre

    # ========== PÉRIODES [s] ==========

    T_YEAR_SIDEREAL: Final[float] = 365.256363004 * 86400.0
    T_YEAR: Final[float] = 365.25 * 86400.0
    T_MOON: Final[float] = 27.321661 * 86400.0  # Mois sidéral

    # ========== VITESSES ANGULAIRES [rad/s] ==========

    OMEGA_EARTH: Final[float] = 2.0 * pi / T_YEAR_SIDEREAL
    N_EARTH: Final[float] = OMEGA_EARTH  # Notation alternative
    OMEGA_MOON: Final[float] = 2.0 * pi / T_MOON

    # ========== VITESSES [m/s] ==========

    V_EARTH: Final[float] = OMEGA_EARTH * R_EARTH_ORBIT
    V_EARTH_PERIHELION: Final[float] = V_EARTH * (1 + E_EARTH)
    V_EARTH_APHELION: Final[float] = V_EARTH * (1 - E_EARTH)

    # ========== PARAMÈTRES GRAVITATIONNELS [m³/s²] ==========
    # Calculés comme μ_corps = G * masse_corps

    MU_SUN: Final[float] = 1.32712440018e20
    MU_EARTH: Final[float] = 3.986004418e14
    MU_MOON: Final[float] = 4.9028e12
    MU_SUN_EARTH_SYSTEM: Final[float] = MU_SUN + MU_EARTH

    # ========== PARAMÈTRES CRTBP (adimensionnels) ==========

    MU_RATIO_SUN_EARTH: Final[float] = M_EARTH / (M_SUN + M_EARTH)
    MU_RATIO_EARTH_MOON: Final[float] = M_MOON / (M_EARTH + M_MOON)

    # ========== ORIENTATION ==========

    EPSILON_EARTH: Final[float] = 23.43928 * pi / 180.0  # Obliquité J2000
    PRECESSION_RATE: Final[float] = (50.3 / 3600.0 * pi / 180.0) / T_YEAR_SIDEREAL


class JWSTParameters:
    """Paramètres spécifiques à la mission JWST."""

    # Spacecraft
    MASS_DRY: Final[float] = 6500.0  # [kg]
    SUNSHIELD_AREA: Final[float] = 161.0  # [m²]
    AREA_TO_MASS_RATIO: Final[float] = SUNSHIELD_AREA / MASS_DRY
    REFLECTIVITY_COEFF: Final[float] = 0.9

    # SRP
    SRP_ACCEL_MIN: Final[float] = 1.15e-13  # [m/s²] (converti de km/s²)
    SRP_ACCEL_MAX: Final[float] = 2.05e-13  # [m/s²]
    SRP_OFFSET_ANGLE_MAX: Final[float] = 24.0  # [degrés]

    # Attitude
    SUN_PITCH_MIN: Final[float] = -53.0  # [degrés]
    SUN_PITCH_MAX: Final[float] = 0.0
    SUN_ROLL_MIN: Final[float] = -5.0
    SUN_ROLL_MAX: Final[float] = 5.0
    SUN_ROLL_MANEUVER: Final[float] = 0.0
    SUN_YAW_MIN: Final[float] = -180.0
    SUN_YAW_MAX: Final[float] = 180.0

    # Delta-V
    DELTA_V_MCC_TOTAL: Final[float] = 66.5  # [m/s]
    DELTA_V_SK_PER_YEAR: Final[float] = 2.43  # [m/s/an]

    # Station-keeping
    SK_CADENCE_DAYS: Final[float] = 21.0
    SK_TARGET_CROSSINGS: Final[int] = 4

    # Orbite
    ORBIT_AMPLITUDE_Y: Final[float] = 771000e3  # [m] = 771,000 km
    ORBIT_AMPLITUDE_Z: Final[float] = 418000e3  # [m] = 418,000 km
    ORBIT_Y_MAX: Final[float] = 832000e3  # [m]
    ORBIT_Z_MAX: Final[float] = 520000e3  # [m]
    DISTANCE_TO_L2: Final[float] = 1500000e3  # [m] = 1.5 million km


class NumericalConstants:
    """Paramètres pour l'intégration numérique."""

    # Tolérances
    INTEGRATION_ATOL_HIGH: Final[float] = 1e-12
    INTEGRATION_RTOL_HIGH: Final[float] = 1e-12
    INTEGRATION_ATOL_MEDIUM: Final[float] = 1e-10
    INTEGRATION_RTOL_MEDIUM: Final[float] = 1e-10

    # Convergence
    LAGRANGE_POINT_TOL: Final[float] = 1e-12
    DIFFERENTIAL_CORRECTOR_TOL_VEL: Final[float] = 1e-6  # [m/s]
    DIFFERENTIAL_CORRECTOR_TOL_POS: Final[float] = 1e-3  # [m]
    MAX_ITERATIONS: Final[int] = 100

    # Événements
    CROSSING_DETECTION_TOL: Final[float] = 1e-6  # [m]

    # Normalisation CRTBP
    LENGTH_UNIT: Final[float] = Constants.AU
    TIME_UNIT: Final[float] = 1.0 / Constants.OMEGA_EARTH
    VELOCITY_UNIT: Final[float] = LENGTH_UNIT * Constants.OMEGA_EARTH
