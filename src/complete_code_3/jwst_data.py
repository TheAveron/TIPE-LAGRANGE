from typing import List

import numpy as np
from constants import Constants
from state import State
from vector import Vector3D


class JWSTData:
    """Données réelles du télescope spatial James Webb"""

    # Masse de JWST [kg]
    MASS = 6200.0  # Environ 6.2 tonnes

    # Amplitudes typiques de l'orbite quasi-halo (depuis document NASA)
    # Pour une date de lancement typique (Janvier 2021)
    AMPLITUDE_Y = 771000e3  # 771,000 km en mètres
    AMPLITUDE_Z = 418000e3  # 418,000 km en mètres

    # Distance typique Terre-L2 (varie avec excentricité)
    EARTH_L2_DISTANCE = 1.5e9  # ~1.5 million de km en mètres

    # Cadence de station-keeping
    STATION_KEEPING_PERIOD = 21 * 24 * 3600  # 21 jours en secondes

    # Budget delta-v pour station-keeping
    STATION_KEEPING_BUDGET = 0.0665  # 66.5 m/s pour toute la mission (10.5 ans)

    # Contraintes d'attitude (angles en degrés)
    SUN_PITCH_MIN = -53.0
    SUN_PITCH_MAX = 0.0
    SUN_ROLL_MIN = -5.0
    SUN_ROLL_MAX = 5.0
    # Sun yaw : libre entre ±180°


def generate_jwst_halo_initial_conditions(
    t: float = 0.0,
    amplitude_y: float = JWSTData.AMPLITUDE_Y,
    amplitude_z: float = JWSTData.AMPLITUDE_Z,
    phase_y: float = 0.0,
    phase_z: float = np.pi / 2,
) -> State:
    """
    Génère des conditions initiales pour une orbite quasi-halo de type JWST
    autour de L2 dans le référentiel héliocentrique inertiel.

    Basé sur les données réelles de JWST (amplitudes Y~771,000 km, Z~418,000 km)

    Args:
        t: Temps initial [s]
        amplitude_y: Amplitude dans la direction Y [m]
        amplitude_z: Amplitude dans la direction Z [m]
        phase_y: Phase initiale pour Y [rad]
        phase_z: Phase initiale pour Z [rad]

    Returns:
        État initial dans le référentiel héliocentrique inertiel
    """
    # Position de L2 dans le référentiel héliocentrique à t=0
    # (à t=0, la Terre est sur l'axe +x)
    mu = Constants.M_EARTH / (Constants.M_SUN + Constants.M_EARTH)
    r_earth = Constants.R_EARTH_ORBIT
    r_L2 = r_earth * (1 + (mu / 3) ** (1 / 3))

    # Période approximative de l'orbite de halo
    # Pour les orbites autour de L2, période ~ période de la Terre
    T_halo = Constants.T_YEAR
    omega_halo = 2 * np.pi / T_halo

    # Position dans le référentiel tournant (RLP)
    # Approximation d'une orbite de Lissajous/quasi-halo
    x_rlp = JWSTData.EARTH_L2_DISTANCE  # Proche de L2 en x
    y_rlp = amplitude_y * np.sin(omega_halo * t + phase_y)
    z_rlp = amplitude_z * np.sin(omega_halo * t + phase_z)

    # Vitesse dans le référentiel tournant
    vx_rlp = 0.0  # Faible vitesse en x
    vy_rlp = amplitude_y * omega_halo * np.cos(omega_halo * t + phase_y)
    vz_rlp = amplitude_z * omega_halo * np.cos(omega_halo * t + phase_z)

    # Angle de rotation du référentiel tournant
    theta = Constants.OMEGA * t
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Conversion du référentiel tournant vers l'inertiel
    # Position : rotation + ajout de la position de la Terre
    x_inertial = (r_earth + x_rlp) * cos_theta - y_rlp * sin_theta
    y_inertial = (r_earth + x_rlp) * sin_theta + y_rlp * cos_theta
    z_inertial = z_rlp

    # Vitesse : rotation + vitesse orbitale de la Terre
    # v_inertial = R(theta) * v_rlp + omega x r_inertial
    vx_temp = vx_rlp * cos_theta - vy_rlp * sin_theta
    vy_temp = vx_rlp * sin_theta + vy_rlp * cos_theta
    vz_temp = vz_rlp

    # Ajout de la composante due à la rotation du référentiel
    vx_inertial = vx_temp - Constants.OMEGA * y_inertial
    vy_inertial = vy_temp + Constants.OMEGA * x_inertial
    vz_inertial = vz_temp

    return State(
        position=Vector3D(x_inertial, y_inertial, z_inertial),
        velocity=Vector3D(vx_inertial, vy_inertial, vz_inertial),
    )
