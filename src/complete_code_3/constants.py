from numpy import pi


class Constants:
    """Constantes physiques du syst�me Soleil-Terre"""

    # Constante gravitationnelle [m^3 kg^-1 s^-2]
    G = 6.67430e-11

    # Masses [kg]
    M_SUN = 1.98892e30  # Masse du Soleil
    M_EARTH = 5.97219e24  # Masse de la Terre

    # Distance Terre-Soleil (1 UA) [m]
    AU = 1.495978707e11

    # Rayon de l'orbite terrestre (circulaire) [m]
    R_EARTH_ORBIT = AU

    # Vitesse angulaire du système Soleil-Terre [rad/s]
    # ω = 2π / T où T = 365.25 jours
    T_YEAR = 365.25 * 24 * 3600  # Période en secondes
    OMEGA = 2 * pi / T_YEAR

    # Vitesse orbitale de la Terre [m/s]
    V_EARTH = OMEGA * R_EARTH_ORBIT

    # Paramètre gravitationnel du Soleil μ_S = G*M_S [m^3/s^2]
    MU_SUN = G * M_SUN

    # Paramètre gravitationnel de la Terre μ_T = G*M_T [m^3/s^2]
    MU_EARTH = G * M_EARTH
