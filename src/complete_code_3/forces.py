from celestial_body import CelestialBody
from vector import Vector3D


def gravitational_force(
    pos_satellite: Vector3D, pos_body: Vector3D, mass_satellite: float, mu_body: float
) -> Vector3D:
    """
    Calcule la force gravitationnelle exercée par un corps sur le satellite

    F = -G * M * m / r^2 * r̂ = -μ * m / r^2 * r̂

    Args:
        pos_satellite: Position du satellite
        pos_body: Position du corps attracteur
        mass_satellite: Masse du satellite [kg]
        mu_body: Paramètre gravitationnel du corps (G*M) [m^3/s^2]

    Returns:
        Force gravitationnelle [N]
    """
    r_vec = pos_satellite - pos_body
    r = r_vec.norm()

    if r == 0:
        return Vector3D(0, 0, 0)

    # F = -μ * m / r^3 * r_vec (le r^3 vient de r^2 * r)
    force_magnitude = -mu_body * mass_satellite / (r**3)
    force = r_vec * force_magnitude

    return force


def total_acceleration(
    pos_satellite: Vector3D, bodies: list[CelestialBody], t: float
) -> Vector3D:
    """
    Calcule l'accélération totale du satellite due à tous les corps

    Args:
        pos_satellite: Position du satellite
        bodies: Liste des corps célestes
        t: Temps actuel [s]

    Returns:
        Accélération totale [m/s^2]
    """
    acceleration = Vector3D(0, 0, 0)
    mass_satellite = 1.0  # On calcule F/m directement

    for body in bodies:
        body_state = body.get_state_at_time(t)
        force = gravitational_force(
            pos_satellite, body_state.position, mass_satellite, body.mu
        )
        acceleration = acceleration + force  # F/m car mass=1

    return acceleration
