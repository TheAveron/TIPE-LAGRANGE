from celestial_body import CelestialBody
from forces import total_acceleration
from state import State


def derivative(state: State, t: float, bodies: list[CelestialBody]) -> State:
    """
    Calcule la dérivée de l'état : dX/dt = (v, a)

    Args:
        state: État actuel (position, vitesse)
        t: Temps actuel [s]
        bodies: Liste des corps célestes

    Returns:
        Dérivée de l'état (vitesse, accélération)
    """
    acceleration = total_acceleration(state.position, bodies, t)
    return State(position=state.velocity, velocity=acceleration)


def rk4_step(state: State, t: float, dt: float, bodies: list[CelestialBody]) -> State:
    """
    Effectue un pas d'intégration Runge-Kutta d'ordre 4

    Args:
        state: État actuel
        t: Temps actuel [s]
        dt: Pas de temps [s]
        bodies: Liste des corps célestes

    Returns:
        Nouvel état après le pas de temps
    """
    # k1 = f(t, y)
    k1 = derivative(state, t, bodies)

    # k2 = f(t + dt/2, y + dt/2 * k1)
    state2 = State(
        position=state.position + k1.position * (dt / 2),
        velocity=state.velocity + k1.velocity * (dt / 2),
    )
    k2 = derivative(state2, t + dt / 2, bodies)

    # k3 = f(t + dt/2, y + dt/2 * k2)
    state3 = State(
        position=state.position + k2.position * (dt / 2),
        velocity=state.velocity + k2.velocity * (dt / 2),
    )
    k3 = derivative(state3, t + dt / 2, bodies)

    # k4 = f(t + dt, y + dt * k3)
    state4 = State(
        position=state.position + k3.position * dt,
        velocity=state.velocity + k3.velocity * dt,
    )
    k4 = derivative(state4, t + dt, bodies)

    # y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    new_position = state.position + (
        k1.position + k2.position * 2 + k3.position * 2 + k4.position
    ) * (dt / 6)
    new_velocity = state.velocity + (
        k1.velocity + k2.velocity * 2 + k3.velocity * 2 + k4.velocity
    ) * (dt / 6)

    return State(position=new_position, velocity=new_velocity)
