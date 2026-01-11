from typing import List, Tuple

import numpy as np
from forces import total_acceleration
from integrator import rk4_step
from simulator import CelestialBody
from state import State
from vector import Vector3D


def compute_state_transition_matrix(
    state: State, t: float, dt: float, bodies: List[CelestialBody]
) -> np.ndarray:
    """
    Calcule la matrice de transition d'état Φ(t+dt, t) par intégration numérique

    La matrice de transition satisfait : dΦ/dt = A(t) * Φ
    où A(t) est la matrice jacobienne du système

    Args:
        state: État nominal au temps t
        t: Temps initial [s]
        dt: Intervalle de temps [s]
        bodies: Liste des corps célestes

    Returns:
        Matrice de transition 6x6
    """
    # Initialisation : Φ(t,t) = I (identité)
    phi = np.eye(6)

    # Nombre de sous-pas pour l'intégration
    n_steps = 10
    sub_dt = dt / n_steps

    state_nom = state.copy()

    for _ in range(n_steps):
        # Calcul de la jacobienne A(t) au point nominal
        A = compute_jacobian(state_nom, t, bodies)

        # Intégration de dΦ/dt = A * Φ avec RK4
        k1 = A @ phi
        k2 = A @ (phi + sub_dt / 2 * k1)
        k3 = A @ (phi + sub_dt / 2 * k2)
        k4 = A @ (phi + sub_dt * k3)

        phi = phi + sub_dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Propagation de l'état nominal
        state_nom = rk4_step(state_nom, t, sub_dt, bodies)
        t += sub_dt

    return phi


def compute_jacobian(state: State, t: float, bodies: List[CelestialBody]) -> np.ndarray:
    """
    Calcule la matrice jacobienne A = ∂f/∂x du système dynamique

    Pour le système dx/dt = f(x,t) où x = [r, v]^T

    Args:
        state: État actuel
        t: Temps [s]
        bodies: Liste des corps célestes

    Returns:
        Matrice jacobienne 6x6
    """
    A = np.zeros((6, 6))

    # Bloc supérieur : ∂(dr/dt)/∂r = 0, ∂(dr/dt)/∂v = I
    A[0:3, 3:6] = np.eye(3)

    # Bloc inférieur : ∂(dv/dt)/∂r = ∂a/∂r, ∂(dv/dt)/∂v = 0
    # Calcul de ∂a/∂r (dérivées secondes du potentiel gravitationnel)

    r = state.position
    epsilon = 1e-6  # Perturbation pour différences finies

    # Approximation par différences finies
    for i in range(3):
        # Perturbation positive
        r_plus = Vector3D(r.x, r.y, r.z)
        if i == 0:
            r_plus.x += epsilon
        elif i == 1:
            r_plus.y += epsilon
        else:
            r_plus.z += epsilon

        # Perturbation négative
        r_minus = Vector3D(r.x, r.y, r.z)
        if i == 0:
            r_minus.x -= epsilon
        elif i == 1:
            r_minus.y -= epsilon
        else:
            r_minus.z -= epsilon

        # Accélérations aux positions perturbées
        a_plus = total_acceleration(r_plus, bodies, t)
        a_minus = total_acceleration(r_minus, bodies, t)

        # Dérivée partielle
        A[3, i] = (a_plus.x - a_minus.x) / (2 * epsilon)
        A[4, i] = (a_plus.y - a_minus.y) / (2 * epsilon)
        A[5, i] = (a_plus.z - a_minus.z) / (2 * epsilon)

    return A


def compute_monodromy_matrix(
    initial_state: State,
    t_start: float,
    period: float,
    bodies: List[CelestialBody],
    n_steps: int = 100,
) -> np.ndarray:
    """
    Calcule la matrice de monodromie M = Φ(t+T, t) sur une période complète

    La matrice de monodromie caractérise la stabilité de l'orbite périodique

    Args:
        initial_state: État initial sur l'orbite
        t_start: Temps initial [s]
        period: Période de l'orbite [s]
        bodies: Liste des corps célestes
        n_steps: Nombre de pas d'intégration

    Returns:
        Matrice de monodromie 6x6
    """
    dt = period / n_steps
    phi_total = np.eye(6)

    state = initial_state.copy()
    t = t_start

    for _ in range(n_steps):
        # Calcul de Φ pour ce pas de temps
        phi_step = compute_state_transition_matrix(state, t, dt, bodies)

        # Composition : Φ(t+nΔt, t) = Φ(t+nΔt, t+(n-1)Δt) * ... * Φ(t+Δt, t)
        phi_total = phi_step @ phi_total

        # Propagation de l'état
        state = rk4_step(state, t, dt, bodies)
        t += dt

    return phi_total


def find_stable_eigenvector(monodromy_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Trouve le vecteur propre stable de la matrice de monodromie

    Le vecteur propre stable correspond à la valeur propre λ avec |λ| < 1
    Pour les orbites autour de L2, il y a une valeur propre réelle stable

    Args:
        monodromy_matrix: Matrice de monodromie 6x6

    Returns:
        Tuple (vecteur propre stable normalisé, valeur propre stable)
    """
    # Calcul des valeurs et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eig(monodromy_matrix)

    # Recherche de la valeur propre stable (|λ| < 1, partie réelle)
    stable_idx = None
    stable_eigenvalue = None

    for i, lam in enumerate(eigenvalues):
        # On cherche une valeur propre réelle avec |λ| < 1
        if np.isreal(lam) and abs(lam) < 1.0:
            if stable_idx is None or abs(lam) < abs(stable_eigenvalue):
                stable_idx = i
                stable_eigenvalue = lam

    if stable_idx is None or stable_eigenvalue is None:
        # Si aucune valeur propre stable n'est trouvée, prendre celle avec
        # la plus petite valeur absolue
        print(
            "Attention : Aucune valeur propre stable trouvée. "
            "Utilisation de la valeur propre minimale en valeur absolue."
        )
        stable_idx = np.argmin(np.abs(eigenvalues))
        stable_eigenvalue = eigenvalues[stable_idx]

    # Vecteur propre correspondant (prendre la partie réelle)
    eigenvector = np.real(eigenvectors[:, stable_idx])

    # Normalisation
    eigenvector = eigenvector / np.linalg.norm(eigenvector)

    return eigenvector, float(np.real(stable_eigenvalue))
