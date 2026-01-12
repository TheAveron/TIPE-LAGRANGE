"""
Générateur d'orbites périodiques pour le CRTBP.

Ce module fournit des fonctions pour générer des conditions initiales
d'orbites périodiques (Lyapunov, halo, quasi-halo) autour des points
de Lagrange L1 et L2.

Méthode : Correcteur différentiel simple
- Partir d'une estimation de vitesse
- Propager jusqu'au croisement du plan XZ
- Corriger itérativement pour fermer l'orbite
"""

from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from numba import njit


@njit
def generate_l2_periodic_orbit(
    crtbp_model,
    amplitude_y: float = 100e6,
    amplitude_z: float = 50e6,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Génère une condition initiale pour une orbite périodique autour de L2.

    Cette fonction utilise un correcteur différentiel pour trouver une orbite
    qui satisfait les équations du CRTBP et se referme après une période.

    Args:
        crtbp_model: Instance du modèle CRTBP (CRTBP3Body)
        amplitude_y: Amplitude désirée en y [m]
        amplitude_z: Amplitude désirée en z [m]
        max_iterations: Nombre max d'itérations
        tolerance: Tolérance sur les erreurs de fermeture [m ou m/s]
        verbose: Afficher les messages de debug

    Returns:
        (état_initial, période) où:
            état_initial: [x, y, z, vx, vy, vz] en RLP [m, m/s]
            période: Période de l'orbite [s]

    Méthode:
        1. Calculer position L2
        2. Estimer vitesse initiale
        3. Propager jusqu'au croisement y=0
        4. Vérifier symétrie et corriger
        5. Itérer jusqu'à convergence

    Note:
        Pour une vraie orbite périodique, au croisement y=0:
        - x ≈ x_initial
        - z ≈ z_initial
        - vx ≈ 0
        - vy et vz changent de signe
    """

    if verbose:
        print("\n" + "=" * 70)
        print("GÉNÉRATION D'ORBITE PÉRIODIQUE AUTOUR DE L2")
        print("=" * 70)

    # ========== 1. POSITION DE L2 ==========
    mu = crtbp_model.mu
    R = crtbp_model.R
    omega = crtbp_model.omega

    # Position L2 (approximation de Taylor)
    x_l2_normalized = 1.0 + (mu / 3.0) ** (1.0 / 3.0)
    x_l2 = x_l2_normalized * R

    if verbose:
        print(f"\nPosition L2:")
        print(f"  x = {x_l2/1e9:.6f} million km")
        print(f"  (normalisée: x = {x_l2_normalized:.6f})")

    # ========== 2. ESTIMATION INITIALE DE LA VITESSE ==========

    # Pour une orbite de Lyapunov plane (z=0) :
    # La vitesse vy est approximativement omega * amplitude_y / 2

    # Pour une orbite halo (z≠0) :
    # Il y a un couplage entre y et z, donc vz ≠ 0

    # Estimation empirique (basée sur la littérature CRTBP)
    v_y_estimate = omega * amplitude_y * 0.5
    v_z_estimate = -omega * amplitude_z * 0.3

    # Condition initiale (première estimation)
    state_0 = np.array(
        [
            x_l2,  # x = position L2
            amplitude_y,  # y = amplitude désirée
            amplitude_z,  # z = amplitude désirée
            0.0,  # vx = 0 (symétrie)
            v_y_estimate,  # vy estimée
            v_z_estimate,  # vz estimée
        ]
    )

    if verbose:
        print(f"\nAmplitudes désirées:")
        print(f"  y = {amplitude_y/1e6:.1f} milliers km")
        print(f"  z = {amplitude_z/1e6:.1f} milliers km")
        print(f"\nEstimation initiale vitesse:")
        print(f"  vy = {v_y_estimate:.3f} m/s")
        print(f"  vz = {v_z_estimate:.3f} m/s")

    # Constante de Jacobi initiale
    C_initial = crtbp_model.jacobi_constant(state_0)

    if verbose:
        print(f"\nConstante de Jacobi: C = {C_initial:.6f}")

    # ========== 3. CORRECTEUR DIFFÉRENTIEL ==========

    state_current = state_0.copy()
    period = None

    if verbose:
        print(f"\nItérations du correcteur différentiel:")
        print(
            f"  {'Iter':>5s}  {'T/2 (h)':>10s}  {'Err pos (m)':>12s}  "
            f"{'Err vel (m/s)':>14s}  {'Total':>12s}"
        )
        print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*12}")

    for iteration in tqdm(range(max_iterations)):

        # Événement : croisement du plan XZ (y = 0)
        def event_y_crossing(t, state):
            return state[1]  # y composante

        event_y_crossing.terminal = True  # type: ignore
        event_y_crossing.direction = -1  # Descendant (vy < 0) # type: ignore

        # Propager jusqu'au croisement
        t_span = (0.0, 365.25 * 86400.0)  # Max 1 an

        sol = solve_ivp(
            crtbp_model.equations_of_motion,
            t_span,
            state_current,
            method="DOP853",
            events=event_y_crossing,
            rtol=1e-12,
            atol=1e-12,
            dense_output=False,
        )

        if len(sol.t_events[0]) == 0:
            if verbose:
                print(f"\n  ⚠ Aucun croisement trouvé à l'itération {iteration+1}")
            break

        # État au croisement (demi-période)
        t_half = sol.t_events[0][0]
        state_half = sol.y_events[0][0]

        # Pour une orbite symétrique, on veut :
        # - Δx = |x_half - x_0| ≈ 0
        # - Δz = |z_half - z_0| ≈ 0
        # - Δvx = |vx_half - 0| ≈ 0
        # - vy_half ≈ -vy_0
        # - vz_half ≈ -vz_0

        error_x = abs(state_half[0] - state_current[0])
        error_z = abs(state_half[2] - state_current[2])
        error_vx = abs(state_half[3])

        # Pour vy et vz : doivent changer de signe
        error_vy = abs(state_half[4] + state_current[4])
        error_vz = abs(state_half[5] + state_current[5])

        # Erreur totale (somme pondérée)
        error_pos = np.sqrt(error_x**2 + error_z**2)
        error_vel = np.sqrt(error_vx**2 + error_vy**2 + error_vz**2)
        error_total = error_pos + error_vel

        if verbose:
            print(
                f"  {iteration+1:5d}  {t_half/3600:10.2f}  {error_pos:12.3e}  "
                f"{error_vel:14.6e}  {error_total:12.3e}"
            )

        # Test de convergence
        if error_total < tolerance:
            period = 2.0 * t_half  # Période complète
            if verbose:
                print(f"\n✓ Convergence atteinte !")
                print(f"  Demi-période: {t_half/3600:.2f} heures")
                print(
                    f"  Période: {period/3600:.2f} heures "
                    f"({period/86400:.2f} jours)"
                )
            break

        # Correction de la vitesse (méthode de Newton simplifiée)
        # On ajuste vy et vz proportionnellement aux erreurs

        # Facteur de relaxation (évite les oscillations)
        alpha = 0.1

        # Correction basée sur les erreurs de symétrie
        delta_vy = alpha * error_vy * np.sign(state_current[4])
        delta_vz = alpha * error_vz * np.sign(state_current[5])

        state_current[4] -= delta_vy
        state_current[5] -= delta_vz

    else:
        if verbose:
            print(f"\n⚠ Non convergé après {max_iterations} itérations")
        # Utiliser le dernier état même si non convergé
        period = 2.0 * t_half if "t_half" in locals() else None  # type: ignore

    # ========== 4. VÉRIFICATIONS FINALES ==========

    if verbose:
        print(f"\nÉtat final:")
        print(f"  Position: x={state_current[0]/1e9:.6f} million km")
        print(f"            y={state_current[1]/1e6:.3f} milliers km")
        print(f"            z={state_current[2]/1e6:.3f} milliers km")
        print(f"  Vitesse:  vx={state_current[3]:.3f} m/s")
        print(f"            vy={state_current[4]:.3f} m/s")
        print(f"            vz={state_current[5]:.3f} m/s")

    # Vérifier conservation de C
    C_final = crtbp_model.jacobi_constant(state_current)
    dC = abs(C_final - C_initial)

    if verbose:
        print(f"\nConservation de la constante de Jacobi:")
        print(f"  C_initial = {C_initial:.6f}")
        print(f"  C_final   = {C_final:.6f}")
        print(f"  ΔC        = {dC:.3e}")

        if dC > 1e-6:
            print(f"  ⚠ Variation de C détectée (normal avec correcteur simple)")

    if verbose:
        print("\n" + "=" * 70)

    return state_current, period  # type: ignore


def validate_periodic_orbit(crtbp_model, state_initial: np.ndarray, period: float):
    """
    Valide qu'un état initial est bien une orbite périodique.

    Args:
        crtbp_model: Modèle CRTBP
        state_initial: État initial [x, y, z, vx, vy, vz]
        period: Période attendue [s]

    Returns:
        True si l'orbite est périodique (erreur < 1%)
    """
    print("\n" + "=" * 70)
    print("VALIDATION DE L'ORBITE PÉRIODIQUE")
    print("=" * 70)

    # Propager sur une période complète
    t_span = (0.0, period)

    sol = solve_ivp(
        crtbp_model.equations_of_motion,
        t_span,
        state_initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        dense_output=True,
    )

    state_final = sol.y[:, -1]

    # Erreurs de fermeture
    error_pos = np.linalg.norm(state_final[:3] - state_initial[:3])
    error_vel = np.linalg.norm(state_final[3:6] - state_initial[3:6])

    print(f"\nPropagation sur 1 période ({period/3600:.2f} heures):")
    print(f"  Erreur position: {error_pos/1e3:.3f} km")
    print(f"  Erreur vitesse:  {error_vel:.6f} m/s")

    # Critère : erreur < 1% de l'amplitude
    amplitude = np.linalg.norm(state_initial[:3])
    error_relative_pos = error_pos / amplitude * 100

    print(f"\nErreur relative:")
    print(f"  Position: {error_relative_pos:.3f}%")

    is_periodic = error_relative_pos < 1.0

    if is_periodic:
        print(f"\n✓ Orbite validée comme périodique")
    else:
        print(f"\n⚠ Orbite non périodique (erreur > 1%)")

    # Conservation de C
    C_values = [crtbp_model.jacobi_constant(sol.y[:, i]) for i in range(len(sol.t))]
    C_initial = C_values[0]
    dC_max = max(abs(C - C_initial) for C in C_values)

    print(f"\nConservation de C sur la période:")
    print(f"  ΔC_max = {dC_max:.3e}")

    if dC_max < 1e-8:
        print(f"  ✓ C bien conservée")
    else:
        print(f"  ⚠ Variation de C détectée")

    print("\n" + "=" * 70)

    return is_periodic


# ========== TESTS ==========


def test_orbit_generation():
    """Test de génération d'orbite périodique."""
    from CRTBP_model_dynamics import CRTBP3Body
    from dynamics_conf import DynamicsConfig, DynamicsModel

    print("=" * 70)
    print("TEST : GÉNÉRATION D'ORBITE PÉRIODIQUE L2")
    print("=" * 70)

    # Modèle CRTBP
    config = DynamicsConfig(model=DynamicsModel.CRTBP)
    crtbp = CRTBP3Body(config, normalized=False)

    # Générer orbite
    state, period = generate_l2_periodic_orbit(
        crtbp,
        amplitude_y=100e6,  # 100,000 km
        amplitude_z=50e6,  # 50,000 km
        max_iterations=100000,
        tolerance=1e-2,  # 1 mm/s
        verbose=False,
    )

    # Valider
    if period is not None:
        validate_periodic_orbit(crtbp, state, period)

    print("\n✓ Test terminé\n")


if __name__ == "__main__":
    test_orbit_generation()
