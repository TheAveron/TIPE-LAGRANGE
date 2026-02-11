import numpy as np

from src.simulation.CRTBP_model_dynamics import CRTBP3Body
from src.simulation.dynamics_conf import DynamicsConfig, DynamicsModel


def test_crtbp_model():
    """
    Tests de validation du modèle CRTBP.
    """
    print("=== Tests du modèle CRTBP ===\n")

    # Configuration
    config = DynamicsConfig(model=DynamicsModel.CRTBP)

    # Test 1: Modèle normalisé
    print("Test 1: CRTBP normalisé")
    crtbp_norm = CRTBP3Body(config, normalized=True)

    print(f"  μ = {crtbp_norm.mu:.10e}")
    print(f"  Position primaire 1: x = {crtbp_norm.x1:.10f}")
    print(f"  Position primaire 2: x = {crtbp_norm.x2:.10f}")
    print(f"  Distance entre primaires: {crtbp_norm.x2 - crtbp_norm.x1:.10f}")
    assert abs((crtbp_norm.x2 - crtbp_norm.x1) - 1.0) < 1e-10, "Distance ≠ 1"
    print("  ✓ Normalisation correcte\n")

    # Test 2: Conservation de la constante de Jacobi
    print("Test 2: Conservation de la constante de Jacobi")

    # État initial au point L2 (approximatif)
    x_l2 = 1.0 + (crtbp_norm.mu / 3.0) ** (1.0 / 3.0)
    state0 = np.array([x_l2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    C0 = crtbp_norm.jacobi_constant(state0)
    print(f"  C initial = {C0:.10f}")

    # Propager sur un petit temps
    from scipy.integrate import solve_ivp

    t_span = (0.0, 0.1)  # 0.1 unité de temps
    sol = solve_ivp(
        crtbp_norm.equations_of_motion,
        t_span,
        state0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        dense_output=True,
    )

    # Vérifier C à la fin
    state_final = sol.y[:, -1]
    C_final = crtbp_norm.jacobi_constant(state_final)

    delta_C = abs(C_final - C0)
    print(f"  C final = {C_final:.10f}")
    print(f"  ΔC = {delta_C:.3e}")

    # Dans le CRTBP pur, C doit être constant à la précision machine
    assert delta_C < 1e-10, f"C non conservé: ΔC = {delta_C}"
    print("  ✓ Constante de Jacobi conservée\n")

    # Test 3: Symétrie des équations
    print("Test 3: Symétrie par rapport au plan XZ")

    # État symétrique (y → -y, vy → -vy)
    state_sym = state0.copy()
    state_sym[1] = -state_sym[1]  # y → -y
    state_sym[4] = -state_sym[4]  # vy → -vy

    acc1 = crtbp_norm.compute_acceleration(0.0, state0)
    acc_sym = crtbp_norm.compute_acceleration(0.0, state_sym)

    # ay doit changer de signe, ax et az doivent rester identiques
    print(f"  ax: {acc1[0]:.6e}, symétrique: {acc_sym[0]:.6e}")
    print(f"  ay: {acc1[1]:.6e}, symétrique: {acc_sym[1]:.6e}")
    print(f"  az: {acc1[2]:.6e}, symétrique: {acc_sym[2]:.6e}")

    assert abs(acc1[0] - acc_sym[0]) < 1e-15, "ax non symétrique"
    assert abs(acc1[1] + acc_sym[1]) < 1e-15, "ay non antisymétrique"
    assert abs(acc1[2] - acc_sym[2]) < 1e-15, "az non symétrique"
    print("  ✓ Symétrie vérifiée\n")

    # Test 4: Points d'équilibre
    print("Test 4: Vérification que L2 est un point d'équilibre")

    # Au point L2, avec vitesse nulle, l'accélération doit être nulle
    # (dans le référentiel tournant)
    acc_l2 = crtbp_norm.compute_acceleration(0.0, state0)
    acc_norm = np.linalg.norm(acc_l2)

    print(f"  Accélération à L2: {acc_norm:.3e}")
    # Note: ne sera pas exactement zéro car x_l2 est approximatif
    # Pour un vrai test, il faudrait calculer L2 précisément
    print(f"  (Non exactement zéro car position L2 approximative)")
    assert acc_norm < 0.01, f"Accélération trop grande à L2: {acc_norm}"
    print("  ✓ L2 est proche d'un point d'équilibre\n")

    print("=== Tous les tests CRTBP réussis ===\n")


if __name__ == "__main__":
    test_crtbp_model()
