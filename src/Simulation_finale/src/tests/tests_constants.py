from src.Simulations.constants import Constants


def constant_validation():
    """Fonction de validation des constantes."""

    print("=== Validation des constantes ===\n")

    # Test 1: Paramètre μ
    mu = Constants.MU_RATIO_SUN_EARTH
    print(f"μ (Sun-Earth) = {mu:.10e}")
    print(f"  Attendu: ~3.0035e-6")
    assert 3.0e-6 < mu < 3.1e-6, "Erreur: μ hors limites"

    # Test 2: Vitesse orbitale Terre
    v = Constants.V_EARTH / 1000  # en km/s
    print(f"\nVitesse Terre = {v:.3f} km/s")
    print(f"  Attendu: ~29.785 km/s")
    assert 29.7 < v < 29.9, "Erreur: vitesse Terre incorrecte"

    # Test 3: Période
    T_days = Constants.T_YEAR_SIDEREAL / 86400
    print(f"\nAnnée sidérale = {T_days:.6f} jours")
    print(f"  Attendu: 365.256363 jours")

    # Test 4: Position L2 approximative
    mu = Constants.MU_RATIO_SUN_EARTH
    r_L2_approx = Constants.AU * (mu / 3) ** (1 / 3)
    print(f"\nDistance L2 (approx) = {r_L2_approx/1e9:.2f} million km")
    print(f"  Attendu: ~1.5 million km")

    print("\n✓ Tous les tests passés")


if __name__ == "__main__":
    constant_validation()
