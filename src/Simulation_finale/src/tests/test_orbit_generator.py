"""
Tests et exemples pour le module initial_orbits.py

Ce script teste et valide la génération d'orbites initiales
pour JWST autour de L2.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.simulation.constants import Constants, JWSTParameters
from src.simulation.lagrange_points import LagrangePoint
from src.simulation.orbit_generator import (OrbitGenerator,
                                            OrbitInitialConditions, OrbitType,
                                            print_orbit_summary)


def test_1_jwst_nominal_orbit():
    """Test 1: Génération orbite nominale JWST."""
    print("\n" + "=" * 70)
    print(" TEST 1: Orbite Nominale JWST")
    print("=" * 70)

    gen = OrbitGenerator(lagrange_point=LagrangePoint.L2)

    orbit = gen.generate_jwst_nominal_orbit()

    print_orbit_summary(orbit)

    # Vérifications
    assert orbit.orbit_type == OrbitType.QUASI_HALO
    assert orbit.lagrange_point == LagrangePoint.L2

    # Amplitudes proches des cibles
    if orbit.amplitudes:
        y_target = JWSTParameters.ORBIT_AMPLITUDE_Y
        z_target = JWSTParameters.ORBIT_AMPLITUDE_Z

        y_achieved = orbit.amplitudes.get("y", 0)
        z_achieved = orbit.amplitudes.get("z", 0)

        print(f"\nVérification amplitudes:")
        print(f"  Y: cible={y_target/1e6:.0f} km, atteint={y_achieved/1e6:.0f} km")
        print(f"  Z: cible={z_target/1e6:.0f} km, atteint={z_achieved/1e6:.0f} km")

        # Tolérance 20%
        assert abs(y_achieved - y_target) / y_target < 0.2
        assert abs(z_achieved - z_target) / z_target < 0.2

    print("\n✓ Test RÉUSSI")
    return orbit


def test_2_linear_method():
    """Test 2: Méthode linéaire."""
    print("\n" + "=" * 70)
    print(" TEST 2: Méthode Perturbation Linéaire")
    print("=" * 70)

    gen = OrbitGenerator(lagrange_point=LagrangePoint.L2)

    orbit = gen.generate_quasi_halo_linear(
        target_amplitude_y=750000e3,  # 750,000 km
        target_amplitude_z=420000e3,  # 420,000 km
    )

    print_orbit_summary(orbit)

    # Vérifications
    assert orbit.orbit_type == OrbitType.QUASI_HALO
    assert orbit.generation_method == "linear_perturbation"
    assert not orbit.is_physical  # Doit être normalisé

    # Convertir en physique
    orbit_phys = orbit.to_physical()
    assert orbit_phys.is_physical

    print("\n✓ Test RÉUSSI")
    return orbit


def test_3_lissajous():
    """Test 3: Génération orbite Lissajous."""
    print("\n" + "=" * 70)
    print(" TEST 3: Orbite de Lissajous")
    print("=" * 70)

    gen = OrbitGenerator(lagrange_point=LagrangePoint.L2)

    # Lissajous avec phase π/2 (proche halo)
    orbit = gen.generate_lissajous(
        amplitude_y=800000e3, amplitude_z=400000e3, phase_y=0.0, phase_z=np.pi / 2
    )

    print_orbit_summary(orbit)

    # Vérifications
    assert orbit.orbit_type == OrbitType.LISSAJOUS
    assert orbit.amplitudes is not None

    print("\n✓ Test RÉUSSI")
    return orbit


def test_4_orbit_validation():
    """Test 4: Validation d'une orbite."""
    print("\n" + "=" * 70)
    print(" TEST 4: Validation Orbite (Intégration 180 jours)")
    print("=" * 70)

    gen = OrbitGenerator(lagrange_point=LagrangePoint.L2)

    # Générer orbite
    orbit = gen.generate_quasi_halo_linear(
        target_amplitude_y=750000e3, target_amplitude_z=420000e3
    )

    # Valider par intégration
    print("\nIntégration en cours...")
    results = gen.validate_orbit(orbit, duration=180 * 86400.0)

    print(f"\nRésultats validation:")
    print(f"  Amplitudes atteintes:")
    for axis, amp in results["amplitudes_achieved"].items():
        print(f"    {axis.upper()}: {amp/1e6:.3f} millions km")

    print(f"  Variation Jacobi: ΔC = {results['jacobi_variation']:.6e}")
    print(f"  Orbite stable: {results['is_stable']}")

    # Vérifications
    assert results["is_stable"], "Orbite devrait être stable"
    assert results["jacobi_variation"] < 0.01, "Jacobi devrait être presque constant"

    print("\n✓ Test RÉUSSI")
    return results


def test_5_multiple_amplitudes():
    """Test 5: Génération pour différentes amplitudes."""
    print("\n" + "=" * 70)
    print(" TEST 5: Génération Multiple Amplitudes")
    print("=" * 70)

    gen = OrbitGenerator(lagrange_point=LagrangePoint.L2)

    # Gamme d'amplitudes
    amplitudes_y = [600000e3, 750000e3, 900000e3]  # km
    amplitudes_z = [350000e3, 420000e3, 500000e3]

    print(f"\nGénération de {len(amplitudes_y)} orbites...")

    orbits = []
    for i, (ay, az) in enumerate(zip(amplitudes_y, amplitudes_z)):
        print(f"\n  Orbite {i+1}/{len(amplitudes_y)}:")
        print(f"    Cible Y={ay/1e6:.0f} km, Z={az/1e6:.0f} km")

        orbit = gen.generate_quasi_halo_linear(ay, az)
        orbits.append(orbit)

        # Vérifier Jacobi constant décroît avec amplitude
        print(f"    Jacobi C = {orbit.jacobi_constant:.8f}")

    # Vérifier: Jacobi doit décroître avec amplitude
    C_values = [o.jacobi_constant for o in orbits]
    print(f"\nConstantes de Jacobi:")
    for i, C in enumerate(C_values):
        print(f"  Orbite {i+1}: C = {C:.8f}")

    # Plus grande amplitude → plus petit C
    # (plus d'énergie)
    assert C_values[0] > C_values[-1], "C devrait décroître avec amplitude"

    print("\n✓ Test RÉUSSI")
    return orbits


def test_6_unit_conversions():
    """Test 6: Conversions d'unités."""
    print("\n" + "=" * 70)
    print(" TEST 6: Conversions Normalisé ↔ Physique")
    print("=" * 70)

    gen = OrbitGenerator(lagrange_point=LagrangePoint.L2)

    # Générer orbite normalisée
    orbit_norm = gen.generate_quasi_halo_linear(750000e3, 420000e3)

    print(f"\nOrbite normalisée:")
    print(f"  Position: {orbit_norm.state[:3]}")
    print(f"  is_physical: {orbit_norm.is_physical}")

    # Convertir en physique
    orbit_phys = orbit_norm.to_physical()

    print(f"\nOrbite physique:")
    print(f"  Position: {orbit_phys.state[:3]/1e9} millions km")
    print(f"  is_physical: {orbit_phys.is_physical}")

    # Reconvertir en normalisé
    orbit_norm_2 = orbit_phys.to_normalized()

    # Vérifier aller-retour
    diff_pos = np.linalg.norm(orbit_norm.state[:3] - orbit_norm_2.state[:3])
    diff_vel = np.linalg.norm(orbit_norm.state[3:6] - orbit_norm_2.state[3:6])

    print(f"\nVérification aller-retour:")
    print(f"  Erreur position: {diff_pos:.3e}")
    print(f"  Erreur vitesse: {diff_vel:.3e}")

    assert diff_pos < 1e-10, "Conversion position incorrecte"
    assert diff_vel < 1e-10, "Conversion vitesse incorrecte"

    print("\n✓ Test RÉUSSI")


def example_complete_workflow():
    """Exemple complet: workflow de génération."""
    print("\n" + "=" * 70)
    print(" EXEMPLE: Workflow Complet Génération Orbite JWST")
    print("=" * 70)

    # 1. Créer générateur
    print("\n1. Création du générateur...")
    gen = OrbitGenerator(lagrange_point=LagrangePoint.L2)

    # 2. Générer orbite nominale JWST
    print("\n2. Génération orbite nominale JWST...")
    orbit = gen.generate_jwst_nominal_orbit()
    print_orbit_summary(orbit)

    # 3. Convertir en unités physiques
    print("\n3. Conversion en unités physiques...")
    orbit_phys = orbit.to_physical()

    print(f"\nÉtat initial (km, m/s):")
    print(f"  x  = {orbit_phys.state[0]/1e6:.3f} millions km")
    print(f"  y  = {orbit_phys.state[1]/1e6:.3f} millions km")
    print(f"  z  = {orbit_phys.state[2]/1e6:.3f} millions km")
    print(f"  vx = {orbit_phys.state[3]:.6f} m/s")
    print(f"  vy = {orbit_phys.state[4]:.6f} m/s")
    print(f"  vz = {orbit_phys.state[5]:.6f} m/s")

    # 4. Valider orbite
    print("\n4. Validation par intégration (30 jours)...")
    results = gen.validate_orbit(orbit, duration=30 * 86400.0)

    print(f"\nRésultats:")
    print(f"  Stable: {results['is_stable']}")
    print(f"  ΔC: {results['jacobi_variation']:.3e}")

    # 5. Informations pour utilisation
    print("\n5. Prêt pour utilisation!")
    print(f"\n  Cet état initial peut être utilisé pour:")
    print(f"    - Simulation propagation orbite")
    print(f"    - Calcul manœuvres station-keeping")
    print(f"    - Analyse Monte Carlo")
    print(f"    - Tests de navigation")

    return orbit_phys


def run_all_tests():
    """Exécute tous les tests."""
    print("\n" + "=" * 70)
    print(" SUITE DE TESTS - MODULE INITIAL_ORBITS")
    print("=" * 70)

    tests = [
        ("Orbite nominale JWST", test_1_jwst_nominal_orbit),
        ("Méthode linéaire", test_2_linear_method),
        ("Lissajous", test_3_lissajous),
        ("Validation orbite", test_4_orbit_validation),
        ("Multiple amplitudes", test_5_multiple_amplitudes),
        ("Conversions unités", test_6_unit_conversions),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except AssertionError as e:
            results.append((name, False, str(e)))
            print(f"\n✗ Test ÉCHOUÉ: {e}")
        except Exception as e:
            results.append((name, False, f"Erreur: {e}"))
            print(f"\n✗ Erreur inattendue: {e}")

    # Résumé
    print("\n" + "=" * 70)
    print(" RÉSUMÉ DES TESTS")
    print("=" * 70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {name}")
        if error:
            print(f"         → {error}")

    print(f"\n{passed}/{total} tests réussis ({100*passed//total}%)")

    if passed == total:
        print("\n🎉 TOUS LES TESTS RÉUSSIS !")

    return passed == total


if __name__ == "__main__":
    # Exécuter tests
    all_passed = run_all_tests()

    # Exemple workflow
    example_complete_workflow()

    sys.exit(0 if all_passed else 1)
