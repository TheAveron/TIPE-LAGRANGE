import numpy as np

from src.Simulations.calcul_pos_lagrange import (LagrangePoint,
                                                 LagrangePointCalculator,
                                                 Stability)
from src.Simulations.constants import Constants

# ========== TESTS COMPLETS DU MODULE ==========


def test_lagrange_points():
    """
    Suite de tests complète pour le module points de Lagrange.
    """
    print("=" * 70)
    print("TESTS DU MODULE POINTS DE LAGRANGE")
    print("=" * 70 + "\n")

    # ========== TEST 1: Positions des points colinéaires ==========
    print("-" * 70)
    print("TEST 1: Calcul des points colinéaires (L1, L2, L3)")
    print("-" * 70)

    # Système Soleil-Terre)

    calc = LagrangePointCalculator(Constants.MU_RATIO_SUN_EARTH)
    points = calc.compute_all_lagrange_points()

    # L1
    print("\n1.1. Point L1 (entre Soleil et Terre)")
    l1_info = points[LagrangePoint.L1]
    print(l1_info)

    # Vérifications
    x_l1 = l1_info.position[0]
    earth_pos = calc.x2

    # L1 doit être entre Soleil et Terre
    assert 0 < x_l1 < earth_pos, "L1 n'est pas entre Soleil et Terre"

    # Distance Terre-L1 attendue : ~1.5 million km
    d_earth_l1 = abs(x_l1 - earth_pos)
    print(f"Distance Terre-L1: {d_earth_l1/1e6:.3f} milliers km")
    print(f"Attendu: ~1500 milliers km")
    assert (
        1.4e9 < d_earth_l1 < 1.6e9
    ), f"Distance Terre-L1 incorrecte: {d_earth_l1/1e6:.1f} milliers km"
    print("✓ L1 position correcte\n")

    # L2
    print("1.2. Point L2 (au-delà de la Terre)")
    l2_info = points[LagrangePoint.L2]
    print(l2_info)

    # Vérifications
    x_l2 = l2_info.position[0]

    # L2 doit être au-delà de la Terre
    assert x_l2 > earth_pos, "L2 n'est pas au-delà de la Terre"

    # Distance Terre-L2 attendue : ~1.5 million km
    d_earth_l2 = abs(x_l2 - earth_pos)
    print(f"Distance Terre-L2: {d_earth_l2/1e6:.3f} milliers km")
    assert (
        1.4e9 < d_earth_l2 < 1.6e9
    ), f"Distance Terre-L2 incorrecte: {d_earth_l2/1e6:.1f} milliers km"
    print("✓ L2 position correcte\n")

    # L3
    print("1.3. Point L3 (opposé à la Terre)")
    l3_info = points[LagrangePoint.L3]
    print(l3_info)

    # Vérifications
    x_l3 = l3_info.position[0]
    sun_pos = calc.x1

    # L3 doit être du côté opposé du Soleil
    assert x_l3 < sun_pos, "L3 n'est pas du côté opposé"

    # Distance L3-Soleil ≈ 1 AU
    d_sun_l3 = abs(x_l3 - sun_pos)
    print(f"Distance Soleil-L3: {d_sun_l3/Constants.AU:.6f} AU")
    assert 0.99 < d_sun_l3 / Constants.AU < 1.01, "Distance Soleil-L3 incorrecte"
    print("✓ L3 position correcte\n")

    # ========== TEST 2: Points triangulaires ==========
    print("-" * 70)
    print("TEST 2: Calcul des points triangulaires (L4, L5)")
    print("-" * 70)

    # L4
    print("\n2.1. Point L4 (triangle équilatéral, y > 0)")
    l4_info = points[LagrangePoint.L4]
    print(l4_info)

    # Vérifications géométriques
    x_l4, y_l4 = l4_info.position[0], l4_info.position[1]

    # Distance L4-Soleil doit être = 1 AU
    d_sun_l4 = np.sqrt((x_l4 - sun_pos) ** 2 + y_l4**2)
    print(f"Distance Soleil-L4: {d_sun_l4/Constants.AU:.6f} AU")
    assert abs(d_sun_l4 / Constants.AU - 1.0) < 0.001, "L4 pas sur orbite terrestre"

    # Distance L4-Terre doit être = 1 AU
    d_earth_l4 = np.sqrt((x_l4 - earth_pos) ** 2 + y_l4**2)
    print(f"Distance Terre-L4: {d_earth_l4/Constants.AU:.6f} AU")
    assert (
        abs(d_earth_l4 / Constants.AU - 1.0) < 0.001
    ), "L4 ne forme pas triangle équilatéral"

    # Angle Soleil-Terre-L4 doit être 60°
    vec_earth_sun = np.array([sun_pos - earth_pos, 0], dtype=np.float64)
    vec_earth_l4 = np.array([x_l4 - earth_pos, y_l4], dtype=np.float64)
    cos_angle = np.dot(vec_earth_sun, vec_earth_l4) / (
        np.linalg.norm(vec_earth_sun) * np.linalg.norm(vec_earth_l4)
    )
    angle_deg = np.arccos(cos_angle) * 180 / np.pi
    print(f"Angle Soleil-Terre-L4: {angle_deg:.2f}°")
    assert abs(angle_deg - 60.0) < 0.1, "Angle incorrect pour triangle équilatéral"
    print("✓ L4 position correcte\n")

    # L5
    print("2.2. Point L5 (triangle équilatéral, y < 0)")
    l5_info = points[LagrangePoint.L5]
    print(l5_info)

    # Vérifications similaires
    x_l5, y_l5 = l5_info.position[0], l5_info.position[1]

    # L5 doit être symétrique de L4 par rapport à XZ
    assert abs(x_l5 - x_l4) < 1e-6, "L5 pas symétrique de L4 en x"
    assert abs(y_l5 + y_l4) < 1e-6, "L5 pas symétrique de L4 en y"
    print("✓ L5 symétrique de L4\n")

    # ========== TEST 3: Constante de Jacobi ==========
    print("-" * 70)
    print("TEST 3: Constante de Jacobi aux points de Lagrange")
    print("-" * 70)

    all_points = calc.compute_all_lagrange_points()

    print("\nConstantes de Jacobi:")
    for point_enum, info in all_points.items():
        print(f"  {point_enum.value}: C = {info.jacobi_constant:.6f}")

    C_l1 = all_points[LagrangePoint.L1].jacobi_constant
    C_l2 = all_points[LagrangePoint.L2].jacobi_constant
    C_l3 = all_points[LagrangePoint.L3].jacobi_constant

    print(f"\nOrdre des constantes de Jacobi:")
    print(f"  C(L1) = {C_l1:.6f}")
    print(f"  C(L2) = {C_l2:.6f}")
    print(f"  C(L3) = {C_l3:.6f}")

    assert C_l1 > C_l2, "C(L1) devrait être > C(L2)"
    assert C_l2 > C_l3, "C(L2) devrait être > C(L3)"
    print("✓ Ordre correct des constantes de Jacobi\n")

    # ========== TEST 4: Analyse de stabilité ==========
    print("-" * 70)
    print("TEST 4: Analyse de stabilité")
    print("-" * 70)

    for point_enum in [LagrangePoint.L1, LagrangePoint.L2, LagrangePoint.L4]:
        print(f"\n4.{point_enum.value[-1]}. Analyse de {point_enum.value}")
        analysis = calc.analyze_stability(point_enum)

        print(analysis["classification"])
        print(f"\nValeurs propres:")
        for i, lam in enumerate(analysis["eigenvalues"]):
            if abs(lam.imag) < 1e-10:
                print(f"  λ_{i+1} = {lam.real:+.6e} (réel)")
            else:
                print(f"  λ_{i+1} = {lam.real:+.6e} {lam.imag:+.6e}i")

        # Vérifications
        if point_enum in [LagrangePoint.L1, LagrangePoint.L2, LagrangePoint.L3]:
            assert (
                analysis["unstable_modes"] > 0
            ), f"{point_enum.value} devrait être instable"
            assert (
                analysis["neutral_modes"] > 0
            ), f"{point_enum.value} devrait avoir modes oscillatoires"
            print(f"✓ {point_enum.value} correctement identifié comme instable")
        else:  # L4 ou L5
            if calc.mu < calc.mu_critical:
                assert (
                    analysis["unstable_modes"] == 0
                ), f"{point_enum.value} devrait être stable"
                print(f"✓ {point_enum.value} correctement identifié comme stable")
            else:
                assert (
                    analysis["unstable_modes"] > 0
                ), f"{point_enum.value} devrait être instable"
                print(f"✓ {point_enum.value} correctement identifié comme instable")

    print()

    # ========== TEST 5: Système Terre-Lune ==========
    print("-" * 70)
    print("TEST 5: Points de Lagrange Terre-Lune")
    print("-" * 70)

    calc_em = LagrangePointCalculator(
        mu=Constants.MU_RATIO_EARTH_MOON,
        distance_unit=Constants.R_EARTH_MOON,
        normalized=False,
    )

    print(f"\nμ (Terre-Lune) = {Constants.MU_RATIO_EARTH_MOON:.6f}")
    print(f"μ_critique = {calc_em.mu_critical:.6f}")
    print(
        f"→ L4 et L5 sont {'STABLES' if Constants.MU_RATIO_EARTH_MOON < calc_em.mu_critical else 'INSTABLES'}\n"
    )

    l1_em = calc_em.compute_l1()
    l2_em = calc_em.compute_l2()
    l4_em = calc_em.compute_l4()

    print(f"L1 (Terre-Lune): {l1_em.distance_to_secondary/1e3:.1f} km de la Lune")  # type: ignore
    print(f"L2 (Terre-Lune): {l2_em.distance_to_secondary/1e3:.1f} km de la Lune")  # type: ignore

    # Vérifier que L1 est proche de la Lune (~ 60,000 km)
    assert 50e3 < l1_em.distance_to_secondary / 1e3 < 70e3, "L1 Terre-Lune à distance incorrecte"  # type: ignore
    print("✓ Points Terre-Lune corrects\n")

    # ========== TEST 6: Convergence Newton-Raphson ==========
    print("-" * 70)
    print("TEST 6: Convergence de Newton-Raphson")
    print("-" * 70)

    # Test avec différentes estimations initiales
    print("\n6.1. Robustesse de la convergence pour L2")

    initial_guesses = [
        1.0 + (calc.mu / 3.0) ** (1.0 / 3.0),  # Estimation théorique
        1.01,  # Proche de la Terre
        1.02,  # Plus loin
    ]

    for i, x0 in enumerate(initial_guesses):
        try:
            l2 = calc.compute_l2(initial_guess=x0)
            print(
                f"  Estimation {i+1}: x₀ = {x0:.6f} → x_L2 = {l2.position[0]/Constants.AU:.6f} AU ✓"
            )
        except RuntimeError as e:
            print(f"  Estimation {i+1}: x₀ = {x0:.6f} → ÉCHEC: {e}")
            raise

    print("✓ Convergence robuste\n")

    # ========== TEST 7: Propriétés géométriques ==========
    print("-" * 70)
    print("TEST 7: Vérification des propriétés géométriques")
    print("-" * 70)

    # Distance Soleil-Terre
    d_sun_earth = abs(earth_pos - sun_pos)
    print(f"\n7.1. Distance Soleil-Terre: {d_sun_earth/Constants.AU:.6f} AU")
    assert (
        abs(d_sun_earth / Constants.AU - 1.0) < 1e-10
    ), "Distance normalisée incorrecte"
    print("✓ Normalisation correcte")

    # Symétrie L1-L2
    d_l1 = abs(l1_info.distance_to_secondary)  # type: ignore
    d_l2 = abs(l2_info.distance_to_secondary)  # type: ignore
    ratio = d_l1 / d_l2
    print(f"\n7.2. Distances Terre-L1 / Terre-L2: {ratio:.6f}")
    print(f"     (proche de 1.0 pour μ << 1)")
    assert (
        abs(ratio - 1.0) < 0.01
    ), "L1 et L2 devraient être quasi-symétriques pour μ << 1"
    print("✓ Symétrie approximative vérifiée\n")

    # ========== RÉSUMÉ ==========
    print("=" * 70)
    print("RÉSUMÉ DES TESTS")
    print("=" * 70)
    print("✓ Test 1: Positions points colinéaires - OK")
    print("✓ Test 2: Positions points triangulaires - OK")
    print("✓ Test 3: Constantes de Jacobi - OK")
    print("✓ Test 4: Analyse de stabilité - OK")
    print("✓ Test 5: Système Terre-Lune - OK")
    print("✓ Test 6: Convergence Newton-Raphson - OK")
    print("✓ Test 7: Propriétés géométriques - OK")
    print("\nTous les tests du module points de Lagrange sont réussis avec succès !\n")
