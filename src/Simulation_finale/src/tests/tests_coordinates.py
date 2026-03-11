import numpy as np

from src.simulation.coordinates import (
    FRAME_ECLIPTIC_J2000,
    FRAME_RLP,
    CoordinateTransformer,
)


def validate_transformations():
    """Teste la cohérence des transformations (VERSION CORRIGÉE)."""
    print("=" * 70)
    print("VALIDATION DES TRANSFORMATIONS (VERSION CORRIGÉE)")
    print("=" * 70 + "\n")

    # Test SANS éphémérides (orbite circulaire)
    print("Test 1: Transformations sans éphémérides (orbite circulaire)")
    transformer = CoordinateTransformer(include_moon=False)

    state_ecliptic = np.array([1.5e11, 0.0, 0.0, 0.0, 3e4, 0.0])
    time = 0.0

    state_rlp = transformer.ecliptic_to_rlp(state_ecliptic, time)
    state_back = transformer.rlp_to_ecliptic(state_rlp, time)

    error = np.linalg.norm(state_ecliptic - state_back)
    print(f"  Erreur aller-retour: {error:.3e} m")
    assert error < 1e-3, f"Erreur trop grande: {error}"
    print("  ✓ Test réussi\n")

    # Test normalisation CRTBP
    print("Test 3: Normalisation CRTBP")
    state_crtbp = transformer.rlp_to_crtbp(state_rlp)
    state_rlp_back = transformer.crtbp_to_rlp(state_crtbp)

    error = np.linalg.norm(state_rlp - state_rlp_back)
    print(f"  Erreur: {error:.3e} m")
    assert error < 1e-6, f"Erreur trop grande: {error}"
    print("  ✓ Test réussi\n")

    # Test 3: Position L2 dans RLP
    print("Test 3: Position de L2 dans RLP")

    # Position de la Terre dans RLP
    earth_pos_rlp = transformer.compute_earth_position_rlp()
    print(f"  Position Terre dans RLP: {earth_pos_rlp[0]/1e9:.3f} million km")

    # Position de L2 dans RLP
    l2_pos_rlp = transformer.compute_l2_position_rlp()
    print(f"  Position L2 dans RLP: {l2_pos_rlp[0]/1e9:.3f} million km")

    # Distance L2 depuis le barycentre (origine de RLP)
    l2_dist_from_barycenter = np.linalg.norm(l2_pos_rlp)
    print(
        f"  Distance L2 depuis barycentre: {l2_dist_from_barycenter/1e9:.3f} million km"
    )

    # Distance L2 depuis la Terre (ce qui nous intéresse vraiment)
    l2_dist_from_earth = np.linalg.norm(l2_pos_rlp - earth_pos_rlp)
    print(f"  Distance L2 depuis Terre: {l2_dist_from_earth/1e6:.3f} milliers de km")

    # Vérification avec la méthode directe
    l2_dist_expected = transformer.compute_l2_distance_from_earth()
    print(f"  Distance attendue (formule): {l2_dist_expected/1e6:.3f} milliers de km")

    # Test de cohérence
    error = abs(l2_dist_from_earth - l2_dist_expected)
    print(f"  Erreur: {error:.3e} m")
    assert error < 1.0, f"Erreur trop grande: {error}"

    # Vérification de l'ordre de grandeur (1.5 million km ± 10%)
    expected_range = (1.35e9, 1.65e9)  # 1.35 à 1.65 million km
    assert (
        expected_range[0] < l2_dist_from_earth < expected_range[1]
    ), f"Distance Terre-L2 hors limites: {l2_dist_from_earth/1e6:.3f} milliers de km"

    print("  ✓ Test réussi\n")

    # Test 4: Position du Soleil dans RLP
    print("Test 4: Position du Soleil dans RLP")
    sun_pos_rlp = transformer.compute_sun_position_rlp()
    sun_dist_from_barycenter = np.linalg.norm(sun_pos_rlp)
    print(f"  Distance Soleil depuis barycentre: {sun_dist_from_barycenter/1e3:.1f} km")
    print(f"  (Le barycentre est très proche du Soleil!)")

    # Vérification: doit être ≈ μ × R
    expected_sun_dist = transformer.mu_ratio * transformer.R
    error = abs(sun_dist_from_barycenter - expected_sun_dist)
    assert error < 1000, f"Position Soleil incorrecte: erreur = {error} m"
    print("  ✓ Test réussi\n")

    # Test 5: Vérification que Soleil-Terre = R
    print("Test 5: Distance Soleil-Terre dans RLP")
    sun_earth_dist = np.linalg.norm(earth_pos_rlp - sun_pos_rlp)
    print(f"  Distance Soleil-Terre: {sun_earth_dist/1e9:.6f} million km")
    print(f"  Distance attendue (1 AU): {transformer.R/1e9:.6f} million km")

    error = abs(sun_earth_dist - transformer.R)
    print(f"  Erreur: {error:.3e} m")
    assert error < 1.0, f"Distance Soleil-Terre incorrecte"
    print("  ✓ Test réussi\n")

    # Test 6: Conservation de la norme
    print("Test 6: Écliptique ↔ ECI")
    state_eci = transformer.ecliptic_to_eci(state_ecliptic)
    state_ecliptic_back = transformer.eci_to_ecliptic(state_eci)

    error = np.linalg.norm(state_ecliptic - state_ecliptic_back)
    print(f"  Erreur: {error:.3e} m")
    assert error < 1e-6, f"Erreur trop grande: {error}"
    print("  ✓ Test réussi\n")

    print("=== Tous les tests réussis ===")


if __name__ == "__main__":
    validate_transformations()

    # Exemple d'utilisation
    print("\n" + "=" * 60)
    print("Exemple d'utilisation")
    print("=" * 60 + "\n")

    transformer = CoordinateTransformer(include_moon=True)

    # Positions des corps dans RLP
    print("Positions dans le référentiel RLP:")
    print("-" * 40)

    sun_pos = transformer.compute_sun_position_rlp()
    earth_pos = transformer.compute_earth_position_rlp()
    l2_pos = transformer.compute_l2_position_rlp()

    print(f"Soleil:  x = {sun_pos[0]/1e6:8.1f} km  (très proche du barycentre)")
    print(f"Terre:   x = {earth_pos[0]/1e9:8.3f} million km")
    print(f"L2:      x = {l2_pos[0]/1e9:8.3f} million km")
    print()

    # Distances
    print("Distances:")
    print("-" * 40)
    d_sun_earth = np.linalg.norm(earth_pos - sun_pos)
    d_earth_l2 = np.linalg.norm(l2_pos - earth_pos)
    d_sun_l2 = np.linalg.norm(l2_pos - sun_pos)

    print(f"Soleil → Terre:  {d_sun_earth/1e9:.6f} million km  (= 1 AU)")
    print(f"Terre → L2:      {d_earth_l2/1e6:.3f} milliers km  (~1.5 million km)")
    print(f"Soleil → L2:     {d_sun_l2/1e9:.3f} million km")
    print()

    # État au point L2
    state_rlp = np.array([l2_pos[0], l2_pos[1], l2_pos[2], 0.0, 0.0, 0.0])

    print("État au point L2 (RLP):")
    print(
        f"  Position: [{state_rlp[0]/1e9:.3f}, {state_rlp[1]/1e9:.3f}, "
        f"{state_rlp[2]/1e9:.3f}] million km"
    )
    print(
        f"  (Distance depuis barycentre: {np.linalg.norm(state_rlp[:3])/1e9:.3f} million km)"
    )
    print(f"  (Distance depuis Terre: {d_earth_l2/1e6:.3f} milliers km)")
    print()

    # Conversion en écliptique
    time = 0.0  # J2000.0
    state_ecliptic = transformer.rlp_to_ecliptic(state_rlp, time)
    print("État en écliptique J2000:")
    print(
        f"  Position: [{state_ecliptic[0]/1e9:.3f}, {state_ecliptic[1]/1e9:.3f}, "
        f"{state_ecliptic[2]/1e9:.3f}] million km"
    )
    print()

    # Normalisation CRTBP
    state_crtbp = transformer.rlp_to_crtbp(state_rlp)
    print("État normalisé (CRTBP):")
    print(
        f"  Position: [{state_crtbp[0]:.6f}, {state_crtbp[1]:.6f}, "
        f"{state_crtbp[2]:.6f}] (adimensionnel)"
    )
    print(f"  (L2 est à x ≈ 1.01 en unités normalisées)")
    print()

    # Informations sur la transformation
    print(transformer.get_transformation_info(FRAME_ECLIPTIC_J2000, FRAME_RLP, time))
