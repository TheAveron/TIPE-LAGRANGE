"""
Installation requise:
    pip install spiceypy

Kernels SPICE nécessaires (à télécharger depuis NASA NAIF):
    - de440.bsp : éphémérides planétaires DE440
    - naif0012.tls : leap seconds
    - pck00010.tpc : constantes physiques planétaires
"""

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import spiceypy as spice

from .constants import Constants


class EphemerisManager:
    """
    Gestionnaire des éphémérides SPICE pour le modèle haute-fidélité.

    Responsabilités:
        - Charger les kernels SPICE
        - Fournir les positions/vitesses des corps célestes
        - Gérer la conversion des temps
        - Cache pour optimiser les performances

    Kernels requis:
        - LSK (Leap Second Kernel): naif0012.tls
        - SPK (Spacecraft/Planet Kernel): de440.bsp ou de441.bsp
        - PCK (Physical Constants Kernel): pck00010.tpc
    """

    def __init__(self, kernel_dir: Optional[Path] = None):
        """
        Initialise le gestionnaire d'éphémérides.

        Args:
            kernel_dir: Répertoire contenant les kernels SPICE.
                       Si None, cherche dans ./data/spice/

        Raises:
            FileNotFoundError: Si les kernels ne sont pas trouvés
        """
        if kernel_dir is None:
            kernel_dir = Path(__file__).parent.parent.parent / "data" / "spice"

        self.kernel_dir = Path(kernel_dir)
        self.loaded_kernels: List[str] = []

        # Cache pour les positions (optimisation)
        self._position_cache = {}
        self._cache_tolerance = 1.0  # Tolérance en secondes

        # Charger les kernels
        self._load_kernels()

    def _load_kernels(self):
        """
        Charge les kernels SPICE nécessaires.

        Ordre de chargement important:
        1. LSK (temps)
        2. PCK (constantes)
        3. SPK (éphémérides)
        """
        kernel_files = {
            "lsk": "naif0012.tls",  # Leap seconds
            "pck": "pck00010.tpc",  # Physical constants
            "spk": "de440.bsp",  # Éphémérides DE440
        }

        for kernel_type, filename in kernel_files.items():
            kernel_path = self.kernel_dir / filename

            if not kernel_path.exists():
                # Essayer de trouver une version alternative
                alternatives = list(
                    self.kernel_dir.glob(f"{kernel_type}*.{filename.split('.')[-1]}")
                )

                if alternatives:
                    kernel_path = alternatives[0]
                    warnings.warn(
                        f"Kernel {filename} non trouvé, utilisation de {kernel_path.name}"
                    )
                else:
                    raise FileNotFoundError(
                        f"Kernel SPICE non trouvé: {kernel_path}\n"
                        f"Téléchargez depuis: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
                    )

            try:
                spice.furnsh(str(kernel_path))
                self.loaded_kernels.append(str(kernel_path))
                print(f"  ✓ Kernel chargé: {kernel_path.name}")
            except Exception as e:
                raise RuntimeError(f"Erreur lors du chargement de {kernel_path}: {e}")

    def __del__(self):
        """Décharge les kernels lors de la destruction."""
        self.unload_kernels()

    def unload_kernels(self):
        """Décharge tous les kernels SPICE."""
        for kernel in self.loaded_kernels:
            try:
                spice.unload(kernel)
            except:
                pass
        self.loaded_kernels.clear()
        self._position_cache.clear()

    def et_from_j2000(self, seconds_since_j2000: float) -> float:
        """
        Convertit le temps depuis J2000.0 vers ET (Ephemeris Time).

        Args:
            seconds_since_j2000: Secondes depuis J2000.0 (2000-01-01 12:00:00 TDB)

        Returns:
            ET (Ephemeris Time) utilisé par SPICE

        Note:
            J2000.0 en SPICE = 0.0 ET (epoch de référence)
        """
        # Dans SPICE, ET=0 correspond exactement à J2000.0
        return seconds_since_j2000

    def get_body_state(
        self,
        body: str,
        time_et: float,
        observer: str = "SSB",
        reference_frame: str = "J2000",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtient la position et vitesse d'un corps céleste.

        Args:
            body: Nom du corps (ex: "SUN", "EARTH", "MOON", "MARS")
            time_et: Temps en ET [s]
            observer: Observateur (défaut: "SSB" = barycentre système solaire)
            reference_frame: Référentiel (défaut: "J2000" = écliptique J2000)

        Returns:
            (position [m], vitesse [m/s]) dans le référentiel spécifié

        Codes SPICE standards:
            - SSB: Solar System Barycenter
            - SUN: Soleil
            - EARTH: Terre
            - MOON: Lune
            - EARTH BARYCENTER: Barycentre Terre-Lune
            - MARS BARYCENTER, JUPITER BARYCENTER, etc.
        """
        # Vérifier le cache
        cache_key = (body, time_et, observer, reference_frame)
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]

        try:
            # spkezr retourne [x, y, z, vx, vy, vz] en km et km/s
            # Le dernier argument est le temps de lumière (ignoré ici)
            state_km, _ = spice.spkezr(body, time_et, reference_frame, "NONE", observer)

            # Conversion km → m et km/s → m/s
            position = state_km[:3] * 1000.0  # km → m # type: ignore
            velocity = state_km[3:6] * 1000.0  # km/s → m/s # type: ignore

            # Mise en cache
            self._position_cache[cache_key] = (position, velocity)

            return position, velocity

        except Exception as e:
            raise RuntimeError(f"Erreur SPICE pour {body} à t={time_et}: {e}")

    def get_earth_moon_barycenter(
        self, time_et: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtient la position du barycentre Terre-Lune.

        Args:
            time_et: Temps en ET [s]

        Returns:
            (position [m], vitesse [m/s]) du barycentre Terre-Lune
        """
        return self.get_body_state("EARTH BARYCENTER", time_et, "SSB", "J2000")

    def clear_cache(self):
        """Vide le cache des positions."""
        self._position_cache.clear()


# ========== TESTS DU GESTIONNAIRE D'ÉPHÉMÉRIDES ==========


def test_ephemeris_manager():
    """
    Teste le gestionnaire d'éphémérides.

    Note: Nécessite les kernels SPICE installés.
    """
    print("=== Tests du gestionnaire d'éphémérides ===\n")

    try:
        ephem = EphemerisManager()
    except FileNotFoundError as e:
        print(f"⚠ Kernels SPICE non trouvés: {e}")
        print(
            "  Téléchargez-les depuis https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        )
        print("  Test ignoré.\n")
        return

    # Test 1: Position de la Terre à J2000.0
    print("Test 1: Position de la Terre à J2000.0")

    t_j2000 = 0.0  # J2000.0
    pos_earth, vel_earth = ephem.get_body_state("EARTH", t_j2000, "SSB", "J2000")

    print(
        f"  Position Terre: [{pos_earth[0]/1e11:.6f}, {pos_earth[1]/1e11:.6f}, "
        f"{pos_earth[2]/1e11:.6f}] × 10¹¹ m"
    )
    print(
        f"  Vitesse Terre:  [{vel_earth[0]/1e3:.3f}, {vel_earth[1]/1e3:.3f}, "
        f"{vel_earth[2]/1e3:.3f}] km/s"
    )

    # Distance au Soleil
    r_earth = np.linalg.norm(pos_earth)
    print(f"  Distance au SSB: {r_earth/Constants.AU:.6f} AU")

    # Devrait être proche de 1 AU
    assert 0.98 < r_earth / Constants.AU < 1.02, "Distance Terre-Soleil incorrecte"
    print("  ✓ Position cohérente\n")

    # Test 2: Position de la Lune
    print("Test 2: Distance Terre-Lune")

    pos_moon, vel_moon = ephem.get_body_state("MOON", t_j2000, "EARTH", "J2000")
    r_moon = np.linalg.norm(pos_moon)

    print(f"  Distance Terre-Lune: {r_moon/1e6:.1f} milliers de km")
    print(f"  Distance attendue: {Constants.R_EARTH_MOON/1e6:.1f} milliers de km")

    # Devrait être autour de 384,400 km (±50,000 km à cause de l'excentricité)
    assert 330e6 < r_moon < 410e6, "Distance Terre-Lune incorrecte"
    print("  ✓ Distance cohérente\n")

    # Test 3: Barycentre Terre-Lune
    print("Test 3: Barycentre Terre-Lune")

    pos_emb, vel_emb = ephem.get_earth_moon_barycenter(t_j2000)
    pos_earth_ssb, _ = ephem.get_body_state("EARTH", t_j2000, "SSB", "J2000")

    # Distance Terre - Barycentre
    d_earth_emb = np.linalg.norm(pos_earth_ssb - pos_emb)

    print(f"  Distance Terre - Barycentre TL: {d_earth_emb/1e3:.1f} km")

    # Devrait être environ 4,670 km (rayon orbite lunaire × μ_Moon)
    expected = Constants.R_EARTH_MOON * Constants.MU_RATIO_EARTH_MOON
    print(f"  Distance attendue: {expected/1e3:.1f} km")

    assert abs(d_earth_emb - expected) / expected < 0.1, "Barycentre TL incorrect"
    print("  ✓ Barycentre correct\n")

    # Test 4: Évolution temporelle
    print("Test 4: Évolution sur une année")

    times = np.linspace(0, Constants.T_YEAR_SIDEREAL, 13)  # 13 points sur 1 an
    distances = []

    for t in times:
        pos, _ = ephem.get_body_state("EARTH", t, "SSB", "J2000")
        distances.append(np.linalg.norm(pos))

    distances = np.array(distances) / Constants.AU

    print(f"  Distance min: {np.min(distances):.6f} AU (périhélie)")
    print(f"  Distance max: {np.max(distances):.6f} AU (aphélie)")
    print(f"  Écart: {(np.max(distances) - np.min(distances)):.6f} AU")

    # Avec excentricité e=0.0167, écart attendu ≈ 2e ≈ 0.0334 AU
    expected_diff = 2 * Constants.E_EARTH
    actual_diff = np.max(distances) - np.min(distances)

    print(f"  Écart attendu (2e): {expected_diff:.6f} AU")
    assert (
        abs(actual_diff - expected_diff) / expected_diff < 0.3
    ), f"Excentricité incorrecte: {abs(actual_diff - expected_diff) / expected_diff}"
    print("  ✓ Excentricité cohérente\n")

    # Nettoyage
    ephem.unload_kernels()

    print("=== Tous les tests éphémérides réussis ===\n")


if __name__ == "__main__":
    test_ephemeris_manager()
