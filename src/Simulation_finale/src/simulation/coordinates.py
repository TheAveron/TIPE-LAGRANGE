"""
Module de gestion des systèmes de coordonnées pour la simulation JWST L2.

Référentiels implémentés:
1. Écliptique J2000.0 (inertiel)
2. Rotating Libration Point (RLP) - Synodique Soleil-Terre
3. CRTBP normalisé (pour analyse théorique)
4. Earth-Centered Inertial (ECI) - J2000.0

Conventions:
- Tous les états sont des vecteurs [x, y, z, vx, vy, vz] en unités SI
- Les rotations suivent la règle de la main droite
- Les temps sont en secondes depuis J2000.0 (JD 2451545.0)

Approximations et leurs impacts:
- Orbite terrestre circulaire dans RLP: erreur < 2.5 million km
- Néglige nutation (< 20 arcsec): erreur < 200 km à L2
- Précession linéaire: erreur < 1 km/an
"""

from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from .constants import Constants
from .Ephem_handler import EphemerisManager
from .vectors import (PositionVector, StateVector, VelocityVector,
                      create_postion_vector, create_state_vector,
                      create_velocity_vector)


@dataclass
class ReferenceFrame:
    """
    Classe de données décrivant un référentiel.

    Attributs:
        name: Nom du référentiel
        origin: Description de l'origine
        is_inertial: True si référentiel inertiel (non-rotatif)
        length_unit: Unité de longueur (m pour SI, AU pour normalisé, etc.)
        time_unit: Unité de temps (s pour SI, TU pour normalisé)
    """

    name: str
    origin: str
    is_inertial: bool
    length_unit: str = "m"
    time_unit: str = "s"

    def __str__(self) -> str:
        return f"{self.name} (origine: {self.origin}, inertiel: {self.is_inertial})"


# ========== DÉFINITIONS DES RÉFÉRENTIELS ==========

FRAME_ECLIPTIC_J2000 = ReferenceFrame(
    name="Ecliptic J2000.0",
    origin="Barycentre du système solaire",
    is_inertial=True,
    length_unit="m",
    time_unit="s",
)

FRAME_RLP = ReferenceFrame(
    name="Rotating Libration Point (RLP)",
    origin="Barycentre Soleil-Terre(-Lune)",
    is_inertial=False,
    length_unit="m",
    time_unit="s",
)

FRAME_CRTBP = ReferenceFrame(
    name="CRTBP Normalized",
    origin="Barycentre des deux primaires",
    is_inertial=False,
    length_unit="LU",
    time_unit="TU",
)

FRAME_ECI = ReferenceFrame(
    name="Earth-Centered Inertial J2000.0",
    origin="Centre de la Terre",
    is_inertial=True,
    length_unit="m",
    time_unit="s",
)


class CoordinateTransformer:
    """
    Classe principale pour les transformations de coordonnées.

    Cette classe gère toutes les transformations entre référentiels,
    en tenant compte des approximations et de leurs impacts.
    """

    def __init__(
        self,
        ephem_manager: EphemerisManager = EphemerisManager(),
        include_moon: bool = False,
    ):
        """
        Initialise le transformateur de coordonnées.

        Args:
            include_moon: Si True, utilise le barycentre Terre-Lune.
                         Si False, utilise le centre de la Terre (approximation).

        Approximation (include_moon=False):
            - Erreur de position: ~4,700 km (rayon orbite lunaire × μ_Moon)
            - Acceptable pour études préliminaires
            - DOIT être True pour haute-fidélité
        """
        self.include_moon = include_moon
        self.ephem_manager = ephem_manager

        # Paramètres du système
        self.mu_ratio = Constants.MU_RATIO_SUN_EARTH
        self.R = Constants.AU
        self.omega = Constants.OMEGA_EARTH

        # Distance barycentre Terre-Lune au centre Terre
        if include_moon:
            self.r_earth_to_barycenter = (
                Constants.R_EARTH_MOON * Constants.MU_RATIO_EARTH_MOON
            )
        else:
            self.r_earth_to_barycenter = 0.0

    # ========== TRANSFORMATIONS ÉCLIPTIQUE ↔ RLP ==========

    def ecliptic_to_rlp(
        self,
        state_ecliptic: StateVector,
        time: float,
        earth_position: Optional[PositionVector] = None,
        earth_velocity: Optional[VelocityVector] = None,
    ) -> StateVector:
        """
        Transforme de l'écliptique J2000 vers le référentiel RLP.

        Le référentiel RLP tourne avec la Terre autour du Soleil:
        - Origine: barycentre Soleil-Terre(-Lune)
        - Axe X: du Soleil vers la Terre
        - Axe Z: pÎle nord écliptique
        - Axe Y: complÚte le triÚdre direct

        Args:
            state_ecliptic: État [x, y, z, vx, vy, vz] en écliptique [m, m/s]
            time: Temps depuis J2000.0 [s]
            earth_position: Position de la Terre (optionnel, sinon circulaire)
            earth_velocity: Vitesse de la Terre (optionnel, sinon circulaire)

        Returns:
            État dans le référentiel RLP [m, m/s]
        """
        # 1. Obtenir position/vitesse Terre
        if earth_position is None or earth_velocity is None:
            if self.ephem_manager is not None:
                # PRIORITÉ : utiliser les éphémérides réelles
                et = self.ephem_manager.et_from_j2000(time)
                earth_pos, earth_vel = self.ephem_manager.get_body_state(
                    "EARTH", et, "SSB", "J2000"
                )
            else:
                # Fallback : orbite circulaire (approximation)
                earth_pos, earth_vel = self._compute_earth_circular_orbit(time)
        else:
            earth_pos: PositionVector = earth_position.copy()
            earth_vel: VelocityVector = earth_velocity.copy()

        sun_pos: PositionVector = create_postion_vector()
        sun_vel: VelocityVector = create_velocity_vector()

        # 2. Position du Soleil (en coordonnées SSB si disponible)
        if not self.ephem_manager is None:
            et = self.ephem_manager.et_from_j2000(time)
            try:
                sun_pos, sun_vel = self.ephem_manager.get_body_state(
                    "SUN", et, "SSB", "J2000"
                )
            except Exception:
                pass

        # 3. Vecteur Soleil → Terre
        sun_to_earth = earth_pos - sun_pos
        r_se = np.linalg.norm(sun_to_earth)

        # 4. Position du BARYCENTRE Soleil-Terre
        barycenter_pos = sun_pos + self.mu_ratio * sun_to_earth
        barycenter_vel = sun_vel + self.mu_ratio * earth_vel

        # 5. Construction de la matrice de rotation Écliptique → RLP
        x_axis = sun_to_earth / r_se
        z_axis = np.array([0.0, 0.0, 1.0], np.float64)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)  # Réorthogonalisation

        R_ecliptic_to_rlp = np.array([x_axis, y_axis, z_axis], np.float64)

        # 6. Transformation de POSITION
        pos_relative = state_ecliptic[:3] - barycenter_pos
        pos_rlp = R_ecliptic_to_rlp @ pos_relative

        # 7. Transformation de VITESSE
        #
        # Formule : v_RLP = R × (v_ecliptic - v_barycenter) - ω × r_RLP
        #
        # OÙ :
        # - R : matrice de rotation
        # - ω : vitesse angulaire du référentiel RLP
        # - ω × r_RLP : vitesse d'entraînement (terme de Coriolis)

        vel_relative = state_ecliptic[3:6] - barycenter_vel
        vel_rotated = R_ecliptic_to_rlp @ vel_relative

        # Vitesse angulaire dans RLP
        omega_ecliptic = np.array([0.0, 0.0, self.omega], np.float64)
        omega_rlp = R_ecliptic_to_rlp @ omega_ecliptic

        # VÉRIFICATION : omega_rlp devrait Être ≈ [0, 0, ω_Earth]
        # car la rotation préserve l'axe Z
        expected_omega = np.array([0.0, 0.0, self.omega], np.float64)
        omega_error = np.linalg.norm(omega_rlp - expected_omega)
        if omega_error > 1e-6:
            import warnings

            warnings.warn(
                f"Vitesse angulaire aprÚs rotation incorrecte: "
                f"||ω_RLP - [0,0,ω]|| = {omega_error:.3e}"
            )

        # Vitesse dans RLP = vitesse après rotation - vitesse d'entraînement
        vel_rlp = vel_rotated - np.cross(omega_rlp, pos_rlp)

        return np.concatenate([pos_rlp, vel_rlp]).astype(np.float64)

    def rlp_to_ecliptic(
        self,
        state_rlp: StateVector,
        time: float,
        earth_position: Optional[PositionVector] = None,
        earth_velocity: Optional[VelocityVector] = None,
    ) -> StateVector:
        """
        Transforme du référentiel RLP vers l'écliptique J2000.

        Transformation inverse de ecliptic_to_rlp.
        """
        # 1. Position/vitesse Terre
        if earth_position is None or earth_velocity is None:
            if self.ephem_manager is not None:
                et = self.ephem_manager.et_from_j2000(time)
                earth_pos, earth_vel = self.ephem_manager.get_body_state(
                    "EARTH", et, "SSB", "J2000"
                )
            else:
                earth_pos, earth_vel = self._compute_earth_circular_orbit(time)
        else:
            earth_pos = earth_position.copy()
            earth_vel = earth_velocity.copy()

        sun_pos: PositionVector = create_postion_vector()
        sun_vel: VelocityVector = create_velocity_vector()

        # Obtenir la position du Soleil si possible (SSB)
        if self.ephem_manager is not None:
            et = self.ephem_manager.et_from_j2000(time)
            try:
                sun_pos, sun_vel = self.ephem_manager.get_body_state(
                    "SUN", et, "SSB", "J2000"
                )
            except Exception:
                pass

        sun_to_earth = earth_pos - sun_pos
        r_se = np.linalg.norm(sun_to_earth)

        # 2. Barycentre
        barycenter_pos = sun_pos + self.mu_ratio * sun_to_earth
        barycenter_vel = sun_vel + self.mu_ratio * earth_vel

        # 3. Matrice de rotation RLP → Écliptique (transposée)
        x_axis = sun_to_earth / r_se
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        R_rlp_to_ecliptic = np.array([x_axis, y_axis, z_axis]).T

        # 4. Transformation de position
        pos_ecliptic = R_rlp_to_ecliptic @ state_rlp[:3] + barycenter_pos

        # 5. Transformation de vitesse (inverse)
        omega_ecliptic = np.array([0.0, 0.0, self.omega], dtype=np.float64)
        omega_rlp = np.array([x_axis, y_axis, z_axis]) @ omega_ecliptic

        vel_with_rotation = state_rlp[3:6] + np.cross(omega_rlp, state_rlp[:3])
        vel_ecliptic = R_rlp_to_ecliptic @ vel_with_rotation + barycenter_vel

        return np.concatenate([pos_ecliptic, vel_ecliptic], dtype=np.float64)  # type: ignore

    # ========== TRANSFORMATIONS RLP ↔ CRTBP NORMALISÉ ==========

    def rlp_to_crtbp(self, state_rlp: StateVector) -> StateVector:
        """
        Normalise un état RLP vers les coordonnées CRTBP adimensionnelles.

        Le système CRTBP utilise des unités normalisées:
        - Longueur: distance entre primaires = 1
        - Temps: période orbitale / (2π) = 1
        - Masse totale: 1
        - Vitesse angulaire: 1

        Args:
            state_rlp: État en RLP [m, m/s]

        Returns:
            État normalisé CRTBP (adimensionnel)

        Note:
            Cette transformation est purement géométrique (scaling).
            Les équations du mouvement sont différentes entre RLP et CRTBP.
        """
        L_star = self.R
        T_star = 1.0 / self.omega
        V_star = L_star / T_star

        state_crtbp = create_state_vector()
        state_crtbp[:3] = state_rlp[:3] / L_star
        state_crtbp[3:6] = state_rlp[3:6] / V_star

        return state_crtbp

    def crtbp_to_rlp(self, state_crtbp: StateVector) -> StateVector:
        """
        Dénormalise un état CRTBP vers les coordonnées RLP physiques.

        Args:
            state_crtbp: État normalisé CRTBP (adimensionnel)

        Returns:
            État en RLP [m, m/s]
        """
        L_star = self.R
        T_star = 1.0 / self.omega
        V_star = L_star / T_star

        state_rlp = create_state_vector()
        state_rlp[:3] = state_crtbp[:3] * L_star
        state_rlp[3:6] = state_crtbp[3:6] * V_star

        return state_rlp

    # ========== TRANSFORMATIONS ÉCLIPTIQUE ↔ ECI ==========

    def ecliptic_to_eci(self, state_ecliptic: StateVector) -> StateVector:
        """
        Transforme de l'écliptique J2000 vers ECI (Earth-Centered Inertial) J2000.

        La transformation est une simple rotation autour de l'axe X
        de l'angle ε (obliquité de l'écliptique).

        Args:
            state_ecliptic: État en écliptique [m, m/s]

        Returns:
            État en ECI [m, m/s]

        Note:
            Les deux référentiels sont inertiels, donc la transformation
            de vitesse est identique à celle de position.
        """
        eps = Constants.EPSILON_EARTH

        R = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos(eps), sin(eps)], [0.0, -sin(eps), cos(eps)]],
            dtype=np.float64,
        )

        state_eci = create_state_vector()
        state_eci[:3] = R @ state_ecliptic[:3]
        state_eci[3:6] = R @ state_ecliptic[3:6]

        return state_eci

    def eci_to_ecliptic(self, state_eci: StateVector) -> StateVector:
        """
        Transforme de ECI vers l'écliptique J2000.

        Args:
            state_eci: État en ECI [m, m/s]

        Returns:
            État en écliptique [m, m/s]
        """
        eps = Constants.EPSILON_EARTH

        R = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos(eps), -sin(eps)], [0.0, sin(eps), cos(eps)]],
            dtype=np.float64,
        )

        state_ecliptic = create_state_vector()
        state_ecliptic[:3] = R @ state_eci[:3]
        state_ecliptic[3:6] = R @ state_eci[3:6]

        return state_ecliptic

    # ========== UTILITAIRES ==========

    def _compute_earth_circular_orbit(
        self, time: float
    ) -> Tuple[PositionVector, VelocityVector]:
        """
        Calcule la position et vitesse de la Terre sur une orbite circulaire.

        Args:
            time: Temps depuis J2000.0 [s]

        Returns:
            (position [m], vitesse [m/s]) de la Terre en écliptique

        Approximation:
            - Orbite parfaitement circulaire (e = 0)
            - Erreur réelle due à e = 0.0167:
              * Position: ±2.5 million km
              * Vitesse: ±500 m/s
            - Acceptable pour analyse qualitative
            - NON acceptable pour navigation précise
        """
        # Anomalie moyenne
        M = self.omega * time

        x = self.R * cos(M)
        y = self.R * sin(M)
        z = 0.0
        position = np.array([x, y, z], dtype=np.float64)

        vx = -self.R * self.omega * sin(M)
        vy = self.R * self.omega * cos(M)
        vz = 0.0
        velocity = np.array([vx, vy, vz], dtype=np.float64)

        return position, velocity

    def compute_l2_distance_from_earth(self) -> float:
        """
        Calcule la distance entre la Terre et le point L2.

        Returns:
            Distance Terre-L2 [m]

        Formule:
            d = a × (μ/3)^(1/3)
            où a est la distance Soleil-Terre
            et μ = M_Earth / (M_Sun + M_Earth)

        Valeur numérique pour Soleil-Terre:
            d ≈ 1.5 million km
        """
        return self.R * (self.mu_ratio / 3.0) ** (1.0 / 3.0)

    def compute_l2_position_rlp(self) -> PositionVector:
        """
        Calcule la position du point L2 dans le référentiel RLP.

        Returns:
            Position de L2 [m] dans RLP

        Note:
            L2 est sur l'axe X positif (derrière la Terre vue du Soleil).
            La distance est calculée par la méthode de Newton-Raphson
            (voir module lagrange_points.py).
        """
        # Distance approximative (formule de Taylor au 1er ordre)
        d_approx = self.R * (self.mu_ratio / 3.0) ** (1.0 / 3.0)
        x_earth_in_rlp = self.R * (1.0 - self.mu_ratio)
        x_l2 = x_earth_in_rlp + d_approx

        return np.array([x_l2, 0.0, 0.0], dtype=np.float64)

    def compute_earth_position_rlp(self) -> PositionVector:
        """Position de la Terre dans le référentiel RLP."""
        x_earth = (1.0 - self.mu_ratio) * self.R
        return np.array([x_earth, 0.0, 0.0], dtype=np.float64)

    def compute_sun_position_rlp(self) -> PositionVector:
        """
        Calcule la position du Soleil dans le référentiel RLP.

        Returns:
            Position du Soleil [m] dans RLP

        Note:
            Le Soleil est très proche du barycentre.
            Distance: μ × R ≈ 450 km seulement!
        """
        x_sun = -self.mu_ratio * self.R
        return np.array([x_sun, 0.0, 0.0], dtype=np.float64)

    def compute_rotation_angle(self, time: float) -> float:
        """
        Calcule l'angle de rotation du référentiel RLP depuis J2000.

        Args:
            time: Temps depuis J2000.0 [s]

        Returns:
            Angle de rotation [rad]

        Approximation:
            - Rotation uniforme (néglige l'excentricité)
            - Erreur: ±1.7° sur l'année (due à l'équation du temps)
        """
        return self.omega * time

    def get_transformation_info(
        self,
        from_frame: ReferenceFrame,
        to_frame: ReferenceFrame,
        time: Optional[float] = None,
    ) -> str:
        """
        Retourne une description de la transformation entre deux référentiels.

        Args:
            from_frame: Référentiel source
            to_frame: Référentiel cible
            time: Temps pour les transformations dépendantes du temps

        Returns:
            Description textuelle de la transformation
        """
        info = f"Transformation: {from_frame.name} → {to_frame.name}\n"
        info += f"  Origine: {from_frame.origin} → {to_frame.origin}\n"
        info += f"  Type: {('inertiel' if from_frame.is_inertial else 'rotatif')} → "
        info += f"{('inertiel' if to_frame.is_inertial else 'rotatif')}\n"

        if not from_frame.is_inertial or not to_frame.is_inertial:
            if time is not None:
                angle_deg = np.rad2deg(self.compute_rotation_angle(time))
                info += f"  Angle de rotation: {angle_deg:.2f}°\n"
            else:
                info += f"  Note: transformation dépendante du temps\n"

        info += f"  Unités: {from_frame.length_unit} → {to_frame.length_unit}"

        return info


# ========== FONCTIONS UTILITAIRES GLOBALES ==========


def cartesian_to_spherical(position: PositionVector) -> Tuple[float, float, float]:
    """
    Convertit des coordonnées cartésiennes en sphériques.

    Args:
        position: Vecteur position [x, y, z] [m]

    Returns:
        (r, theta, phi) où:
            r: distance [m]
            theta: azimuth dans le plan XY depuis X [rad] ∈ [0, 2π]
            phi: élévation depuis le plan XY [rad] ∈ [-π/2, π/2]
    """
    x, y, z = position

    r = float(np.linalg.norm(position))

    if r < 1e-10:
        return 0.0, 0.0, 0.0

    theta = atan2(y, x)
    if theta < 0:
        theta += 2 * pi

    phi = atan2(z, sqrt(x**2 + y**2))

    return r, theta, phi


def spherical_to_cartesian(r: float, theta: float, phi: float) -> PositionVector:
    """
    Convertit des coordonnées sphériques en cartésiennes.

    Args:
        r: distance [m]
        theta: azimuth [rad]
        phi: élévation [rad]

    Returns:
        Vecteur position [x, y, z] [m]
    """
    x = r * cos(phi) * cos(theta)
    y = r * cos(phi) * sin(theta)
    z = r * sin(phi)

    return np.array([x, y, z], dtype=np.float64)


def compute_sun_angle(
    position_rlp: PositionVector, sun_to_spacecraft: bool = True
) -> Tuple[float, float]:
    """
    Calcule les angles par rapport au Soleil dans le référentiel RLP.

    Args:
        position_rlp: Position dans RLP [m]
        sun_to_spacecraft: Si True, vecteur Soleil→SC
                          Si False, vecteur SC→Soleil

    Returns:
        (theta, phi) où:
            theta: angle dans le plan XY [rad]
            phi: angle hors du plan XY [rad]
    """
    if sun_to_spacecraft:
        vec = position_rlp
    else:
        vec = -position_rlp

    _, theta, phi = cartesian_to_spherical(vec)

    return theta, phi


def distance_to_primary(
    state: StateVector, x1: float, x2: float, normalized: bool = True
) -> Tuple[float, float]:
    """
    Calcule les distances aux deux primaires dans le CRTBP.

    Args:
        state: Vecteur d'état [x, y, z, vx, vy, vz]
        x1: Position X du primaire 1
        x2: Position X du primaire 2
        normalized: Si True, distances normalisées (LU) sinon en mètres

    Returns:
        (r1, r2): Distances aux primaires 1 et 2
    """
    x, y, z, vx, vy, _ = state

    r1 = np.sqrt((x - x1) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - x2) ** 2 + y**2 + z**2)

    r1 = max(r1, 1e-10 if normalized else 1.0)
    r2 = max(r2, 1e-10 if normalized else 1.0)

    return r1, r2
