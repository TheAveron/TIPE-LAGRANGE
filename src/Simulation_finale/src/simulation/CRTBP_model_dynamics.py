"""
Module du modèle dynamique pour la simulation JWST L2.

Ce module implémente deux niveaux de modélisation:
1. CRTBP (Circular Restricted Three-Body Problem) - pour l'analyse théorique
2. Modèle haute-fidélité avec éphémérides JPL - pour la simulation opérationnelle

Hiérarchie des approximations:
    CRTBP < Éphémérides < Éphémérides+SRP < Éphémérides+SRP+Relativité

Organisation:
    - Classe abstraite BaseDynamics (interface commune)
    - Classe CRTBP3Body (modèle simplifié)
    - Classe HighFidelityDynamics (modèle complet)

Auteur: Assistant
Date: 2025-01-10
"""

import numpy as np

from .constants import Constants
from .coordinates import PositionVector, StateVector
from .dynamics_conf import BaseDynamics, DynamicsConfig

# ========== TYPES ET ÉNUMÉRATIONS ==========


class CRTBP3Body(BaseDynamics):
    """
    Problème Restreint Circulaire à Trois Corps (CRTBP).

    Hypothèses:
    1. Les deux corps primaires (Soleil et Terre) sont en orbite circulaire
    2. Les masses sont ponctuelles
    3. Le troisième corps (spacecraft) a une masse négligeable
    4. Pas de perturbations externes

    Référentiel:
        - Origine: barycentre des deux primaires
        - Axe X: du primaire vers le secondaire
        - Axe Z: perpendiculaire au plan orbital
        - Le référentiel tourne avec vitesse angulaire constante ω

    Unités:
        - Si normalized=True: unités canoniques (distance=1, période=2π)
        - Si normalized=False: unités SI (m, s)

    Applications:
        - Analyse qualitative des trajectoires
        - Calcul de la constante de Jacobi
        - Détermination des vecteurs propres
        - Génération d'orbites initiales

    Limites:
        - Néglige l'excentricité terrestre (erreur ~2.5 million km/an)
        - Néglige les perturbations lunaires et planétaires
        - Pas adapté pour navigation précise > 100 jours
    """

    def __init__(
        self,
        config: DynamicsConfig,
        normalized: bool = False,
        primary: str = "sun-earth",
    ):
        """
        Initialise le modèle CRTBP.

        Args:
            config: Configuration du modèle
            normalized: Si True, utilise les unités normalisées
            primary: Système primaire ("sun-earth" ou "earth-moon")
        """
        super().__init__(config)

        self.normalized = normalized
        self.primary = primary

        if primary == "sun-earth":
            self.mu = Constants.MU_RATIO_SUN_EARTH
            self.R = Constants.AU
            self.omega = Constants.OMEGA_EARTH
            self.M_total = Constants.M_SUN + Constants.M_EARTH

            self.GM_1 = Constants.MU_SUN
            self.GM_2 = Constants.MU_EARTH
        elif primary == "earth-moon":
            self.mu = Constants.MU_RATIO_EARTH_MOON
            self.R = Constants.R_EARTH_MOON
            self.omega = np.sqrt(
                Constants.G * (Constants.M_EARTH + Constants.M_MOON) / self.R**3
            )
            self.M_total = Constants.M_EARTH + Constants.M_MOON

            self.GM_1 = Constants.MU_EARTH
            self.GM_2 = Constants.MU_MOON
        else:
            raise ValueError(f"Système primaire inconnu: {primary}")

        # Unités caractéristiques pour normalisation
        self.L_star = self.R
        self.T_star = 1.0 / self.omega
        self.V_star = self.L_star / self.T_star

        # Positions des primaires dans le référentiel tournant normalisé
        # Primaire 1 (plus massif, ex: Soleil) à x = -μ
        # Primaire 2 (moins massif, ex: Terre) à x = 1-μ
        self.x1 = -self.mu
        self.x2 = 1.0 - self.mu

        if not normalized:
            self.x1 *= self.R
            self.x2 *= self.R

    def equations_of_motion(self, t: float, state: StateVector) -> StateVector:
        """
        Équations du mouvement dans le référentiel tournant.

        Équations (en notation vectorielle):
            r̈ - 2ω × ṙ - ω × (ω × r) = -∇U

        où U est le pseudo-potentiel:
            U = -G(m₁/r₁ + m₂/r₂)

        En coordonnées explicites:
            ẍ - 2ẏ = ∂U*/∂x
            ÿ + 2ẋ = ∂U*/∂y
            z̈ = ∂U*/∂z

        où U* inclut le terme centrifuge:
            U* = U + ½ω²(x² + y²)

        Args:
            t: Temps (non utilisé dans CRTBP autonome)
            state: [x, y, z, vx, vy, vz]

        Returns:
            [vx, vy, vz, ax, ay, az]

        Note sur les unités:
            Si normalized=True, ω=1 et les distances sont normalisées
            Si normalized=False, tout est en unités SI
        """
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]

        acc = self.compute_acceleration(t, state)

        state_dot = np.array([vx, vy, vz, acc[0], acc[1], acc[2]])
        return state_dot

    def compute_acceleration(self, t: float, state: StateVector) -> PositionVector:
        """
        Calcule l'accélération dans le CRTBP.

        Équations du mouvement dans le référentiel tournant :
            r̈ = ∇U* - 2Ω × ṙ

        où U* est le pseudo-potentiel (effectif) :
            U* = (1-μ)/r₁ + μ/r₂ + ½(x² + y²)

        Le gradient de U* contient DÉJÀ la force centrifuge (terme ½(x²+y²)).
        La force de Coriolis -2Ω×ṙ est traitée séparément.

        En composantes explicites :
            ẍ - 2ẏ = ∂U*/∂x = -(1-μ)(x-x₁)/r₁³ - μ(x-x₂)/r₂³ + x
            ÿ + 2ẋ = ∂U*/∂y = -(1-μ)y/r₁³ - μy/r₂³ + y
            z̈      = ∂U*/∂z = -(1-μ)z/r₁³ - μz/r₂³

        Note : Le signe NÉGATIF devant (x-x₁) vient de :
            ∂(1/r₁)/∂x = ∂/∂x[1/√((x-x₁)²+y²+z²)] = -(x-x₁)/r₁³

        Args:
            t: Temps (non utilisé dans CRTBP autonome)
            state: État [x, y, z, vx, vy, vz]

        Returns:
            Accélération [ax, ay, az]
        """
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]

        # Distances aux primaires
        r1 = np.sqrt((x - self.x1) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - self.x2) ** 2 + y**2 + z**2)

        # Protection contre division par zéro (collision)
        r1 = max(r1, 1e-10 if self.normalized else 1.0)
        r2 = max(r2, 1e-10 if self.normalized else 1.0)

        if self.normalized:
            # Équations normalisées (ω = 1)

            # ∂U*/∂x = -(1-μ)(x-x₁)/r₁³ - μ(x-x₂)/r₂³ + x
            dU_dx = (
                -(1.0 - self.mu) * (x - self.x1) / r1**3
                - self.mu * (x - self.x2) / r2**3
                + x
            )

            # ∂U*/∂y = -(1-μ)y/r₁³ - μy/r₂³ + y
            dU_dy = -(1.0 - self.mu) * y / r1**3 - self.mu * y / r2**3 + y

            # ∂U*/∂z = -(1-μ)z/r₁³ - μz/r₂³
            dU_dz = -(1.0 - self.mu) * z / r1**3 - self.mu * z / r2**3

            # Force de Coriolis : -2Ω × v = -2[0,0,1] × [vx,vy,vz] = [-2vy, 2vx, 0]
            # (Le signe est négatif, donc on a +2vy en x et -2vx en y)
            ax = dU_dx + 2.0 * vy
            ay = dU_dy - 2.0 * vx
            az = dU_dz

        else:
            # Équations non normalisées (unités SI)
            omega_sq = self.omega**2

            # Masses effectives
            GM_1 = self.GM_1
            GM_2 = self.GM_2

            # ∂U*/∂x = -G×m₁(x-x₁)/r₁³ - G×m₂(x-x₂)/r₂³ + ω²x
            dU_dx = (
                -GM_1 * (x - self.x1) / r1**3
                - GM_2 * (x - self.x2) / r2**3
                + omega_sq * x
            )

            # ∂U*/∂y = -G×m₁y/r₁³ - G×m₂y/r₂³ + ω²y
            dU_dy = -GM_1 * y / r1**3 - GM_2 * y / r2**3 + omega_sq * y

            # ∂U*/∂z = -G×m₁z/r₁³ - G×m₂z/r₂³
            dU_dz = -GM_1 * z / r1**3 - GM_2 * z / r2**3

            # Force de Coriolis : -2Ω × v
            ax = dU_dx + 2.0 * self.omega * vy
            ay = dU_dy - 2.0 * self.omega * vx
            az = dU_dz

        return np.array([ax, ay, az])

    def jacobi_constant(self, state: StateVector) -> float:
        """
        Calcule la constante de Jacobi (intégrale du mouvement dans CRTBP).

        Formule:
            C = 2U* - v²

        où:
            U* = (1-μ)/r₁ + μ/r₂ + ½(x² + y²)  (normalisé)
            v² = vx² + vy² + vz²

        Propriété:
            Dans le CRTBP pur, C est constant le long d'une trajectoire.
            Une variation de C indique des perturbations externes.

        Args:
            state: État [x, y, z, vx, vy, vz]

        Returns:
            Constante de Jacobi C (adimensionnelle si normalized=True)

        Note:
            Dans le modèle haute-fidélité, C varie à cause de:
            - L'excentricité de l'orbite terrestre (~1% d'oscillation)
            - Les perturbations lunaires et planétaires
            - La pression de radiation solaire
        """
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]

        r1 = np.sqrt((x - self.x1) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - self.x2) ** 2 + y**2 + z**2)

        r1 = max(r1, 1e-10 if self.normalized else 1.0)
        r2 = max(r2, 1e-10 if self.normalized else 1.0)

        v_squared = vx**2 + vy**2 + vz**2

        if self.normalized:
            # Pseudo-potentiel normalisé
            U_star = (1.0 - self.mu) / r1 + self.mu / r2 + 0.5 * (x**2 + y**2)

            C = 2.0 * U_star - v_squared
        else:

            GM_1 = self.GM_1
            GM_2 = self.GM_2

            U_star = GM_1 / r1 + GM_2 / r2 + 0.5 * self.omega**2 * (x**2 + y**2)

            C = 2.0 * U_star - v_squared

        return C

    def effective_potential(self, x: float, y: float, z: float = 0.0) -> float:
        """
        Calcule le potentiel effectif U* en un point.

        Utile pour visualiser les régions interdites (courbes de Hill).

        Args:
            x, y, z: Coordonnées du point

        Returns:
            U* (pseudo-potentiel)
        """
        r1 = np.sqrt((x - self.x1) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - self.x2) ** 2 + y**2 + z**2)

        r1 = max(r1, 1e-10 if self.normalized else 1.0)
        r2 = max(r2, 1e-10 if self.normalized else 1.0)

        if self.normalized:
            U_star = (1.0 - self.mu) / r1 + self.mu / r2 + 0.5 * (x**2 + y**2)
        else:
            GM_1 = self.GM_1
            GM_2 = self.GM_2

            U_star = GM_1 / r1 + GM_2 / r2 + 0.5 * self.omega**2 * (x**2 + y**2)

        return U_star
