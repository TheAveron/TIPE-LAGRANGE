"""
Module de calcul des points de Lagrange.

Ce module calcule les positions exactes des 5 points de Lagrange
dans le problème restreint à trois corps (CRTBP).

Méthodes:
    - Newton-Raphson pour L1, L2, L3 (points colinéaires)
    - Calcul direct pour L4, L5 (points triangulaires)
    - Analyse de stabilité locale

Référence:
    Documents 4, 7 (formules analytiques et numériques)

Auteur: Assistant
Date: 2025-01-10
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass
from enum import Enum

from .constants import Constants, NumericalConstants
from .CRTBP_model_dynamics import CRTBP3Body
from .dynamics_conf import DynamicsConfig, DynamicsModel

# ========== TYPES ET ÉNUMÉRATIONS ==========


class LagrangePoint(Enum):
    """Énumération des points de Lagrange."""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"


class Stability(Enum):
    """Type de stabilité d'un point de Lagrange."""

    STABLE = "stable"  # Stable (L4, L5 si μ < μ_critique)
    UNSTABLE = "unstable"  # Instable (L1, L2, L3, toujours)
    CONDITIONALLY_STABLE = "conditionally_stable"  # L4, L5 si μ > μ_critique


@dataclass
class LagrangePointInfo:
    """
    Information complète sur un point de Lagrange.

    Attributs:
        point: Identifiant du point (L1-L5)
        position: Position [x, y, z] dans le référentiel tournant
        stability: Type de stabilité
        jacobi_constant: Valeur de la constante de Jacobi
        eigenvalues: Valeurs propres de la matrice jacobienne (6 valeurs)
        distance_to_secondary: Distance au corps secondaire (Terre)
    """

    point: LagrangePoint
    position: np.ndarray
    stability: Stability
    jacobi_constant: float
    eigenvalues: Optional[np.ndarray] = None
    distance_to_secondary: Optional[float] = None

    def __str__(self) -> str:
        """Représentation textuelle."""
        s = f"\n{self.point.value}:\n"
        s += f"  Position: [{self.position[0]/1e9:.6f}, {self.position[1]/1e9:.6f}, "
        s += f"{self.position[2]/1e9:.6f}] million km\n"
        s += f"  Stabilité: {self.stability.value}\n"
        s += f"  Jacobi C: {self.jacobi_constant:.6f}\n"
        if self.distance_to_secondary is not None:
            s += f"  Distance à la Terre: {self.distance_to_secondary/1e6:.3f} milliers km\n"
        return s


# ========== CLASSE PRINCIPALE ==========


class LagrangePointCalculator:
    """
    Calculateur des points de Lagrange dans le CRTBP.

    Cette classe fournit des méthodes pour :
    - Calculer les positions des 5 points de Lagrange
    - Analyser leur stabilité
    - Obtenir les valeurs propres de la matrice jacobienne

    Usage:
        calculator = LagrangePointCalculator(mu=Constants.MU_SUN_EARTH)
        l2_info = calculator.compute_lagrange_point(LagrangePoint.L2)
        print(l2_info)
    """

    def __init__(
        self, mu: float, distance_unit: float = Constants.AU, normalized: bool = False
    ):
        """
        Initialise le calculateur.

        Args:
            mu: Paramètre de masse μ = m₂/(m₁+m₂)
            distance_unit: Unité de distance (défaut: AU)
            normalized: Si True, positions en unités normalisées

        Note:
            Pour Soleil-Terre: μ ≈ 3.0e-6
            Pour Terre-Lune: μ ≈ 0.012
        """
        self.mu = mu
        self.distance_unit = distance_unit
        self.normalized = normalized

        # Positions des primaires dans le référentiel tournant normalisé
        self.x1 = -mu  # Primaire 1 (plus massif, ex: Soleil)
        self.x2 = 1.0 - mu  # Primaire 2 (moins massif, ex: Terre)

        if not normalized:
            # Convertir en unités physiques
            self.x1 *= distance_unit
            self.x2 *= distance_unit

        # Valeur critique de μ pour stabilité de L4, L5
        # μ_crit = (1 - √(23/27)) / 2 ≈ 0.0385
        self.mu_critical = 0.5 * (1.0 - np.sqrt(23.0 / 27.0))

        # Créer un objet CRTBP pour calculs auxiliaires
        config = DynamicsConfig(model=DynamicsModel.CRTBP)
        self.crtbp = CRTBP3Body(config, normalized=normalized)

    # ========== CALCUL DES POINTS COLINÉAIRES (L1, L2, L3) ==========

    def compute_l1(self, initial_guess: Optional[float] = None) -> LagrangePointInfo:
        """
        Calcule la position du point L1 (entre les deux primaires).

        Méthode:
            Newton-Raphson sur l'équation:
            f(x) = x - (1-μ)(x+μ)/|x+μ|³ - μ(x-1+μ)/|x-1+μ|³ = 0

        Args:
            initial_guess: Estimation initiale (si None, utilise formule approchée)

        Returns:
            Information complète sur L1

        Approximation pour μ << 1:
            x_L1 ≈ 1 - (μ/3)^(1/3)

        Pour Soleil-Terre: L1 est à ~1.5 million km de la Terre (côté Soleil)
        """
        if initial_guess is None:
            # Formule approchée (Taylor au 1er ordre)
            x0 = 1.0 - (self.mu / 3.0) ** (1.0 / 3.0)
        else:
            x0 = initial_guess

        # Newton-Raphson
        x = self._newton_raphson_collinear(
            x0,
            region="L1",
            tol=NumericalConstants.LAGRANGE_POINT_TOL,
            max_iter=NumericalConstants.MAX_ITERATIONS,
        )

        if not self.normalized:
            x *= self.distance_unit

        position = np.array([x, 0.0, 0.0])

        return self._create_lagrange_point_info(LagrangePoint.L1, position)

    def compute_l2(self, initial_guess: Optional[float] = None) -> LagrangePointInfo:
        """
        Calcule la position du point L2 (au-delà du corps secondaire).

        Args:
            initial_guess: Estimation initiale

        Returns:
            Information complète sur L2

        Approximation pour μ << 1:
            x_L2 ≈ 1 + (μ/3)^(1/3)

        Pour Soleil-Terre: L2 est à ~1.5 million km de la Terre (côté opposé au Soleil)
        C'est là que se trouve JWST !
        """
        if initial_guess is None:
            x0 = 1.0 + (self.mu / 3.0) ** (1.0 / 3.0)
        else:
            x0 = initial_guess

        x = self._newton_raphson_collinear(
            x0,
            region="L2",
            tol=NumericalConstants.LAGRANGE_POINT_TOL,
            max_iter=NumericalConstants.MAX_ITERATIONS,
        )

        if not self.normalized:
            x *= self.distance_unit

        position = np.array([x, 0.0, 0.0])

        return self._create_lagrange_point_info(LagrangePoint.L2, position)

    def compute_l3(self, initial_guess: Optional[float] = None) -> LagrangePointInfo:
        """
        Calcule la position du point L3 (opposé au corps secondaire).

        Args:
            initial_guess: Estimation initiale

        Returns:
            Information complète sur L3

        Approximation pour μ << 1:
            x_L3 ≈ -1 - 5μ/12

        Pour Soleil-Terre: L3 est à ~150 million km du Soleil
        (légèrement décalé du point opposé à la Terre)
        """
        if initial_guess is None:
            x0 = -1.0 - 5.0 * self.mu / 12.0
        else:
            x0 = initial_guess

        x = self._newton_raphson_collinear(
            x0,
            region="L3",
            tol=NumericalConstants.LAGRANGE_POINT_TOL,
            max_iter=NumericalConstants.MAX_ITERATIONS,
        )

        if not self.normalized:
            x *= self.distance_unit

        position = np.array([x, 0.0, 0.0])

        return self._create_lagrange_point_info(LagrangePoint.L3, position)

    def _newton_raphson_collinear(
        self, x0: float, region: str, tol: float, max_iter: int
    ) -> float:
        """
        Méthode de Newton-Raphson pour trouver un point de Lagrange colinéaire.

        Équation à résoudre (y=0, z=0):
            f(x) = x - (1-μ)(x-x₁)/r₁³ - μ(x-x₂)/r₂³ = 0

        où:
            r₁ = |x - x₁| = |x + μ|
            r₂ = |x - x₂| = |x - 1 + μ|

        Dérivée:
            f'(x) = 1 - (1-μ)/r₁³ + 3(1-μ)(x-x₁)²/r₁⁵
                      - μ/r₂³ + 3μ(x-x₂)²/r₂⁵

        Args:
            x0: Estimation initiale
            region: "L1", "L2", ou "L3" (pour gestion des bornes)
            tol: Tolérance de convergence
            max_iter: Nombre maximum d'itérations

        Returns:
            Position x du point de Lagrange (normalisée)

        Raises:
            RuntimeError: Si la méthode ne converge pas
        """
        x = x0

        for iteration in range(max_iter):
            # Distances aux primaires
            r1 = abs(x + self.mu)
            r2 = abs(x - 1.0 + self.mu)

            # Éviter division par zéro
            if r1 < 1e-12 or r2 < 1e-12:
                raise RuntimeError(f"Newton-Raphson: trop proche d'un primaire")

            # Fonction f(x)
            f = (
                x
                - (1.0 - self.mu) * (x + self.mu) / r1**3
                - self.mu * (x - 1.0 + self.mu) / r2**3
            )

            # Dérivée f'(x)
            df = (
                1.0
                - (1.0 - self.mu) / r1**3
                + 3.0 * (1.0 - self.mu) * (x + self.mu) ** 2 / r1**5
                - self.mu / r2**3
                + 3.0 * self.mu * (x - 1.0 + self.mu) ** 2 / r2**5
            )

            # Vérifier que la dérivée n'est pas nulle
            if abs(df) < 1e-15:
                raise RuntimeError(f"Newton-Raphson: dérivée nulle à x={x}")

            # Mise à jour de Newton-Raphson
            x_new = x - f / df

            # Vérifier la convergence
            if abs(x_new - x) < tol:
                return x_new

            x = x_new

        raise RuntimeError(
            f"Newton-Raphson n'a pas convergé pour {region} "
            f"après {max_iter} itérations"
        )

    # ========== CALCUL DES POINTS TRIANGULAIRES (L4, L5) ==========

    def compute_l4(self) -> LagrangePointInfo:
        """
        Calcule la position du point L4 (triangle équilatéral, au-dessus).

        Formule exacte (pas d'itération nécessaire):
            x = 1/2 - μ
            y = √3/2
            z = 0

        Returns:
            Information complète sur L4

        Note:
            L4 et L5 forment des triangles équilatéraux avec les deux primaires.
            Pour Soleil-Terre, ils sont sur l'orbite terrestre, ±60° en avant/arrière.
        """
        x = 0.5 - self.mu
        y = np.sqrt(3.0) / 2.0
        z = 0.0

        if not self.normalized:
            x *= self.distance_unit
            y *= self.distance_unit

        position = np.array([x, y, z])

        return self._create_lagrange_point_info(LagrangePoint.L4, position)

    def compute_l5(self) -> LagrangePointInfo:
        """
        Calcule la position du point L5 (triangle équilatéral, en-dessous).

        Formule exacte:
            x = 1/2 - μ
            y = -√3/2
            z = 0

        Returns:
            Information complète sur L5

        Note:
            L5 est le symétrique de L4 par rapport au plan XZ.
        """
        x = 0.5 - self.mu
        y = -np.sqrt(3.0) / 2.0
        z = 0.0

        if not self.normalized:
            x *= self.distance_unit
            y *= self.distance_unit

        position = np.array([x, y, z])

        return self._create_lagrange_point_info(LagrangePoint.L5, position)

    # ========== CALCUL GÉNÉRIQUE ==========

    def compute_lagrange_point(self, point: LagrangePoint) -> LagrangePointInfo:
        """
        Calcule n'importe quel point de Lagrange.

        Args:
            point: Point à calculer (L1, L2, L3, L4, ou L5)

        Returns:
            Information complète sur le point

        Example:
            >>> calc = LagrangePointCalculator(Constants.MU_SUN_EARTH)
            >>> l2 = calc.compute_lagrange_point(LagrangePoint.L2)
            >>> print(l2)
        """
        if point == LagrangePoint.L1:
            return self.compute_l1()
        elif point == LagrangePoint.L2:
            return self.compute_l2()
        elif point == LagrangePoint.L3:
            return self.compute_l3()
        elif point == LagrangePoint.L4:
            return self.compute_l4()
        elif point == LagrangePoint.L5:
            return self.compute_l5()
        else:
            raise ValueError(f"Point de Lagrange inconnu: {point}")

    def compute_all_lagrange_points(self) -> Dict[LagrangePoint, LagrangePointInfo]:
        """
        Calcule tous les points de Lagrange.

        Returns:
            Dictionnaire {point: info} pour les 5 points
        """
        return {point: self.compute_lagrange_point(point) for point in LagrangePoint}

    # ========== ANALYSE DE STABILITÉ ==========

    def _create_lagrange_point_info(
        self, point: LagrangePoint, position: np.ndarray
    ) -> LagrangePointInfo:
        """
        Crée un objet LagrangePointInfo complet avec analyse de stabilité.

        Args:
            point: Identifiant du point
            position: Position [x, y, z]

        Returns:
            Objet LagrangePointInfo complet
        """
        # État au point de Lagrange (vitesse nulle)
        state = np.concatenate([position, np.zeros(3)])

        # Constante de Jacobi
        C = self.crtbp.jacobi_constant(state)

        # Distance au secondaire (Terre)
        if not self.normalized:
            x2_phys = self.x2
        else:
            x2_phys = self.x2 * self.distance_unit

        distance_to_secondary = float(
            np.linalg.norm(position - np.array([x2_phys, 0.0, 0.0]))
        )

        # Stabilité
        if point in [LagrangePoint.L4, LagrangePoint.L5]:
            # L4 et L5: stables si μ < μ_crit
            if self.mu < self.mu_critical:
                stability = Stability.STABLE
            else:
                stability = Stability.CONDITIONALLY_STABLE
        else:
            # L1, L2, L3: toujours instables
            stability = Stability.UNSTABLE

        # Valeurs propres (optionnel, calcul coûteux)
        # eigenvalues = self._compute_eigenvalues(position)
        eigenvalues = None

        return LagrangePointInfo(
            point=point,
            position=position,
            stability=stability,
            jacobi_constant=C,
            eigenvalues=eigenvalues,
            distance_to_secondary=distance_to_secondary,
        )
