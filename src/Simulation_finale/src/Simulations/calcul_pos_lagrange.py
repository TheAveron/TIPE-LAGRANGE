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

"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from src.Simulations.coordinates import distance_to_primary

from ..Models.base_dynamics import DynamicsConfig, DynamicsModel
from ..Models.vectors import PositionVector
from .constants import Constants, NumericalConstants
from .CRTBP3_dynamics import CRTBP3Body


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
    position: PositionVector
    stability: Stability
    jacobi_constant: float
    eigenvalues: np.ndarray
    jacobian_matrix: np.ndarray
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
        self.x1 = -mu
        self.x2 = 1.0 - mu
        if not normalized:
            # Convertir en unités physiques
            self.x1 *= distance_unit
            self.x2 *= distance_unit

        # Valeur critique de μ pour stabilité de L4, L5
        # μ_crit = (1 - √(23/27)) / 2 ≈ 0.0385
        self.mu_critical = 0.5 * (1.0 - np.sqrt(23.0 / 27.0))

        config = DynamicsConfig(model=DynamicsModel.CRTBP)
        self.crtbp = CRTBP3Body(config, normalized=normalized)

    def _newton_raphson_collinear(
        self, x0: float, region: str, tol: float, max_iter: int = 500
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

        for _ in range(max_iter):
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

            if abs(df) < 1e-15:
                raise RuntimeError(f"Newton-Raphson: dérivée nulle à x={x}")

            x_new = x - f / df

            if abs(x_new - x) < tol:
                return x_new

            x = x_new

        raise RuntimeError(
            f"Newton-Raphson n'a pas convergé pour {region} "
            f"après {max_iter} itérations"
        )

    def _create_lagrange_point_info(
        self, point: LagrangePoint, position: PositionVector
    ) -> LagrangePointInfo:
        """
        Crée un objet LagrangePointInfo complet avec analyse de stabilité.

        Args:
            point: Identifiant du point
            position: Position [x, y, z]

        Returns:
            Objet LagrangePointInfo complet
        """
        state = np.concatenate([position, np.zeros(3)])

        # Constante de Jacobi
        C = self.crtbp.jacobi_constant(state)

        if not self.normalized:
            x2_phys = self.x2
        else:
            x2_phys = self.x2 * self.distance_unit

        distance_to_secondary = float(
            np.linalg.norm(position - np.array([x2_phys, 0.0, 0.0], dtype=np.float64))
        )

        if point in [LagrangePoint.L4, LagrangePoint.L5]:
            if self.mu < self.mu_critical:
                stability = Stability.STABLE
            else:
                stability = Stability.CONDITIONALLY_STABLE
        else:
            stability = Stability.UNSTABLE

        eigenvalues = self._compute_eigenvalues(position)
        jacobian_matrix = self._compute_jacobian_matrix(position)

        return LagrangePointInfo(
            point=point,
            position=position,
            stability=stability,
            jacobi_constant=C,
            eigenvalues=eigenvalues,
            distance_to_secondary=distance_to_secondary,
            jacobian_matrix=jacobian_matrix,
        )

    def _compute_jacobian_matrix(self, position: PositionVector) -> npt.NDArray:
        """
        Calcule la matrice jacobienne du système au point donné.

        La matrice jacobienne A(x,y,z) des équations du CRTBP est :

        A = [  0    0    0    1    0    0  ]
            [  0    0    0    0    1    0  ]
            [  0    0    0    0    0    1  ]
            [ U_xx U_xy U_xz  0    2    0  ]
            [ U_yx U_yy U_yz -2    0    0  ]
            [ U_zx U_zy U_zz  0    0    0  ]

        où U*_ij = ∂²U*/∂i∂j est la dérivée seconde du pseudo-potentiel.

        Pseudo-potentiel (unités normalisées) :
            U* = (1-μ)/r₁ + μ/r₂ + ½(x² + y²)

        Dérivées secondes :
            U*_xx = -(1-μ)(2x₁² - y² - z²)/r₁⁵ - μ(2x₂² - y² - z²)/r₂⁵ - (1-μ)/r₁³ - μ/r₂³ + 1
            U*_yy = -(1-μ)(2y² - x₁² - z²)/r₁⁵ - μ(2y² - x₂² - z²)/r₂⁵ - (1-μ)/r₁³ - μ/r₂³ + 1
            U*_zz = -(1-μ)(2z² - x₁² - y²)/r₁⁵ - μ(2z² - x₂² - y²)/r₂⁵ - (1-μ)/r₁³ - μ/r₂³
            U*_xy = -3(1-μ)x₁y/r₁⁵ - 3μx₂y/r₂⁵
            U*_xz = -3(1-μ)x₁z/r₁⁵ - 3μx₂z/r₂⁵
            U*_yz = -3(1-μ)yz/r₁⁵ - 3μyz/r₂⁵

        où :
            x₁ = x - x₁ = x + μ
            x₂ = x - x₂ = x - 1 + μ
            r₁ = √(x₁² + y² + z²)
            r₂ = √(x₂² + y² + z²)

        Args:
            position: Position [x, y, z] (normalisée)

        Returns:
            Matrice jacobienne 6×6

        Note:
            Pour les points de Lagrange colinéaires (y=0, z=0), les termes
            croisés U*_xy, U*_xz, U*_yz sont nuls.
        """
        x, y, z = position

        x_norm = x
        y_norm = y
        z_norm = z

        if not self.normalized:
            x_norm /= self.distance_unit
            y_norm /= self.distance_unit
            z_norm /= self.distance_unit

        x1_pos = -self.mu
        x2_pos = 1.0 - self.mu
        r1, r2 = distance_to_primary(
            np.array([x_norm, y_norm, z_norm, 0, 0, 0]),
            x1_pos,
            x2_pos,
            normalized=True,
        )
        x1 = x_norm - x1_pos
        x2 = x_norm - x2_pos

        if r1 <= 1e-10 or r2 <= 1e-10:
            raise ValueError("Position trop proche d'un primaire pour calcul jacobien")

        r1_3 = r1**3
        r1_5 = r1**5
        r2_3 = r2**3
        r2_5 = r2**5

        c1 = 1.0 - self.mu
        c2 = self.mu

        # Dérivées secondes du pseudo-potentiel
        # U*_xx
        U_xx = (
            -c1 * (2 * x1**2 - y_norm**2 - z_norm**2) / r1_5
            - c2 * (2 * x2**2 - y_norm**2 - z_norm**2) / r2_5
            - c1 / r1_3
            - c2 / r2_3
            + 1.0
        )

        # U*_yy
        U_yy = (
            -c1 * (2 * y_norm**2 - x1**2 - z_norm**2) / r1_5
            - c2 * (2 * y_norm**2 - x2**2 - z_norm**2) / r2_5
            - c1 / r1_3
            - c2 / r2_3
            + 1.0
        )

        # U*_zz
        U_zz = (
            -c1 * (2 * z_norm**2 - x1**2 - y_norm**2) / r1_5
            - c2 * (2 * z_norm**2 - x2**2 - y_norm**2) / r2_5
            - c1 / r1_3
            - c2 / r2_3
        )

        # U*_xy = U*_yx
        U_xy = -3 * c1 * x1 * y_norm / r1_5 - 3 * c2 * x2 * y_norm / r2_5

        # U*_xz = U*_zx
        U_xz = -3 * c1 * x1 * z_norm / r1_5 - 3 * c2 * x2 * z_norm / r2_5

        # U*_yz = U*_zy
        U_yz = -3 * c1 * y_norm * z_norm / r1_5 - 3 * c2 * y_norm * z_norm / r2_5

        # Construction de la matrice jacobienne 6×6
        A = np.zeros((6, 6), dtype=np.float64)

        # Bloc identité 3×3 en haut à droite (dérivée position = vitesse)
        A[0:3, 3:6] = np.eye(3)

        # Bloc des dérivées secondes (en bas à gauche)
        A[3, 0] = U_xx
        A[3, 1] = U_xy
        A[3, 2] = U_xz
        A[3, 4] = 2.0  # Terme de Coriolis

        A[4, 0] = U_xy
        A[4, 1] = U_yy
        A[4, 2] = U_yz
        A[4, 3] = -2.0  # Terme de Coriolis

        A[5, 0] = U_xz
        A[5, 1] = U_yz
        A[5, 2] = U_zz

        return A

    def _compute_eigenvalues(self, position: PositionVector) -> np.ndarray:
        """
        Calcule les valeurs propres de la matrice jacobienne.

        Les valeurs propres λ satisfont :
            det(A - λI) = 0

        Pour les points de Lagrange, on obtient 6 valeurs propres qui
        déterminent la stabilité :

        Points colinéaires (L1, L2, L3) :
            - 2 valeurs propres réelles : ±λ_r (mode instable)
            - 4 valeurs propres imaginaires pures : ±iλ_i1, ±iλ_i2 (modes oscillatoires)
            → INSTABLE (exponentielle croissante)

        Points triangulaires (L4, L5) :
            Si μ < μ_crit ≈ 0.0385 :
                - 6 valeurs propres imaginaires pures
                → STABLE (oscillations périodiques)
            Si μ > μ_crit :
                - 2 valeurs propres réelles
                - 4 valeurs propres imaginaires
                → INSTABLE

        Args:
            position: Position [x, y, z]

        Returns:
            Array de 6 valeurs propres complexes

        Interprétation physique :
            - Re(λ) > 0 : mode exponentiellement croissant (instable)
            - Re(λ) = 0 : mode oscillatoire (neutre)
            - Re(λ) < 0 : mode exponentiellement décroissant (stable)
        """
        A = self._compute_jacobian_matrix(position)
        eigenvalues = np.linalg.eigvals(A)
        eigenvalues = eigenvalues[np.argsort(-eigenvalues.real)]

        return eigenvalues

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
            x0 = 1.0 - (self.mu / 3.0) ** (1.0 / 3.0)
        else:
            x0 = initial_guess

        x = self._newton_raphson_collinear(
            x0,
            region="L1",
            tol=NumericalConstants.LAGRANGE_POINT_TOL,
            max_iter=NumericalConstants.MAX_ITERATIONS,
        )

        if not self.normalized:
            x *= self.distance_unit

        position = np.array([x, 0.0, 0.0], dtype=np.float64)

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
            x0 = (1.0 - self.mu) + (self.mu / 3.0) ** (1.0 / 3.0)
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

        position = np.array([x, 0.0, 0.0], dtype=np.float64)

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

        position = np.array([x, 0.0, 0.0], dtype=np.float64)

        return self._create_lagrange_point_info(LagrangePoint.L3, position)

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

        position = np.array([x, y, z], dtype=np.float64)

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

        position = np.array([x, y, z], dtype=np.float64)

        return self._create_lagrange_point_info(LagrangePoint.L5, position)

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

    def analyze_stability(self, point: LagrangePoint) -> Dict:
        """
        Analyse détaillée de la stabilité d'un point de Lagrange.

        Args:
            point: Point à analyser

        Returns:
            Dictionnaire avec :
                - 'eigenvalues': valeurs propres
                - 'stable_modes': nombre de modes stables
                - 'unstable_modes': nombre de modes instables
                - 'neutral_modes': nombre de modes neutres
                - 'dominant_timescale': échelle de temps du mode dominant (jours)
                - 'classification': description textuelle

        Exemple pour L2 (Soleil-Terre) :
            - Mode instable : τ ≈ 23 jours (document 1)
            - Modes oscillatoires : périodes ~140-200 jours
        """
        info = self.compute_lagrange_point(point)
        position = info.position

        pos_norm = position.copy()
        if not self.normalized:
            pos_norm /= self.distance_unit

        eigenvalues = info.eigenvalues

        # Analyser les valeurs propres
        tolerance = 1e-10

        stable_modes = 0
        unstable_modes = 0
        neutral_modes = 0

        real_eigenvalues = []
        imaginary_eigenvalues = []

        for lam in eigenvalues:
            real_part = lam.real
            imag_part = abs(lam.imag)

            if abs(real_part) > tolerance:
                if real_part > 0:
                    unstable_modes += 1
                else:
                    stable_modes += 1
                real_eigenvalues.append(lam)
            else:
                neutral_modes += 1
                imaginary_eigenvalues.append(lam)

        # Échelle de temps du mode dominant
        # Pour mode instable : τ = 1/|Re(λ)|
        # Pour mode oscillatoire : T = 2π/|Im(λ)|

        if len(real_eigenvalues) > 0:
            # Mode instable dominant
            max_real = max(abs(lam.real) for lam in real_eigenvalues)

            if not self.normalized:
                omega = Constants.OMEGA_EARTH
                timescale_seconds = 1.0 / (max_real * omega)
            else:
                timescale_seconds = 1.0 / max_real * (2 * np.pi / Constants.OMEGA_EARTH)

            timescale_days = timescale_seconds / 86400.0
            mode_type = "instable (exponentiel)"
        else:
            max_imag = max(abs(lam.imag) for lam in imaginary_eigenvalues)

            if not self.normalized:
                omega = Constants.OMEGA_EARTH
                timescale_seconds = 2 * np.pi / (max_imag * omega)
            else:
                timescale_seconds = (
                    2 * np.pi / max_imag * (2 * np.pi / Constants.OMEGA_EARTH)
                )

            timescale_days = timescale_seconds / 86400.0
            mode_type = "oscillatoire (période)"

        # Rapport de stabilité
        if point in [LagrangePoint.L1, LagrangePoint.L2, LagrangePoint.L3]:
            classification = (
                f"{point.value} : Point colinéaire INSTABLE\n"
                f"  - {unstable_modes} modes instables\n"
                f"  - {neutral_modes} modes oscillatoires\n"
                f"  - Échelle de temps dominante : {timescale_days:.1f} jours ({mode_type})"
            )
        else:
            if self.mu < self.mu_critical:
                classification = (
                    f"{point.value} : Point triangulaire STABLE (μ < μ_crit)\n"
                    f"  - {neutral_modes} modes oscillatoires\n"
                    f"  - Période dominante : {timescale_days:.1f} jours"
                )
            else:
                classification = (
                    f"{point.value} : Point triangulaire INSTABLE (μ > μ_crit)\n"
                    f"  - {unstable_modes} modes instables\n"
                    f"  - {neutral_modes} modes oscillatoires\n"
                    f"  - Échelle de temps : {timescale_days:.1f} jours ({mode_type})"
                )

        return {
            "eigenvalues": eigenvalues,
            "stable_modes": stable_modes,
            "unstable_modes": unstable_modes,
            "neutral_modes": neutral_modes,
            "dominant_timescale_days": timescale_days,
            "mode_type": mode_type,
            "classification": classification,
        }
