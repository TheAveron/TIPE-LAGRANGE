"""
Module de calcul des points de Lagrange.

Ce module calcule les positions exactes des 5 points de Lagrange
dans le problème restreint à trois corps (CRTBP).

Ce module implémente :
- Calcul des 5 points de Lagrange (L1-L5) dans le CRTBP
- Analyse de stabilité (valeurs propres, vecteurs propres)
- Classification des points (stable/instable/selle)

Références :
- Document 4 (2000lag.pdf) : Formules analytiques des points de Lagrange
- Document 7 (UTF-8lagrange_theorie.pdf) : Théorie détaillée
- Szebehely (1967) : Theory of Orbits
"""

import numpy as np
from typing import Optional, Dict, NamedTuple
from dataclasses import dataclass
from enum import Enum

from src.simulation.coordinates import StateVector, distance_to_primary


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


class LagrangePointInfo(NamedTuple):
    """Information complète sur un point de Lagrange.

    Attributes:
        position: Position [x, y, z] en unités normalisées CRTBP
        name: Nom du point (L1, L2, etc.)
        eigenvalues: Valeurs propres de la matrice jacobienne (6 valeurs complexes)
        eigenvectors: Vecteurs propres correspondants (6x6)
        stability_type: Type de stabilité ('unstable', 'saddle', 'stable')
        is_collinear: True si point colinéaire (L1, L2, L3)
        jacobi_constant: Constante de Jacobi au point
    """

    position: np.ndarray
    name: LagrangePoint
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    stability_type: str
    is_collinear: bool
    jacobi_constant: float


@dataclass
class LagrangePointConfig:
    """Configuration pour le calcul des points de Lagrange.

    Attributes:
        mu: Paramètre de masse du CRTBP (M2 / (M1 + M2))
        tolerance: Tolérance pour les solveurs Newton-Raphson
        max_iterations: Nombre maximum d'itérations
        use_high_order: Si True, utilise les développements d'ordre supérieur
    """

    mu: float
    distance_unit: float
    tolerance: float = 1e-12
    max_iterations: int = 100
    use_high_order: bool = True

    # Valeur critique de μ pour stabilité de L4, L5
    # μ_crit = (1 - √(23/27)) / 2 ≈ 0.0385
    mu_critical = 0.5 * (1.0 - np.sqrt(23.0 / 27.0))


class LagrangePointCalculator:
    """Calculateur des points de Lagrange dans le CRTBP.

    Cette classe calcule les positions et analyse la stabilité des 5 points
    de Lagrange dans le problème restreint des trois corps circulaire.

    Les points colinéaires (L1, L2, L3) sont calculés par Newton-Raphson.
    Les points triangulaires (L4, L5) ont des positions analytiques.

    Attributes:
        config: Configuration du calculateur
        mu: Paramètre de masse μ = M2/(M1+M2)
    """

    def __init__(self, config: LagrangePointConfig):
        """Initialise le calculateur.

        Args:
            config: Configuration contenant μ et paramètres numériques

        Raises:
            ValueError: Si μ n'est pas dans [0, 1]
        """
        if not 0 <= config.mu <= 1:
            raise ValueError(f"μ doit être dans [0, 1], reçu : {config.mu}")

        self.config = config
        self.mu = config.mu

    def compute_l1(self) -> np.ndarray:
        """Calcule la position du point L1.

        L1 est situé entre les deux primaires. Position approximative :
        x_L1 ≈ 1 - μ - (μ/3)^(1/3)

        Returns:
            Position [x, 0, 0] en coordonnées CRTBP normalisées
        """
        # Première approximation (valable pour μ << 1)
        x0 = 1 - self.mu - (self.mu / 3) ** (1 / 3)

        # Affinement par Newton-Raphson
        x_l1 = self._newton_raphson_collinear(x0, point="L1")

        return np.array([x_l1, 0.0, 0.0])

    def compute_l2(self) -> np.ndarray:
        """Calcule la position du point L2.

        L2 est situé au-delà de M2 (Terre). Position approximative :
        x_L2 ≈ 1 - μ + (μ/3)^(1/3)

        Returns:
            Position [x, 0, 0] en coordonnées CRTBP normalisées
        """
        x0 = 1 - self.mu + (self.mu / 3) ** (1 / 3)
        x_l2 = self._newton_raphson_collinear(x0, point="L2")
        return np.array([x_l2, 0.0, 0.0])

    def compute_l3(self) -> np.ndarray:
        """Calcule la position du point L3.

        L3 est situé à l'opposé de M2 par rapport à M1.
        Position approximative : x_L3 ≈ -1 - 5μ/12

        Returns:
            Position [x, 0, 0] en coordonnées CRTBP normalisées
        """
        x0 = -1 - 5 * self.mu / 12
        x_l3 = self._newton_raphson_collinear(x0, point="L3")

        return np.array([x_l3, 0.0, 0.0])

    def compute_l4(self) -> np.ndarray:
        """Calcule la position du point L4.

        L4 forme un triangle équilatéral avec M1 et M2.
        Position exacte : (1/2 - μ, √3/2, 0)

        Returns:
            Position [x, y, z] en coordonnées CRTBP normalisées
        """
        x = 0.5 - self.mu
        y = np.sqrt(3) / 2

        return np.array([x, y, 0.0])

    def compute_l5(self) -> np.ndarray:
        """Calcule la position du point L5.

        L5 forme un triangle équilatéral avec M1 et M2 (symétrique de L4).
        Position exacte : (1/2 - μ, -√3/2, 0)

        Returns:
            Position [x, y, z] en coordonnées CRTBP normalisées
        """
        x = 0.5 - self.mu
        y = -np.sqrt(3) / 2

        return np.array([x, y, 0.0])

    def _newton_raphson_collinear(self, x0: float, point: str) -> float:
        """Résout pour un point de Lagrange colinéaire par Newton-Raphson.

        Résout l'équation : U_x(x, 0, 0) = x
        où U est le potentiel effectif du CRTBP.

        Args:
            x0: Estimation initiale de x
            point: Nom du point ('L1', 'L2', ou 'L3')

        Returns:
            Position x du point de Lagrange

        Raises:
            RuntimeError: Si la convergence échoue
        """

        def equation(x: float) -> float:
            """Équation à résoudre : ∂U*/∂x - x = 0"""
            r1 = abs(x + self.mu)
            r2 = abs(x - 1 + self.mu)

            if r1 < 1e-15 or r2 < 1e-15:
                return np.inf

            return (
                x
                - (1 - self.mu) * (x + self.mu) / r1**3
                - self.mu * (x - 1 + self.mu) / r2**3
            )

        def derivative(x: float) -> float:
            """Dérivée de l'équation"""
            r1 = abs(x + self.mu)
            r2 = abs(x - 1 + self.mu)

            if r1 < 1e-15 or r2 < 1e-15:
                return np.inf

            term1 = (1 - self.mu) / r1**3
            term2 = self.mu / r2**3
            term3 = 3 * (1 - self.mu) * (x + self.mu) ** 2 / r1**5
            term4 = 3 * self.mu * (x - 1 + self.mu) ** 2 / r2**5

            return 1 - term1 - term2 + term3 + term4

        # Méthode de Newton-Raphson
        x = x0
        for i in range(self.config.max_iterations):
            f = equation(x)
            fp = derivative(x)

            if abs(f) < self.config.tolerance:
                return x

            if abs(fp) < 1e-15:
                raise RuntimeError(f"Dérivée nulle pour {point}")

            x_new = x - f / fp

            if abs(x_new - x) < self.config.tolerance:
                return x_new

            x = x_new

        raise RuntimeError(
            f"Convergence échouée pour {point} après {self.config.max_iterations} itérations"
        )

    def _compute_jacobian_matrix(self, pos: np.ndarray) -> np.ndarray:
        """Calcule la matrice jacobienne du système linéarisé.

        Pour le CRTBP, la matrice est 6×6 et a la forme :

        J = [  0₃   I₃  ]
            [ Uxx  2Ω  ]

        où Uxx est la matrice hessienne du potentiel effectif et
        Ω est la matrice de Coriolis.

        Args:
            pos: Position [x, y, z] où calculer la jacobienne

        Returns:
            Matrice jacobienne 6×6
        """
        x, y, z = pos

        # Distances aux primaires
        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu) ** 2 + y**2 + z**2)

        # Dérivées secondes du potentiel effectif U*
        # U* = 1/2(x² + y²) + (1-μ)/r1 + μ/r2 + 1/2·μ(1-μ)

        r1_3 = r1**3
        r1_5 = r1**5
        r2_3 = r2**3
        r2_5 = r2**5

        # Composantes de la matrice hessienne
        Uxx = (
            1
            - (1 - self.mu) / r1_3
            - self.mu / r2_3
            + 3 * (1 - self.mu) * (x + self.mu) ** 2 / r1_5
            + 3 * self.mu * (x - 1 + self.mu) ** 2 / r2_5
        )

        Uyy = (
            1
            - (1 - self.mu) / r1_3
            - self.mu / r2_3
            + 3 * (1 - self.mu) * y**2 / r1_5
            + 3 * self.mu * y**2 / r2_5
        )

        Uzz = (
            -(1 - self.mu) / r1_3
            - self.mu / r2_3
            + 3 * (1 - self.mu) * z**2 / r1_5
            + 3 * self.mu * z**2 / r2_5
        )

        Uxy = (
            3 * (1 - self.mu) * (x + self.mu) * y / r1_5
            + 3 * self.mu * (x - 1 + self.mu) * y / r2_5
        )

        Uxz = (
            3 * (1 - self.mu) * (x + self.mu) * z / r1_5
            + 3 * self.mu * (x - 1 + self.mu) * z / r2_5
        )

        Uyz = 3 * (1 - self.mu) * y * z / r1_5 + 3 * self.mu * y * z / r2_5

        # Construction de la matrice jacobienne 6×6
        J = np.zeros((6, 6))

        J[0:3, 3:6] = np.eye(3)

        J[3, 0] = Uxx
        J[3, 1] = Uxy
        J[3, 2] = Uxz
        J[4, 0] = Uxy
        J[4, 1] = Uyy
        J[4, 2] = Uyz
        J[5, 0] = Uxz
        J[5, 1] = Uyz
        J[5, 2] = Uzz

        J[3, 4] = 2.0
        J[4, 3] = -2.0

        return J

    def _analyze_stability(self, eigenvalues: np.ndarray) -> str:
        """Détermine le type de stabilité d'après les valeurs propres.

        Classification :
        - 'stable' : Toutes les valeurs propres ont Re(λ) ≤ 0
        - 'unstable' : Au moins une valeur propre avec Re(λ) > 0
        - 'saddle' : Mélange de valeurs propres stables et instables

        Args:
            eigenvalues: Valeurs propres (6 complexes)

        Returns:
            Type de stabilité ('stable', 'unstable', 'saddle')
        """
        real_parts = eigenvalues.real

        tol = 1e-10

        n_positive = np.sum(real_parts > tol)
        n_negative = np.sum(real_parts < -tol)

        if n_positive == 0:
            return "stable"
        elif n_negative == 0:
            return "unstable"
        else:
            return "saddle"

    def _compute_jacobi_constant(self, state: StateVector) -> float:
        """Calcule la constante de Jacobi au point donné.

        C = 2U*(x,y,z) - v² où v=0 au point de Lagrange

        Args:
            pos: Position [x, y, z]

        Returns:
            Valeur de la constante de Jacobi
        """
        r1, r2 = distance_to_primary(state, self.mu, 1 - self.mu)
        x, y = state[:2]

        # Potentiel effectif
        U_star = (
            0.5 * (x**2 + y**2)
            + (1 - self.mu) / r1
            + self.mu / r2
            + 0.5 * self.mu * (1 - self.mu)
        )

        C = 2 * U_star

        return C

    def _create_lagrange_point_info(
        self, position: np.ndarray, name: LagrangePoint, is_collinear: bool
    ) -> LagrangePointInfo:
        """Crée l'information complète pour un point de Lagrange.

        Args:
            position: Position du point [x, y, z]
            name: Nom du point (L1, L2, etc.)
            is_collinear: True si point colinéaire

        Returns:
            LagrangePointInfo contenant toutes les informations
        """
        J = self._compute_jacobian_matrix(position)

        eigenvalues, eigenvectors = np.linalg.eig(J)

        idx = np.argsort(-eigenvalues.real)  # type: ignore
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        stability = self._analyze_stability(eigenvalues)
        C = self._compute_jacobi_constant(position)

        return LagrangePointInfo(
            position=position,
            name=name,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            stability_type=stability,
            is_collinear=is_collinear,
            jacobi_constant=C,
        )

    def compute_all_lagrange_points(self) -> Dict[LagrangePoint, LagrangePointInfo]:
        """Calcule tous les points de Lagrange avec analyse complète.

        Returns:
            Dictionnaire {nom: LagrangePointInfo} pour L1-L5
        """
        points = {}

        for name, compute_func in [
            (LagrangePoint.L1, self.compute_l1),
            (LagrangePoint.L2, self.compute_l2),
            (LagrangePoint.L3, self.compute_l3),
        ]:
            pos = compute_func()
            points[name] = self._create_lagrange_point_info(
                pos, name, is_collinear=True
            )

        for name, compute_func in [
            (LagrangePoint.L4, self.compute_l4),
            (LagrangePoint.L5, self.compute_l5),
        ]:
            pos = compute_func()
            points[name] = self._create_lagrange_point_info(
                pos, name, is_collinear=False
            )

        return points

    def get_stable_manifold_direction(
        self, point_name: LagrangePoint
    ) -> Optional[np.ndarray]:
        """Retourne la direction du vecteur propre stable (pour station-keeping).

        Cette méthode extrait les composantes position du vecteur propre
        associé à la valeur propre stable (λ < 0 avec |λ| maximal).

        Args:
            point_name: Nom du point ('L1', 'L2', etc.)

        Returns:
            Vecteur direction [x, y, z] normalisé, ou None si pas de mode stable
        """
        info = self.compute_all_lagrange_points()[point_name]

        # Trouver la valeur propre stable avec la plus grande |Re(λ)| < 0
        stable_idx = None
        max_magnitude = 0.0

        for i, lam in enumerate(info.eigenvalues):
            if lam.real < 0 and abs(lam.real) > max_magnitude:
                max_magnitude = abs(lam.real)
                stable_idx = i

        if stable_idx is None:
            return None

        direction = info.eigenvectors[:3, stable_idx].real
        return direction / np.linalg.norm(direction)


def print_lagrange_points_summary(
    points: Dict[str, LagrangePointInfo], mu: float, verbose: bool = True
) -> None:
    """Affiche un résumé des points de Lagrange calculés.

    Args:
        points: Dictionnaire des points de Lagrange
        mu: Paramètre de masse
        verbose: Si True, affiche les valeurs propres
    """
    print("\n" + "=" * 70)
    print(f"POINTS DE LAGRANGE - SYSTÈME SOLEIL-TERRE (μ = {mu:.6e})")
    print("=" * 70)

    for name in ["L1", "L2", "L3", "L4", "L5"]:
        info = points[name]
        pos = info.position

        print(f"\n{name}:")
        print(f"  Position: ({pos[0]:+.10f}, {pos[1]:+.10f}, {pos[2]:+.10f})")
        print(f"  Type: {'Colinéaire' if info.is_collinear else 'Triangulaire'}")
        print(f"  Stabilité: {info.stability_type.upper()}")
        print(f"  Constante Jacobi: C = {info.jacobi_constant:.10f}")

        if verbose:
            print(f"  Valeurs propres:")
            for i, lam in enumerate(info.eigenvalues):
                if abs(lam.imag) < 1e-10:
                    print(f"    λ{i+1} = {lam.real:+.6f}")
                else:
                    print(f"    λ{i+1} = {lam.real:+.6f} {lam.imag:+.6f}i")

    print("\n" + "=" * 70)
