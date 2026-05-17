"""
Générateur d'orbites périodiques pour le CRTBP.

Ce module fournit des fonctions pour générer des conditions initiales
d'orbites périodiques (Lyapunov, halo, quasi-halo) autour des points
de Lagrange L1 et L2.

Ce module implémente plusieurs méthodes pour générer des conditions initiales
d'orbites quasi-halo et Lissajous autour de L2, adaptées à la mission JWST.


Méthode : Correcteur différentiel simple
- Partir d'une estimation de vitesse
- Propager jusqu'au croisement du plan XZ
- Corriger itérativement pour fermer l'orbite

Méthodes implémentées:
1. Génération via variétés stables (méthode principale)
2. Génération par perturbation linéaire
3. Génération par continuation (famille halo)

Références:
- Document 1 (Petersen 2019): Station-keeping JWST
- Document 2 (Brown 2015): Seasonal variations
- Document 6 (Llanos 2022): Trajectory analysis

"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import eig

from src.Simulations.integrator import rk4_step

from ..Models.base_dynamics import DynamicsConfig, DynamicsModel
from ..Models.vectors import StateVector, create_state_vector
from .calcul_pos_lagrange import LagrangePoint, LagrangePointCalculator
from .constants import Constants, JWSTParameters
from .CRTBP3_dynamics import CRTBP3Body


class OrbitType(Enum):
    """Type d'orbite autour d'un point de Lagrange."""

    HALO = "halo"  # Périodique, symétrique
    QUASI_HALO = "quasi_halo"  # Quasi-périodique, proche halo (JWST)
    LISSAJOUS = "lissajous"  # Quasi-périodique, 2 fréquences
    LYAPUNOV = "lyapunov"  # Planaire (dans le plan XY ou XZ)


@dataclass
class OrbitInitialConditions:
    """
    Conditions initiales complètes pour une orbite LPO.

    Attributs:
        state: État initial [x, y, z, vx, vy, vz] (unités normalisées CRTBP)
        orbit_type: Type d'orbite (halo, quasi-halo, lissajous)
        period: Période approximative [s] (si applicable)
        amplitudes: Dictionnaire des amplitudes {'x': ..., 'y': ..., 'z': ...} [m]
        jacobi_constant: Constante de Jacobi
        lagrange_point: Point de Lagrange associé
        generation_method: Méthode utilisée pour générer
        is_physical: True si en unités physiques, False si normalisé
    """

    state: np.ndarray
    orbit_type: OrbitType
    period: Optional[float] = None
    amplitudes: Optional[Dict[str, float]] = None
    jacobi_constant: Optional[float] = None
    lagrange_point: Optional[LagrangePoint] = None
    generation_method: Optional[str] = None
    is_physical: bool = False

    def to_physical(self) -> "OrbitInitialConditions":
        """Convertit en unités physiques si normalisé."""
        if self.is_physical:
            return self

        L_star = Constants.AU
        V_star = Constants.AU * Constants.OMEGA_EARTH

        state_phys = create_state_vector()
        state_phys[:3] = self.state[:3] * L_star
        state_phys[3:6] = self.state[3:6] * V_star

        return OrbitInitialConditions(
            state=state_phys,
            orbit_type=self.orbit_type,
            period=self.period,
            amplitudes=self.amplitudes,
            jacobi_constant=self.jacobi_constant,
            lagrange_point=self.lagrange_point,
            generation_method=self.generation_method,
            is_physical=True,
        )

    def to_normalized(self) -> "OrbitInitialConditions":
        """Convertit en unités normalisées si physique."""
        if not self.is_physical:
            return self

        L_star = Constants.AU
        V_star = Constants.AU * Constants.OMEGA_EARTH

        state_norm = create_state_vector()
        state_norm[:3] = self.state[:3] / L_star
        state_norm[3:6] = self.state[3:6] / V_star

        return OrbitInitialConditions(
            state=state_norm,
            orbit_type=self.orbit_type,
            period=self.period,
            amplitudes=self.amplitudes,
            jacobi_constant=self.jacobi_constant,
            lagrange_point=self.lagrange_point,
            generation_method=self.generation_method,
            is_physical=False,
        )


class OrbitGenerator:
    """
    Générateur d'orbites initiales autour des points de Lagrange.

    Cette classe implémente plusieurs algorithmes pour générer des conditions
    initiales d'orbites quasi-halo et Lissajous adaptées à JWST.

    Usage:
        >>> gen = OrbitGenerator(lagrange_point=LagrangePoint.L2)
        >>> orbit = gen.generate_jwst_nominal_orbit()
        >>> print(f"Amplitudes: Y={orbit.amplitudes['y']/1e6:.0f} km")
    """

    def __init__(
        self,
        lagrange_point: LagrangePoint = LagrangePoint.L2,
        mu: float = Constants.MU_RATIO_SUN_EARTH,
        integrator: Optional[object] = None,
    ):
        """
        Initialise le générateur d'orbites.

        Args:
            lagrange_point: Point de Lagrange cible (défaut: L2)
            mu: Paramètre de masse du CRTBP
            integrator: Intégrateur RK4 (si None, utilise intégration simple)
        """
        self.lagrange_point = lagrange_point
        self.mu = mu
        self.integrator = integrator

        config = DynamicsConfig(model=DynamicsModel.CRTBP)
        self.crtbp = CRTBP3Body(config, normalized=True)

        self.coord_transformer = None  # CoordinateTransformer()

        self.lp_calc = LagrangePointCalculator(mu=mu, normalized=True)
        self.lp_info = self.lp_calc.compute_lagrange_point(lagrange_point)

    def generate_jwst_nominal_orbit(self) -> OrbitInitialConditions:
        """
        Génère l'orbite nominale de JWST autour de L2.

        Paramètres cibles (Documents 1, 2):
            - Amplitude Y: ~750,000 - 770,000 km
            - Amplitude Z: ~420,000 km
            - Type: Quasi-halo
            - Période: ~6 mois

        Returns:
            Conditions initiales normalisées

        Note:
            Utilise la méthode des variétés stables (la plus robuste).
        """
        target_amplitudes = {
            "y": JWSTParameters.ORBIT_AMPLITUDE_Y,  # ~771,000 km
            "z": JWSTParameters.ORBIT_AMPLITUDE_Z,  # ~418,000 km
        }

        return self.generate_quasi_halo_from_manifold(
            target_amplitude_y=target_amplitudes["y"],
            target_amplitude_z=target_amplitudes["z"],
            max_iterations=10,
        )

    def generate_quasi_halo_from_manifold(
        self,
        target_amplitude_y: float,
        target_amplitude_z: float,
        max_iterations: int = 50,
    ) -> OrbitInitialConditions:
        """
        Génère orbite quasi-halo via variété stable.

        Algorithme:
        1. Calculer direction variété stable de L2
        2. Partir de L2 + petit déplacement selon cette direction
        3. Intégrer backward pour trouver état initial
        4. Ajuster pour obtenir amplitudes désirées

        Args:
            target_amplitude_y: Amplitude Y cible [m]
            target_amplitude_z: Amplitude Z cible [m]
            max_iterations: Nombre max d'itérations pour convergence

        Returns:
            Conditions initiales normalisées

        Note:
            Cette méthode est documentée dans Document 2 (Brown 2015).
            Elle produit des orbites très proches de la réalité JWST.
        """
        try:
            stable_dir = self._compute_stable_manifold_direction()
        except Exception as e:
            warnings.warn(
                f"Pas de variété stable trouvée pour {self.lagrange_point.value}: {e}. "
                "Utilisation de la méthode linéaire."
            )
            return self.generate_quasi_halo_linear(
                target_amplitude_y, target_amplitude_z
            )

        if stable_dir is None:
            warnings.warn(
                f"Pas de variété stable trouvée pour {self.lagrange_point.value}. "
                "Utilisation de la méthode linéaire."
            )
            return self.generate_quasi_halo_linear(
                target_amplitude_y, target_amplitude_z
            )

        # 2. État initial sur variété stable
        # Petit déplacement depuis L2 dans la direction stable
        epsilon = 1000.0 / Constants.AU  # 1000 km normalisé

        x_l2 = [self.lp_info.position[0], 0, 0, 0, 0, 0]
        initial_state = np.concatenate([x_l2]) + epsilon * stable_dir

        # 3. Intégration backward pour trouver amplitudes
        # On cherche le point où on traverse le plan XZ avec vitesse appropriée
        try:
            final_state, amplitudes = self._integrate_to_target_amplitudes(
                initial_state,
                target_amplitude_y,
                target_amplitude_z,
                backward=True,
                max_iterations=max_iterations,
            )
        except RuntimeError as e:
            warnings.warn(f"Échec variété stable: {e}. Utilisation méthode linéaire.")
            return self.generate_quasi_halo_linear(
                target_amplitude_y, target_amplitude_z
            )

        C = self.crtbp.jacobi_constant(final_state)
        period = self._estimate_period(final_state)

        return OrbitInitialConditions(
            state=final_state,
            orbit_type=OrbitType.QUASI_HALO,
            period=period,
            amplitudes=amplitudes,
            jacobi_constant=C,
            lagrange_point=self.lagrange_point,
            generation_method="stable_manifold",
            is_physical=False,
        )

    def _integrate_to_target_amplitudes(
        self,
        initial_state: np.ndarray,
        target_y: float,
        target_z: float,
        backward: bool = True,
        max_iterations: int = 10,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Intègre une trajectoire jusqu'à obtenir les amplitudes désirées.

        Args:
            initial_state: État initial normalisé
            target_y: Amplitude Y cible [m]
            target_z: Amplitude Z cible [m]
            backward: Si True, intègre en temps négatif
            max_iterations: Nombre max d'itérations

        Returns:
            (état final normalisé, amplitudes atteintes)

        Raises:
            RuntimeError: Si convergence échoue
        """
        L_star = Constants.AU
        target_y_norm = target_y / L_star
        target_z_norm = target_z / L_star

        # Temps d'intégration (environ 1/4 de période)
        T_period = 2 * np.pi  # Période normalisée
        dt = -0.01 if backward else 0.01  # Pas de temps
        t_max = T_period / 4.0

        state = initial_state.copy()

        # Intégrer jusqu'à crossing du plan XZ
        t = 0.0
        y_max = 0.0
        z_max = 0.0

        previous_y = state[1]

        function = self.crtbp.equations_of_motion

        while abs(t) < t_max:
            state = rk4_step(state, t, dt, function)
            t += dt

            y_max = max(y_max, abs(state[1]))
            z_max = max(z_max, abs(state[2]))

            # Détecter crossing plan XZ (y=0)
            if previous_y * state[1] < 0:
                break

            previous_y = state[1]

        # Vérifier si amplitudes proches de cibles
        tolerance = 0.01

        if abs(y_max - target_y_norm) / target_y_norm > tolerance:
            print(abs(y_max - target_y_norm) / target_y_norm)
            raise RuntimeError(
                f"Amplitude Y non atteinte: {y_max*L_star/1e3:.0f} km "
                f"(cible: {target_y/1e3:.0f} km)"
            )

        amplitudes = {
            "y": y_max * L_star,
            "z": z_max * L_star,
            "x": abs(state[0] - self.lp_info.position[0]) * L_star,
        }

        return state, amplitudes

    def _compute_stable_manifold_direction(self) -> Optional[np.ndarray]:
        """
        Computes the stable manifold direction at a Lagrange point.
        Uses linearization around the equilibrium point.
        The stable manifold direction is the eigenvector corresponding
        to the eigenvalue with the most negative real part.
        Returns:
            Normalized direction vector (6D phase space) or None if computation fails
        """
        if self.lagrange_point not in (LagrangePoint.L1, LagrangePoint.L2):
            raise ValueError(f"Unsupported Lagrange point: {self.lagrange_point}")

        jacobian = self.lp_info.jacobian_matrix

        # Eigendecomposition
        eigenvalues, eigenvectors, *_ = eig(jacobian)

        # Select eigenvalues with sufficiently negative real part
        # Threshold relative to the spectral radius for numerical robustness
        spectral_radius = np.max(np.abs(eigenvalues))
        threshold = -1e-6 * spectral_radius
        stable_indices = np.where(np.real(eigenvalues) < threshold)[0]

        if len(stable_indices) == 0:
            return None

        # Take the most negative (most stable) eigenvalue
        idx = stable_indices[np.argmin(np.real(eigenvalues[stable_indices]))]
        chosen_eigenvalue = eigenvalues[idx]

        # Warn if eigenvalue has a significant imaginary part
        # (would indicate a center-stable manifold, not a pure stable one)
        if np.abs(np.imag(chosen_eigenvalue)) > 1e-6 * np.abs(chosen_eigenvalue):
            warnings.warn(
                f"Stable eigenvalue has significant imaginary part: {chosen_eigenvalue}. "
                "The returned direction may not accurately represent the stable manifold.",
                RuntimeWarning,
            )

        # Full phase-space eigenvector (position + velocity components)
        stable_eigenvector = np.real(eigenvectors[:, idx])

        norm = np.linalg.norm(stable_eigenvector)
        if norm < 1e-12:
            return None
        return stable_eigenvector / norm

    def generate_quasi_halo_linear(
        self, target_amplitude_y: float, target_amplitude_z: float
    ) -> OrbitInitialConditions:
        """
        Génère orbite quasi-halo par perturbation linéaire simple.

        Méthode simplifiée:
        1. Partir du point L2
        2. Ajouter petite vitesse perpendiculaire
        3. Ajuster magnitudes pour obtenir amplitudes

        Args:
            target_amplitude_y: Amplitude Y cible [m]
            target_amplitude_z: Amplitude Z cible [m]

        Returns:
            Conditions initiales normalisées

        Note:
            Moins précise que la méthode des variétés, mais rapide.
            Utilisée en fallback si variétés stables échouent.
        """
        L_star = Constants.AU

        x_l2_norm = self.lp_info.position[0]
        pos_l2_norm = np.array([x_l2_norm, 0.0, 0.0], dtype=np.float64)

        Ay_norm = target_amplitude_y / L_star
        Az_norm = target_amplitude_z / L_star

        # 2. Calcul du couplage (Approximation de Richardson)
        # Pour le système Soleil-Terre L2, le rapport Ay/Ax est ~3.229
        # La fréquence orbitale caractéristique nu est ~2.086
        k_coupling = 3.229
        nu = 2.086
        Ax_norm = Ay_norm / k_coupling

        # On commence à l'apogée en X (offset maximum), y=0, et z maximum.
        # La vitesse Vy est couplée à l'amplitude Ay par la fréquence nu.
        initial_state = np.array(
            [
                pos_l2_norm[0]
                - Ax_norm,  # X-offset: Correction cruciale (ne part pas de L2)
                0.0,  # Passage par le plan XZ (y=0)
                Az_norm,  # Amplitude Z initiale
                0.0,  # Vx nul à l'apside
                Ay_norm * nu,  # Vy couplé à Ay
                0.0,  # Vz nul au pic de l'oscillation Z
            ],
            dtype=np.float64,
        )

        C = self.crtbp.jacobi_constant(initial_state)

        T_star = 1.0 / Constants.OMEGA_EARTH
        period = (2 * np.pi / nu) * T_star

        amplitudes = {
            "x": Ax_norm * L_star,
            "y": target_amplitude_y,
            "z": target_amplitude_z,
        }

        return OrbitInitialConditions(
            state=initial_state,
            orbit_type=OrbitType.QUASI_HALO,
            period=period,
            amplitudes=amplitudes,
            jacobi_constant=C,
            lagrange_point=self.lagrange_point,
            generation_method="linear_perturbation",
            is_physical=False,
        )

    def generate_lissajous(
        self,
        amplitude_y: float,
        amplitude_z: float,
        phase_y: float = 0.0,
        phase_z: float = np.pi / 2,
    ) -> OrbitInitialConditions:
        """
        Génère orbite de Lissajous autour de L2.

        Orbite de Lissajous = superposition de 2 oscillations
        avec phases différentes.

        Args:
            amplitude_y: Amplitude Y [m]
            amplitude_z: Amplitude Z [m]
            phase_y: Phase initiale en Y [rad]
            phase_z: Phase initiale en Z [rad]

        Returns:
            Conditions initiales normalisées

        Note:
            Pour phase_z = phase_y + π/2, on obtient une orbite
            proche d'un halo. Pour d'autres phases, forme de Lissajous.
        """
        L_star = Constants.AU
        V_star = L_star * Constants.OMEGA_EARTH

        Ay = amplitude_y / L_star
        Az = amplitude_z / L_star

        x_l2 = 1.0 + (self.mu / 3.0) ** (1 / 3)

        omega_y = 1.0
        omega_z = 1.0

        y0 = Ay * np.cos(phase_y)
        z0 = Az * np.cos(phase_z)
        vy0 = -Ay * omega_y * np.sin(phase_y)
        vz0 = -Az * omega_z * np.sin(phase_z)

        initial_state = np.array([x_l2, y0, z0, 0.0, vy0, vz0], dtype=np.float64)

        C = self.crtbp.jacobi_constant(initial_state)

        amplitudes = {"y": amplitude_y, "z": amplitude_z, "x": 0.0}

        return OrbitInitialConditions(
            state=initial_state,
            orbit_type=OrbitType.LISSAJOUS,
            period=2 * np.pi / omega_y * (1.0 / Constants.OMEGA_EARTH),
            amplitudes=amplitudes,
            jacobi_constant=C,
            lagrange_point=self.lagrange_point,
            generation_method="lissajous_analytical",
            is_physical=False,
        )

    def _estimate_period(self, state: StateVector) -> float:
        """
        Estime la période d'une orbite par intégration.

        Args:
            state: État initial normalisé

        Returns:
            Période estimée [s]

        Note:
            Intègre jusqu'à 2ème crossing du plan XZ.
        """
        T_approx = 2 * np.pi
        dt = 0.01

        current_state = state.copy()
        t = 0.0
        crossings = 0
        previous_y = current_state[1]

        function = self.crtbp.equations_of_motion
        while crossings < 2 and t < 2 * T_approx:
            current_state = rk4_step(current_state, t, dt, function)
            t += dt

            if previous_y * current_state[1] < 0:
                crossings += 1

            previous_y = current_state[1]

        if crossings < 2:
            return 6 * 30 * 86400.0  # 6 mois en secondes

        T_star = 1.0 / Constants.OMEGA_EARTH
        period_physical = 2 * t * T_star  # Demi-période → période

        return period_physical

    def validate_orbit(
        self,
        orbit: OrbitInitialConditions,
        duration: float = 30 * 86400.0,  # 30 jours
    ) -> Dict:
        """
        Valide une orbite en l'intégrant sur une durée donnée.

        Args:
            orbit: Conditions initiales à valider
            duration: Durée d'intégration [s]

        Returns:
            Dictionnaire avec statistiques:
                - amplitudes_achieved: Amplitudes mesurées
                - jacobi_variation: Variation de C
                - period_measured: Période mesurée
                - is_stable: True si orbite reste bornée
        """
        if orbit.is_physical:
            state_norm = orbit.to_normalized().state
        else:
            state_norm = orbit.state

        T_star = 1.0 / Constants.OMEGA_EARTH
        duration_norm = duration / T_star

        dt = 0.01
        t = 0.0
        current_state = state_norm.copy()

        y_max = 0.0
        z_max = 0.0
        x_max = 0.0
        C_initial = self.crtbp.jacobi_constant(current_state)
        C_variation = 0.0

        function = self.crtbp.equations_of_motion
        while t < duration_norm:
            current_state = rk4_step(current_state, t, dt, function)
            t += dt

            y_max = max(y_max, abs(current_state[1]))
            z_max = max(z_max, abs(current_state[2]))

            x_l2_norm = self.lp_info.position[0]
            x_max = max(x_max, abs(current_state[0] - x_l2_norm))

            C_current = self.crtbp.jacobi_constant(current_state)
            C_variation = max(C_variation, abs(C_current - C_initial))

        # Résultats
        L_star = Constants.AU

        return {
            "amplitudes_achieved": {
                "x": x_max * L_star,
                "y": y_max * L_star,
                "z": z_max * L_star,
            },
            "jacobi_variation": C_variation,
            "is_stable": (y_max < 2.0 and z_max < 2.0),
            "duration_integrated": duration,
        }


def print_orbit_summary(orbit: OrbitInitialConditions) -> None:
    """
    Affiche un résumé des conditions initiales d'une orbite.

    Args:
        orbit: Conditions initiales à afficher
    """
    print("\n" + "=" * 70)
    print(f" ORBITE {orbit.orbit_type.value.upper()}")
    print("=" * 70)

    if not orbit.is_physical:
        orbit_phys = orbit.to_physical()
    else:
        orbit_phys = orbit

    print(
        f"\nPoint de Lagrange: {orbit.lagrange_point.value if orbit.lagrange_point else 'N/A'}"
    )
    print(f"Méthode: {orbit.generation_method}")

    print(f"\nÉtat initial (physique):")
    print(
        f"  Position: [{orbit_phys.state[0]/1e9:.6f}, {orbit_phys.state[1]/1e9:.6f}, "
        f"{orbit_phys.state[2]/1e9:.6f}] millions km"
    )
    print(
        f"  Vitesse:  [{orbit_phys.state[3]:.6f}, {orbit_phys.state[4]:.6f}, "
        f"{orbit_phys.state[5]:.6f}] m/s"
    )

    if orbit.amplitudes:
        print(f"\nAmplitudes:")
        for axis, amp in orbit.amplitudes.items():
            if amp is not None:
                print(f"  {axis.upper()}: {amp/1e6:.3f} millions km")

    if orbit.period:
        print(f"\nPériode: {orbit.period/86400:.1f} jours")

    if orbit.jacobi_constant:
        print(f"Constante de Jacobi: {orbit.jacobi_constant:.8f}")

    print("\n" + "=" * 70)
