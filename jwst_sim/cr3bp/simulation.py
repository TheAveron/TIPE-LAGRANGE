"""
simulation.py — Simulation complète dans le CR3BP.

Orchestre l'intégration RK4 sur l'état du JWST dans le repère tournant,
en enregistrant position, vitesse, temps et constante de Jacobi.
"""

import numpy as np
from core.body import Body
from core.integrator import integrate
from numpy.typing import NDArray

from .equations import MU_SUN_EARTH, eom_factory, jacobi_constant
from .lagrange import lagrange_L2, richardson_halo_L2

# Facteur de conversion unité de temps adim → secondes
# t* = sqrt(l*³ / (G m*))  avec l* = 1 UA, G m* ≈ G(M_sun + M_earth) ≈ G M_sun
# T_orb_Terre = 2π t*  → t* = T_orb / 2π = 365.25 * 86400 / 2π
T_STAR_SEC = np.float64(365.25 * 86400 / (2 * np.pi))  # ≈ 5.023e6 s ≈ 58.15 jours
ZERO = np.float64(0)


class CR3BPSimulation:
    """
    Simulation du JWST dans le repère tournant du CR3BP.

    Parameters
    ----------
    Az : np.float64
        Amplitude hors-plan de l'orbite halo [adim.].
    mu : np.float64
        Paramètre de masse.
    n_revolutions : np.int16
        Nombre de révolutions halo à simuler.
    n_steps_per_rev : np.int16
        Nombre de pas RK4 par révolution (qualité de l'intégration).
    northern : bool
        Halo nord (True) ou sud (False).
    phi : np.float64
        Phase initiale [rad].
    """

    def __init__(
        self,
        Az: np.float64 = np.float64(0.00279),
        mu: np.float64 = MU_SUN_EARTH,
        n_revolutions: np.int16 = np.int16(4),
        n_steps_per_rev: np.int16 = np.int16(5000),
        northern: bool = True,
        phi: np.float64 = ZERO,
    ):
        self.Az = Az
        self.mu = mu
        self.n_revolutions = n_revolutions
        self.n_steps_per_rev = n_steps_per_rev
        self.northern = northern
        self.phi = phi

        # Résultats (remplis après run())
        self.times: NDArray[np.float64] | None = None  # adim.
        self.states: NDArray[np.float64] | None = None  # shape (N, 6)
        self.jacobi: NDArray[np.float64] | None = None  # shape (N,)
        self.T_halo: np.float64 | None = None  # période adim.
        self.state0: NDArray[np.float64] | None = None  # état initial

        # Corps JWST (pour compatibilité avec le module visualization)
        self.jwst = Body(
            "JWST",
            mass=np.float64(6500),
            position=np.zeros(3, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
        )

    def run(self):
        """Lance l'intégration."""
        state0, T_half, c2 = richardson_halo_L2(
            self.Az, self.mu, self.northern, self.phi
        )
        self.T_halo = 2 * T_half
        self.state0 = state0

        t_end = self.n_revolutions * self.T_halo
        h = self.T_halo / self.n_steps_per_rev

        f = eom_factory(self.mu)
        self.times, self.states = integrate(f, state0, ZERO, t_end, h)
        print(np.mean(np.linalg.norm(self.states[:, 3:], axis=1)))

        # Constante de Jacobi à chaque pas
        self.jacobi = np.array(
            [jacobi_constant(self.states[i], self.mu) for i in range(len(self.times))],
            dtype=np.float64,
        )

        # Mise à jour du corps JWST avec l'historique
        self.jwst.clear_history()
        for i in range(len(self.times)):
            self.jwst.position = self.states[i, :3]
            self.jwst.velocity = self.states[i, 3:]
            self.jwst.record()

        print(self._summary())

    def _summary(self) -> str:
        assert self.jacobi is not None and self.T_halo and self.times is not None
        dJ = self.jacobi.max() - self.jacobi.min()
        J0 = self.jacobi[0]
        lines = [
            "\n[CR3BP] Simulation terminée",
            f"  Révolutions   : {self.n_revolutions}",
            f"  Période halo  : {self.T_halo:.4f} adim "
            f"= {self.T_halo * T_STAR_SEC / 86400:.1f} jours",
            f"  Pas h         : {self.T_halo / self.n_steps_per_rev:.2e} adim",
            f"  Pas totaux    : {len(self.times)}",
            f"  Jacobi C0     : {J0:.6f}",
            f"  ΔC (dérive)   : {dJ:.2e}  ({100*dJ/abs(J0):.4f} %)",
        ]
        return "\n".join(lines)

    # Accesseurs pratiques

    @property
    def positions(self) -> NDArray[np.float64]:
        """shape (N, 3) — positions adim. dans le repère tournant."""
        assert self.states is not None
        return self.states[:, :3]

    @property
    def velocities(self) -> NDArray[np.float64]:
        """shape (N, 3) — vitesses adim."""
        assert self.states is not None
        return self.states[:, 3:]

    @property
    def speeds(self) -> NDArray[np.float64]:
        """Norme de la vitesse, shape (N,)."""
        return np.linalg.norm(self.velocities, axis=1)

    @property
    def times_days(self) -> NDArray[np.float64]:
        """Temps en jours."""
        assert self.times is not None
        return self.times * T_STAR_SEC / 86400

    def L2_position(self) -> NDArray[np.float64]:
        """Position de L2 dans le repère tournant [adim.]."""
        return lagrange_L2(self.mu)
