"""
simulation.py — Simulation N-corps dans le référentiel inertiel J2000.

Hypothèses :
  - Soleil fixe à l'origine (masse dominante, déplacement négligeable).
  - Terre sur orbite circulaire dans le plan XY, période = 365.25 j.
  - Lune fusionnée dans la masse Terre (barycentre Terre-Lune).
  - JWST initialisé depuis l'état Richardson converti en coordonnées inertielles.

Conversion CR3BP → inertiel (à t=0, alignement des repères) :
  r_inertiel = R_rot(t=0) · r_adim · l*  +  r_barycentre
  Le repère tournant tourne à ω = 2π / T_Terre par rapport à J2000.
  À t=0 on pose que l'axe x tournant est aligné avec l'axe x inertiel.

Unités : SI (m, kg, s).
"""

import numpy as np
from core.body import Body
from core.integrator import integrate
from .forces import gravitational_acceleration, mechanical_energy, G

# ---------------------------------------------------------------------------
# Constantes physiques SI
# ---------------------------------------------------------------------------

L_STAR: float = 1.495_978_707e11  # 1 UA [m]
M_SUN: float = 1.989e30  # [kg]
M_EARTH_MOON: float = 6.045e24  # Terre + Lune fusionnée [kg]  (≈ 5.972e24 + 7.342e22)
T_EARTH_S: float = 365.25 * 86400  # Période orbitale Terre [s]
OMEGA_EARTH: float = 2 * np.pi / T_EARTH_S  # Vitesse angulaire [rad/s]
R_EARTH_ORBIT: float = L_STAR  # Rayon orbite circulaire Terre = 1 UA


class InertialSimulation:
    """
    Simulation N-corps du JWST dans le référentiel inertiel.

    Parameters
    ----------
    state0_cr3bp : np.ndarray, shape (6,)
        État initial du JWST en coordonnées CR3BP adim. (depuis Richardson).
        Sera converti automatiquement en SI.
    n_revolutions : float
        Nombre de révolutions halo à simuler.
    T_halo_adim : float
        Période halo en unités adim. (fournie par CR3BPSimulation).
    n_steps_per_rev : int
        Nombre de pas RK4 par révolution.
    """

    def __init__(
        self,
        state0_cr3bp: np.ndarray,
        n_revolutions: float = 4.0,
        T_halo_adim: float = 3.0,  # valeur typique JWST
        n_steps_per_rev: int = 5000,
    ):
        self.n_revolutions = n_revolutions
        self.T_halo_adim = T_halo_adim
        self.n_steps_per_rev = n_steps_per_rev

        # Conversion t* → secondes
        self.T_star_s = T_EARTH_S / (2 * np.pi)  # t* [s]
        self.T_halo_s = T_halo_adim * self.T_star_s

        # Corps fixes (mis à jour dynamiquement à chaque pas)
        self.sun = Body("Soleil", M_SUN, np.zeros(3), np.zeros(3), fixed=True)
        self.earth = Body("Terre", M_EARTH_MOON, np.zeros(3), np.zeros(3), fixed=True)

        # État initial JWST converti en SI
        self.state0_si = self._cr3bp_to_inertial(state0_cr3bp)
        self.jwst = Body(
            "JWST",
            mass=6500.0,
            position=self.state0_si[:3],
            velocity=self.state0_si[3:],
        )

        # Résultats
        self.times: np.ndarray | None = None
        self.states: np.ndarray | None = None
        self.energy: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Conversion CR3BP adim. → inertiel SI  (à t=0)
    # ------------------------------------------------------------------

    def _cr3bp_to_inertial(self, state_cr3bp: np.ndarray) -> np.ndarray:
        """
        Convertit [x,y,z,vx,vy,vz] adim. CR3BP → SI inertiel à t=0.

        À t=0 :
          - Le repère tournant est aligné avec le repère inertiel.
          - La vitesse dans le repère tournant inclut le terme de Coriolis
            (ω × r) pour revenir au référentiel inertiel.
        """
        mu = M_EARTH_MOON / (M_SUN + M_EARTH_MOON)

        x_nd, y_nd, z_nd, vx_nd, vy_nd, vz_nd = state_cr3bp

        # Position SI (le barycentre est à l'origine dans le CR3BP)
        # Dans le CR3BP, le Soleil est à x = -μ, la Terre à x = 1-μ
        # On recentre : l'origine inertielle est le Soleil (approximation,
        # car M_sun >> M_earth le barycentre ≈ centre du Soleil)
        # Décalage du barycentre CR3BP par rapport au Soleil : x_bary = μ * L_STAR
        x_offset = mu * L_STAR  # ~450 km, négligeable mais inclus pour cohérence

        pos_si = np.array(
            [
                x_nd * L_STAR - x_offset,
                y_nd * L_STAR,
                z_nd * L_STAR,
            ]
        )

        # Vitesse : passage repère tournant → inertiel
        # v_inertiel = v_tournant + ω × r_tournant
        # ω = ω_earth * ẑ,  r_tournant = (x_nd, y_nd, 0) * L_STAR
        v_star = L_STAR / self.T_star_s  # unité de vitesse adim. → SI

        vx_si = vx_nd * v_star - OMEGA_EARTH * y_nd * L_STAR
        vy_si = vy_nd * v_star + OMEGA_EARTH * (x_nd * L_STAR - x_offset)
        vz_si = vz_nd * v_star

        return np.array([pos_si[0], pos_si[1], pos_si[2], vx_si, vy_si, vz_si])

    # ------------------------------------------------------------------
    # Équations de mouvement N-corps (référentiel inertiel)
    # ------------------------------------------------------------------

    def _eom(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        dy/dt pour le vecteur d'état [x, y, z, vx, vy, vz] du JWST.
        Les corps massifs (Soleil, Terre) sont mis à jour à chaque appel.
        """
        # Mise à jour de la position de la Terre sur son orbite circulaire
        theta = OMEGA_EARTH * t
        self.earth.position = R_EARTH_ORBIT * np.array(
            [np.cos(theta), np.sin(theta), 0.0]
        )
        # Soleil fixe à l'origine

        pos = state[:3]
        vel = state[3:]
        acc = gravitational_acceleration(pos, [self.sun, self.earth])

        return np.concatenate([vel, acc])

    # ------------------------------------------------------------------
    # Lancement
    # ------------------------------------------------------------------

    def run(self):
        t_end = self.n_revolutions * self.T_halo_s
        h = self.T_halo_s / self.n_steps_per_rev

        self.times, self.states = integrate(self._eom, self.state0_si, 0.0, t_end, h)

        # Énergie mécanique spécifique à chaque pas
        self.energy = np.empty(len(self.times))
        for i, (t, s) in enumerate(zip(self.times, self.states)):
            theta = OMEGA_EARTH * t
            self.earth.position = R_EARTH_ORBIT * np.array(
                [np.cos(theta), np.sin(theta), 0.0]
            )
            self.energy[i] = mechanical_energy(
                s[:3], s[3:], self.jwst.mass, [self.sun, self.earth]
            )

        # Historique du corps JWST
        self.jwst.clear_history()
        for s in self.states:
            self.jwst.position = s[:3].copy()
            self.jwst.velocity = s[3:].copy()
            self.jwst.record()

        print(self._summary())

    def _summary(self) -> str:
        assert self.energy is not None and self.T_halo_s and self.times is not None
        dE = self.energy.max() - self.energy.min()
        E0 = self.energy[0]
        lines = [
            "\n[Inertiel] Simulation terminée",
            f"  Révolutions   : {self.n_revolutions}",
            f"  Période halo  : {self.T_halo_s / 86400:.1f} jours",
            f"  Pas h         : {self.T_halo_s / self.n_steps_per_rev / 3600:.2f} h",
            f"  Pas totaux    : {len(self.times)}",
            f"  E0            : {E0:.6e} J/kg",
            f"  ΔE (dérive)   : {dE:.2e}  ({100*dE/abs(E0):.4f} %)",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Accesseurs
    # ------------------------------------------------------------------

    @property
    def positions(self) -> np.ndarray:
        """shape (N, 3) [m]"""
        assert self.states is not None
        return self.states[:, :3]

    @property
    def velocities(self) -> np.ndarray:
        """shape (N, 3) [m/s]"""
        assert self.states is not None
        return self.states[:, 3:]

    @property
    def speeds(self) -> np.ndarray:
        """shape (N,) [m/s]"""
        return np.linalg.norm(self.velocities, axis=1)

    @property
    def times_days(self) -> np.ndarray:
        assert self.times is not None
        return self.times / 86400

    def earth_positions(self) -> np.ndarray:
        """Positions de la Terre à chaque instant, shape (N, 3) [m]."""
        assert self.times is not None
        theta = OMEGA_EARTH * self.times
        return R_EARTH_ORBIT * np.column_stack(
            [np.cos(theta), np.sin(theta), np.zeros_like(theta)]
        )
