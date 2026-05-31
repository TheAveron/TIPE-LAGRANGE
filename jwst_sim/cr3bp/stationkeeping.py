"""
stationkeeping.py — Algorithme EVSK avec vecteurs propres propagés par la STM.

Problème fondamental sans propagation de la STM :
   La valeur propre stable λ_s ≈ -0.261 (négative) implique que la direction
   stable v_s(t) change de signe à chaque demi-période (~73 jours).
   Utiliser v_s(0) pour toutes les manœuvres revient à corriger dans la
   mauvaise direction après t ≈ T/2 → amplification de l'erreur.

Correction : à chaque manœuvre en t_k, les vecteurs propres sont propagés :
   v_s_local      =  Φ(t_k, 0) · v_s(0)               [normalisé L2]
   v_u_left_local = (Φ(t_k, 0)^{-T}) · v_u_left(0)    [biorthonormalisé]

La STM Φ(t_k, 0) est obtenue par interpolation depuis le tableau pré-calculé
lors de la propagation de référence (système augmenté 42 composantes).

Algorithme EVSK (Petersen AAS 19-806) :
   δx = x_réel(t_k) - x_réf(t_k)
   ΔV = α · ê_v  avec  α = -⟨δx, v_u_left_local⟩ / ⟨[0,0,0,ê_v], v_u_left_local⟩
   ê_v = composantes vitesse de v_s_local, normalisées.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from core.integrator import integrate, rk4_step
from numpy.typing import NDArray

from .equations import MU_SUN_EARTH, eom_factory, jacobi_constant
from .lagrange import richardson_halo_L2
from .stm import (
    correct_halo,
    eom_stm_factory,
    print_monodromy_summary,
    stable_unstable_eigvecs,
)

T_STAR_SEC = np.float64(365.25 * 86400 / (2 * np.pi))
V_STAR_MS = np.float64(1.496e11 / T_STAR_SEC)
ZERO = np.float64(0)

# Structure d'une manœuvre


@dataclass
class Maneuver:
    t_adim: np.float64
    t_days: np.float64
    delta_v: NDArray[np.float64]
    dv_norm: np.float64
    dv_norm_ms: np.float64
    state_before: NDArray[np.float64]
    state_after: NDArray[np.float64]
    error_before: NDArray[np.float64]


# ΔV EVSK avec vecteurs propres locaux (propagés par STM)


def _local_eigvecs(
    Phi_k: NDArray[np.float64],
    v_s0: NDArray[np.float64],
    v_u_left0: NDArray[np.float64],
    v_u0: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Propage v_s et v_u_left au temps t_k via la STM.

    v_s(t_k)      = Φ(t_k) · v_s(0)            (stable droit)
    v_u_left(t_k) = Φ^{-T}(t_k) · v_u_left(0)  (instable gauche)

    La normalisation biorthogonale ⟨v_u_left, v_u⟩ = 1 est préservée par
    cette propagation (propriété de la STM), on renormalise quand même pour
    éviter l'accumulation d'erreurs numériques.

    Parameters
    ----------
    Phi_k     : (6,6) STM Φ(t_k, 0)
    v_s0      : (6,)  vecteur propre stable à t=0
    v_u_left0 : (6,) vecteur propre instable gauche à t=0 (biorthonorm.)
    v_u0      : (6,) vecteur propre instable droit à t=0
    """

    # Propagation du vecteur stable
    v_s_k = Phi_k @ v_s0
    norm_s = np.float64(np.linalg.norm(v_s_k))
    if norm_s > 1e-14:
        v_s_k /= norm_s

    # Propagation du vecteur instable gauche : Φ^{-T} w = solve(Φ^T, w)
    v_u_left_k = np.linalg.solve(Phi_k.T, v_u_left0)

    # Propagation du vecteur instable droit (pour biorthonormalisation)
    v_u_k = Phi_k @ v_u0
    dot = np.float64(np.dot(v_u_left_k, v_u_k))
    if abs(dot) > 1e-14:
        v_u_left_k /= dot

    return v_s_k, v_u_left_k


def evsk_delta_v(
    dx: NDArray[np.float64],
    v_s: NDArray[np.float64],
    v_u_left: NDArray[np.float64],
    dv_max_adim: np.float64 = np.float64(1e-4),
) -> NDArray[np.float64]:
    """
    Calcule le vecteur ΔV selon l'algorithme EVSK.

    Parameters
    ----------
    dx          : (6,)  écart d'état δx = x_réel - x_réf
    v_s         : (6,)  vecteur propre stable LOCAL (propagé par STM)
    v_u_left    : (6,)  vecteur propre instable gauche LOCAL (biorthonorm.)
    dv_max_adim : np.float64 plafond de sécurité sur |ΔV| [adim. vitesse]

    Returns
    -------
    dv : (3,)  impulsion en vitesse [adim.]
    """
    e_v = v_s[3:].copy()
    norm_ev = np.float64(np.linalg.norm(e_v))
    if norm_ev < 1e-14:
        return np.zeros(3, dtype=np.float64)
    e_v /= norm_ev

    dv_state = np.zeros(6, dtype=np.float64)
    dv_state[3:] = e_v

    proj_dx = np.dot(v_u_left, dx)
    proj_ev = np.dot(v_u_left, dv_state)

    if abs(proj_ev) < 1e-14:
        return np.zeros(3)

    alpha = -proj_dx / proj_ev

    if abs(alpha) > dv_max_adim:
        alpha = np.sign(alpha) * dv_max_adim

    return alpha * e_v


# Simulation avec station-keeping


class StationKeepingSimulation:
    """
    Simulation CR3BP avec corrections EVSK (vecteurs propres propagés par STM).

    Parameters
    ----------
    Az : np.float64
        Amplitude hors-plan [adim.].
    mu : np.float64
        Paramètre de masse CR3BP.
    n_revolutions : np.float64
        Durée de simulation [révolutions halo].
    n_steps_per_rev : int
        Pas RK4 par révolution.
    dt_maneuver_days : np.float64
        Intervalle entre manœuvres [jours]. Défaut : 21 j.
    perturbation : array (6,) or None
        Perturbation initiale. Défaut : +100 km radial.
    dv_max_ms : np.float64
        Plafond de sécurité sur |ΔV| [m/s].
    n_stm_steps : int
        Pas pour la monodromie et la STM de référence.
    """

    def __init__(
        self,
        Az: np.float64 = np.float64(0.00279),
        mu: np.float64 = MU_SUN_EARTH,
        n_revolutions: np.int16 = np.int16(4),
        n_steps_per_rev: np.int16 = np.int16(5000),
        dt_maneuver_days: np.float32 = np.float32(21),
        perturbation: NDArray[np.float64] | None = None,
        dv_max_ms: np.float64 = np.float64(2.0),
        n_stm_steps: np.int16 = np.int16(10000),
    ):
        self.Az = Az
        self.mu = mu
        self.n_revolutions = n_revolutions
        self.n_steps_per_rev = n_steps_per_rev
        self.n_stm_steps = n_stm_steps
        self.northern = False
        self.dt_maneuver_days = dt_maneuver_days
        self.dt_maneuver = np.float64(dt_maneuver_days * 86400 / T_STAR_SEC)
        self.dv_max_adim = dv_max_ms / V_STAR_MS
        self._perturbation = perturbation

        self.times: NDArray[np.float64] | None = None
        self.states: NDArray[np.float64] | None = None
        self.states_free: NDArray[np.float64] | None = None
        self.states_ref: NDArray[np.float64] | None = None
        self.jacobi: NDArray[np.float64] | None = None
        self.maneuvers: list[Maneuver] = []

        self.M: NDArray[np.float64] | None = None
        self.v_s: NDArray[np.float64] | None = None
        self.v_s_list: list[NDArray[np.float64]] = []
        self.v_u: NDArray[np.float64] | None = None
        self.v_u_left: NDArray[np.float64] | None = None
        self.lam_s: complex | None = None
        self.lam_u: complex | None = None
        self.T_halo: np.float64 | None = None
        self.state0_ref: NDArray[np.float64] | None = None

        # STM dense sur une période (pour propagation des vecteurs propres)
        self._t_stm: NDArray[np.float64] | None = None  # shape (N_stm,)
        self._stm_arr: NDArray[np.float64] | None = None  # shape (N_stm, 6, 6)

    def run(self):
        # 1. État initial et période
        state0, T_half, c2 = richardson_halo_L2(
            self.Az, self.mu, northern=self.northern, phi=ZERO
        )
        # Raffinement en orbite périodique (tir simple) AVANT la monodromie : sinon
        # M = Φ(T,0) n'a pas la structure {1, 1, λ, 1/λ, e^{±iθ}} (valeurs propres
        # toutes complexes) car la graine n'est pas réellement périodique.
        state0, self.T_halo = correct_halo(state0, self.mu, T_guess=2 * T_half)
        self.state0_ref = state0
        t_end = self.n_revolutions * self.T_halo
        h = self.T_halo / self.n_steps_per_rev
        f = eom_factory(self.mu)

        # 2. Propagation référence + STM sur une période (système augmenté 42D)
        print("[SK] Propagation référence + STM sur une période...")
        y0_stm = np.concatenate([state0, np.eye(6).flatten(order="F")])
        t_stm, ys_stm = integrate(
            eom_stm_factory(self.mu),
            y0_stm,
            ZERO,
            self.T_halo,
            self.T_halo / self.n_stm_steps,
        )
        self._t_stm = t_stm
        self._stm_arr = ys_stm[:, 6:].reshape(-1, 6, 6, order="F")

        # NOUVEAU : On utilise cette seule période comme référence de base
        # self._t_ref = t_stm
        # self._s_ref = ys_stm[:, :6]

        # 3. Monodromie et vecteurs propres à t=0
        self.M = self._stm_arr[-1]
        assert self.M is not None
        self.v_s, self.v_u, _, self.v_u_left, self.lam_s, self.lam_u = (
            stable_unstable_eigvecs(self.M)
        )
        print_monodromy_summary(self.M, self.mu)

        # 4. Propagation référence sur toute la durée (pour interpolation δx)
        print("[SK] Propagation de l'orbite de référence complète...")
        t_ref, s_ref = integrate(f, state0, ZERO, t_end, h)
        self._t_ref = t_ref
        self._s_ref = s_ref

        # 5. État initial perturbé
        if self._perturbation is not None:
            s0_p = state0 + self._perturbation
        else:
            pert = np.zeros(6)
            pert[0] = 100e3 / 1.496e11  # +100 km radial
            s0_p = state0 + pert

        # 6. Intégration avec corrections EVSK
        print("[SK] Intégration avec corrections EVSK...")
        self.times, self.states = self._integrate_with_sk(s0_p, t_end, h, f)

        # 7. Intégration libre (comparaison)
        _, self.states_free = integrate(f, s0_p, ZERO, t_end, h)

        # 8. Référence interpolée
        self.states_ref = np.array([self._ref_state(t) for t in self.times], np.float64)

        # 9. Constante de Jacobi
        self.jacobi = np.array(
            [jacobi_constant(self.states[i], self.mu) for i in range(len(self.times))]
        )

        print(self._summary())

    def _ref_state(self, t: np.float64) -> NDArray[np.float64]:
        """Interpolation de la référence à l'instant t."""
        idx = np.clip(
            np.searchsorted(self._t_ref, t),
            1,
            len(self._t_ref) - 1,
            dtype=np.int64,
        )
        t0, t1 = self._t_ref[idx - 1], self._t_ref[idx]
        s0, s1 = self._s_ref[idx - 1], self._s_ref[idx]
        if t1 == t0:
            return s0
        return s0 + (t - t0) / (t1 - t0) * (s1 - s0)

    def _stm_at(self, t: np.float64) -> NDArray[np.float64]:
        """
        STM Φ(t mod T, 0) par interpolation sur le tableau pré-calculé.
        On utilise t mod T_halo car la STM est périodique (approximativement).
        """

        assert self._t_stm is not None and self._stm_arr is not None

        idx = np.clip(
            np.searchsorted(self._t_stm, t),
            1,
            len(self._t_stm) - 1,
            dtype=np.int64,
        )
        t0, t1 = self._t_stm[idx - 1], self._t_stm[idx]
        P0, P1 = self._stm_arr[idx - 1], self._stm_arr[idx]
        if t1 == t0:
            return P0
        alpha = (t - t0) / (t1 - t0)
        return P0 + alpha * (P1 - P0)

    def _integrate_with_sk(
        self, s0: NDArray[np.float64], t_end, h: np.float64, f: Callable
    ):
        n_steps = np.int64(np.ceil(t_end / h))
        times = np.empty(n_steps + 1)
        states = np.empty((n_steps + 1, 6), dtype=np.float64)

        t, s = ZERO, s0.copy()
        times[0], states[0] = t, s
        t_next_man = self.dt_maneuver

        for i in range(1, n_steps + 1):
            h_eff = min(h, t_end - t)

            if t < t_next_man <= t + h_eff:
                h1 = t_next_man - t
                s = rk4_step(f, t, s, h1)
                t = t_next_man
                s = self._apply_maneuver(t, s)
                h2 = h_eff - h1
                if h2 > 1e-15:
                    s = rk4_step(f, t, s, h2)
                    t += h2
                t_next_man += self.dt_maneuver
            else:
                s = rk4_step(f, t, s, h_eff)
                t += h_eff

            times[i], states[i] = t, s

        return times, states

    def _apply_maneuver(
        self, t: np.float64, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        x_ref = self._ref_state(t)
        dx = state - x_ref

        # Vecteurs propres propagés au temps t
        Phi_k = self._stm_at(t)

        assert (
            self.v_s is not None and self.v_u_left is not None and self.v_u is not None
        )
        v_s_k, v_u_left_k = _local_eigvecs(Phi_k, self.v_s, self.v_u_left, self.v_u)
        self.v_s_list.append(v_s_k)

        dv = evsk_delta_v(dx, v_s_k, v_u_left_k, self.dv_max_adim)

        state_after = state.copy()
        state_after[3:] += dv

        self.maneuvers.append(
            Maneuver(
                t_adim=t,
                t_days=t * T_STAR_SEC / 86400,
                delta_v=dv,
                dv_norm=np.float64(np.linalg.norm(dv)),
                dv_norm_ms=np.float64(np.linalg.norm(dv) * V_STAR_MS),
                state_before=state.copy(),
                state_after=state_after.copy(),
                error_before=dx.copy(),
            )
        )
        return state_after

    @property
    def positions(self) -> NDArray[np.float64]:
        assert self.states is not None
        return self.states[:, :3]

    @property
    def positions_free(self) -> NDArray[np.float64]:
        assert self.states_free is not None
        return self.states_free[:, :3]

    @property
    def positions_ref(self) -> NDArray[np.float64]:
        assert self.states_ref is not None
        return self.states_ref[:, :3]

    @property
    def times_days(self) -> NDArray[np.float64]:
        assert self.times is not None
        return self.times * (T_STAR_SEC / 86400)

    @property
    def dv_norms_ms(self) -> NDArray[np.float64]:
        return np.array([m.dv_norm_ms for m in self.maneuvers], np.float64)

    @property
    def maneuver_times_days(self) -> NDArray[np.float64]:
        return np.array([m.t_days for m in self.maneuvers], np.float64)

    @property
    def total_dv_ms(self) -> np.float64:
        return np.float64(np.sum(self.dv_norms_ms))

    @property
    def position_errors(self) -> NDArray[np.float64]:
        return np.linalg.norm(self.positions - self.positions_ref, axis=1) * 1.496e8

    @property
    def position_errors_free(self) -> NDArray[np.float64]:
        return (
            np.linalg.norm(self.positions_free - self.positions_ref, axis=1) * 1.496e8
        )

    def _summary(self) -> str:
        n = len(self.maneuvers)
        dvs = self.dv_norms_ms if n > 0 else np.array([0], np.float64)
        lines = [
            "\n[Station-Keeping] Simulation terminée",
            f"  Révolutions        : {self.n_revolutions}",
            f"  Intervalle manoeuv.: {self.dt_maneuver_days} jours",
            f"  Nombre de manœuvres: {n}",
        ]
        if n > 0:
            lines += [
                f"  ΔV min/max/moy     : {dvs.min():.4f} / {dvs.max():.4f} / {dvs.mean():.4f} m/s",
                f"  ΔV total cumulé    : {self.total_dv_ms:.4f} m/s",
                f"  Erreur pos. max SK : {self.position_errors.max():.1f} km",
                f"  Erreur pos. max lib: {self.position_errors_free.max():.1f} km",
            ]
        return "\n".join(lines)
