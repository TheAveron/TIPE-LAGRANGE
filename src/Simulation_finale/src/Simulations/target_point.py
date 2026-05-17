import numpy as np
from scipy.integrate import solve_ivp

from src.Simulations.station_keeping import ManeuverPlan


class TargetPointController:
    """
    Target Point Method : stratégie opérationnelle JWST.

    À chaque epoch de manœuvre t_m :
        1. Propager l'état actuel sur Δt sans manœuvre → état "libre" x_free(t_m + Δt)
        2. Définir un état cible x_ref(t_m + Δt) sur l'orbite nominale
        3. Calculer le ΔV à t_m via la STM :

           δx(t_m + Δt) = Φ(t_m+Δt, t_m) · [0, 0, 0, δvx, δvy, δvz]ᵀ

           On veut : x_free + δx_propagé = x_ref
           Donc :   Φ_vv · δv = x_ref_pos - x_free_pos  (position seulement)

           où Φ_vv est le bloc 3×3 position/vitesse de la STM.

    Note :
        On ne cible que la POSITION (3 équations pour 3 inconnues δv).
        La vitesse est laissée libre — elle convergera naturellement
        si l'orbite est correctement fermée.
    """

    def __init__(self, crtbp_model, reference_orbit, dt_target_days=21.0):
        self.crtbp = crtbp_model
        self.reference_orbit = reference_orbit
        self.dt_target = dt_target_days * 86400.0

        T = reference_orbit.period
        Ay = reference_orbit.amplitudes["y"]
        Az = reference_orbit.amplitudes["z"]
        Ax = Ay / 3.229
        nu = 2.086
        self.T = T
        self.Ay = Ay
        self.Az = Az
        self.Ax = Ax
        self.nu = nu

        from src.Simulations.constants import Constants

        self.L_star = Constants.AU
        self.V_star = Constants.AU * Constants.OMEGA_EARTH
        self.omega = Constants.OMEGA_EARTH
        self.x_l2_phys = (
            1.0
            - Constants.MU_RATIO_SUN_EARTH
            + (Constants.MU_RATIO_SUN_EARTH / 3) ** (1 / 3)
        ) * Constants.AU

    def _reference_state_at(self, time):
        """État analytique de référence à un temps donné [s]."""
        tau = 2.0 * np.pi * (time % self.T) / self.T
        tau_dot = 2.0 * np.pi / self.T

        return np.array(
            [
                self.x_l2_phys - self.Ax * np.cos(self.nu * tau / (2 * np.pi)),
                self.Ay * np.sin(tau),
                self.Az * np.cos(tau),
                0.0,
                self.Ay * tau_dot * np.cos(tau),
                -self.Az * tau_dot * np.sin(tau),
            ]
        )

    def _propagate_free(self, state, t_start, dt):
        """Propage l'état sans manœuvre sur dt secondes."""
        state_norm = state.copy()
        state_norm[:3] /= self.L_star
        state_norm[3:] /= self.V_star

        T_star = 1.0 / self.omega
        dt_norm = dt / T_star

        sol = solve_ivp(
            fun=self.crtbp.equations_of_motion,
            t_span=(0, dt_norm),
            y0=state_norm,
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
        )

        state_f_norm = sol.y[:, -1]
        state_f = np.zeros(6)
        state_f[:3] = state_f_norm[:3] * self.L_star
        state_f[3:] = state_f_norm[3:] * self.V_star
        return state_f

    def _compute_stm_physical(self, state, dt):
        """STM en unités physiques sur l'intervalle dt."""
        state_norm = state.copy()
        state_norm[:3] /= self.L_star
        state_norm[3:] /= self.V_star

        T_star = 1.0 / self.omega
        dt_norm = dt / T_star

        def jac_fd(t, s):
            eps = 1e-7
            f0 = self.crtbp.equations_of_motion(t, s)
            A = np.zeros((6, 6))
            for j in range(6):
                sp = s.copy()
                sp[j] += eps
                A[:, j] = (self.crtbp.equations_of_motion(t, sp) - f0) / eps
            return A

        def aug_eom(t, sv):
            s = sv[:6]
            phi = sv[6:].reshape(6, 6)
            ds = self.crtbp.equations_of_motion(t, s)
            dphi = jac_fd(t, s) @ phi
            return np.concatenate([ds, dphi.flatten()])

        sv0 = np.concatenate([state_norm, np.eye(6).flatten()])
        sol = solve_ivp(
            aug_eom, (0, dt_norm), sv0, method="DOP853", rtol=1e-11, atol=1e-11
        )

        stm_norm = sol.y[6:, -1].reshape(6, 6)

        # Convertir STM en unités physiques
        # δx_phys = S · δx_norm  avec S = diag(L*, L*, L*, V*, V*, V*)
        S = np.diag([self.L_star] * 3 + [self.V_star] * 3)
        S_inv = np.diag([1 / self.L_star] * 3 + [1 / self.V_star] * 3)
        stm_phys = S @ stm_norm @ S_inv

        return stm_phys

    def compute_maneuver(self, current_state, time):
        """
        Calcule le ΔV optimal par la Target Point Method.

        Système à résoudre :
            Φ_rv · δv = x_target - x_free

        où Φ_rv est le bloc [3x3] position/vitesse de la STM
        (lignes 0:3, colonnes 3:6).
        """
        t_target = time + self.dt_target

        # 1. Propager librement jusqu'à t_target
        state_free = self._propagate_free(current_state, time, self.dt_target)

        # 2. État de référence au temps cible
        state_ref = self._reference_state_at(t_target)

        # 3. Écart de position à corriger
        pos_error = state_ref[:3] - state_free[:3]

        # 4. STM sur l'intervalle
        stm = self._compute_stm_physical(current_state, self.dt_target)

        # 5. Bloc Φ_rv : comment une impulsion δv maintenant
        #    affecte la position dans dt_target
        phi_rv = stm[:3, 3:6]  # lignes position, colonnes vitesse

        # 6. Résoudre Φ_rv · δv = pos_error (système 3×3)
        try:
            delta_v = np.linalg.lstsq(phi_rv, pos_error, rcond=None)[0]
        except np.linalg.LinAlgError:
            delta_v = np.zeros(3)

        magnitude = float(np.linalg.norm(delta_v))

        state_after = current_state.copy()
        state_after[3:] += delta_v

        return ManeuverPlan(
            time=time,
            delta_v=delta_v,
            magnitude=magnitude,
            state_before=current_state.copy(),
            state_after=state_after,
            reason=f"Target Point Method (Δt={self.dt_target/86400:.0f}j)",
        )
