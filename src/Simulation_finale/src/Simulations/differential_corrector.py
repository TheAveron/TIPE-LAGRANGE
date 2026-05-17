# differential_corrector.py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


class DifferentialCorrector:
    """
    Correcteur différentiel pour orbites quasi-halo autour de L2.

    Principe (méthode de Howell 1984) :
        Une orbite halo est symétrique par rapport au plan XZ.
        Condition : partir de (x0, 0, z0, 0, vy0, 0) et arriver
        après T/2 à (xf, 0, zf, 0, vyf, 0).

        On itère sur (x0, z0, vy0) pour satisfaire :
            y(T/2) = 0
            vx(T/2) = 0
            vz(T/2) = 0  [optionnel, redondant par symétrie]
    """

    def __init__(self, crtbp_model, tol=1e-10, max_iter=50):
        self.crtbp = crtbp_model
        self.tol = tol
        self.max_iter = max_iter

    def _propagate_half_period(self, state0_norm, T_half_norm):
        """Propage jusqu'au prochain croisement y=0."""

        class YCrossing:
            terminal = True
            direction = -1

            def __call__(self, t, s):
                return s[1]

        event_y_crossing = YCrossing()

        sol = solve_ivp(
            fun=self.crtbp.equations_of_motion,
            t_span=(0, 2 * T_half_norm),
            y0=state0_norm,
            method="DOP853",
            events=event_y_crossing,
            rtol=1e-12,
            atol=1e-12,
            dense_output=True,
        )

        if sol.t_events[0].size == 0:
            raise RuntimeError("Pas de croisement y=0 trouvé")

        t_cross = sol.t_events[0][0]
        state_cross = sol.y_events[0][0]
        return state_cross, t_cross

    def _compute_stm(self, state0_norm, T_norm):
        """
        Calcule la Matrice de Transition d'État (STM) Φ(T, 0).

        La STM satisfait : δstate(T) = Φ(T,0) · δstate(0)
        Intégrée simultanément avec les EOM via les équations
        variationnelles : Φ̇ = A(t) · Φ, Φ(0) = I

        L'état augmenté est [state(6), Φ(36)] soit 42 équations.
        """

        def augmented_eom(t, sv):
            state = sv[:6]
            phi = sv[6:].reshape(6, 6)

            dstate = self.crtbp.equations_of_motion(t, state)
            A = self._jacobian_eom(t, state)
            dphi = A @ phi

            return np.concatenate([dstate, dphi.flatten()])

        phi0 = np.eye(6).flatten()
        sv0 = np.concatenate([state0_norm, phi0])

        sol = solve_ivp(
            fun=augmented_eom,
            t_span=(0, T_norm),
            y0=sv0,
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
        )

        stm = sol.y[6:, -1].reshape(6, 6)
        return stm, sol.y[:6, -1]

    def _jacobian_eom(self, t, state):
        """Jacobien A = ∂f/∂x des EOM par différences finies."""
        eps = 1e-7
        f0 = self.crtbp.equations_of_motion(t, state)
        A = np.zeros((6, 6))

        for j in range(6):
            state_p = state.copy()
            state_p[j] += eps
            fp = self.crtbp.equations_of_motion(t, state_p)
            A[:, j] = (fp - f0) / eps

        return A

    def correct(self, state0_norm, T_half_norm):
        """
        Itère pour trouver les conditions initiales d'une vraie orbite périodique.

        Variables libres : x0, z0, vy0  (vx0=vz0=0 imposés)
        Conditions cibles : vx(T/2)=0, vz(T/2)=0  (+ y(T/2)=0 via event)

        Returns:
            state_corrected : conditions initiales normalisées corrigées
            period : période en unités normalisées
        """
        state = state0_norm.copy()

        for iteration in range(self.max_iter):
            # Propager jusqu'au croisement y=0
            try:
                state_f, t_half = self._propagate_half_period(state, T_half_norm)
            except RuntimeError as e:
                raise RuntimeError(f"Correction échouée à l'itération {iteration}: {e}")

            # Résidu : on veut vx(T/2)=0 et vz(T/2)=0
            vx_f = state_f[3]
            vz_f = state_f[5]

            residual = np.array([vx_f, vz_f])

            if np.linalg.norm(residual) < self.tol:
                print(f"  Convergé en {iteration+1} itérations")
                print(f"  Résidu : ||[vx,vz]|| = {np.linalg.norm(residual):.2e}")
                return state, 2 * t_half

            # STM pour calculer la correction (Newton-Raphson)
            stm, _ = self._compute_stm(state, t_half)

            # On corrige x0 et vy0 (z0 fixé pour sélectionner la famille)
            # df/d(x0, vy0) → bloc 2x2 de la STM
            # Lignes : vx(T/2)=stm[3,:], vz(T/2)=stm[5,:]
            # Colonnes : x0=col 0, vy0=col 4
            M = np.array(
                [
                    [stm[3, 0], stm[3, 4]],  # ∂vx_f/∂x0, ∂vx_f/∂vy0
                    [stm[5, 0], stm[5, 4]],  # ∂vz_f/∂x0, ∂vz_f/∂vy0
                ]
            )

            if abs(np.linalg.det(M)) < 1e-14:
                raise RuntimeError("Matrice de correction singulière")

            correction = np.linalg.solve(M, -residual)
            state[0] += correction[0]  # corriger x0
            state[4] += correction[1]  # corriger vy0

        raise RuntimeError(f"Non convergé après {self.max_iter} itérations")
