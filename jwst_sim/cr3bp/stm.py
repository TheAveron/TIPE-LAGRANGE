"""
stm.py — Matrice de transition d'état (STM) et matrice de monodromie.

La STM Φ(t, t₀) vérifie :
    Φ̇ = A(t) · Φ,    Φ(t₀, t₀) = I₆

où A(t) = ∂f/∂x est le Jacobien des équations du CR3BP évalué le long
de la trajectoire de référence.

La matrice de monodromie est M = Φ(T, 0) pour T = période de l'orbite halo.

Valeurs propres de M pour une orbite halo (Koon et al. 2011) :
    {1, 1, λ_s, 1/λ_s, e^(iν), e^(-iν)}
    - λ_s < 1 : direction stable   (convergence vers l'orbite)
    - 1/λ_s > 1 : direction instable (divergence depuis l'orbite)
    - e^±iν : centre (mouvement quasi-périodique)

Système augmenté pour l'intégration simultanée de [x, Φ] :
    dim = 6 (état) + 36 (STM vectorisée colonne par colonne) = 42
"""

import numpy as np
from core.integrator import integrate, rk4_step
from numpy.typing import NDArray

from .equations import MU_SUN_EARTH, eom

ZERO = np.float64(0)


# 1. Jacobien analytique des équations CR3BP
def jacobian(state: NDArray[np.float64], mu: np.float64) -> NDArray[np.float64]:
    """
    Jacobien A = ∂f/∂x des équations CR3BP, évalué en state.

    Structure par blocs :
        A = [ 0₃   I₃ ]
            [ Ω    C  ]

    Ω = matrice des dérivées secondes du pseudo-potentiel U*.
    C = [[0, 2, 0], [-2, 0, 0], [0, 0, 0]] (termes de Coriolis).

    Parameters
    ----------
    state : NDArray[np.float64], shape (6,)  [x, y, z, vx, vy, vz]
    mu    : np.float64

    Returns
    -------
    A : NDArray[np.float64], shape (6, 6)
    """

    x, y, z = state[0], state[1], state[2]

    d2 = (x + mu) ** 2 + y**2 + z**2
    r2 = (x - 1 + mu) ** 2 + y**2 + z**2
    d3, d5 = d2**1.5, d2**2.5
    r3, r5 = r2**1.5, r2**2.5

    Uxx = (
        1
        - (1 - mu) / d3
        + 3 * (1 - mu) * (x + mu) ** 2 / d5
        - mu / r3
        + 3 * mu * (x - 1 + mu) ** 2 / r5
    )
    Uyy = 1 - (1 - mu) / d3 + 3 * (1 - mu) * y**2 / d5 - mu / r3 + 3 * mu * y**2 / r5
    Uzz = -(1 - mu) / d3 + 3 * (1 - mu) * z**2 / d5 - mu / r3 + 3 * mu * z**2 / r5
    Uxy = 3 * (1 - mu) * (x + mu) * y / d5 + 3 * mu * (x - 1 + mu) * y / r5
    Uxz = 3 * (1 - mu) * (x + mu) * z / d5 + 3 * mu * (x - 1 + mu) * z / r5
    Uyz = 3 * (1 - mu) * y * z / d5 + 3 * mu * y * z / r5

    A = np.zeros((6, 6), dtype=np.float64)
    A[:3, 3:] = np.eye(3, dtype=np.float64)
    A[3:, :3] = np.array(
        [[Uxx, Uxy, Uxz], [Uxy, Uyy, Uyz], [Uxz, Uyz, Uzz]], dtype=np.float64
    )
    A[3:, 3:] = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]], dtype=np.float64)
    return A


# 2. Équations du système augmenté [état, STM]
def eom_stm(
    t: np.float64, y: NDArray[np.float64], mu: np.float64
) -> NDArray[np.float64]:
    """
    Dérivée du vecteur augmenté [x(6), vec(Φ)(36)].

    y[:6]  = état CR3BP
    y[6:]  = STM Φ vectorisée colonne par colonne (ordre Fortran)
    """
    state = y[:6]
    Phi = y[6:].reshape(6, 6, order="F")
    dstate = eom(t, state, mu)
    dPhi = jacobian(state, mu) @ Phi
    return np.concatenate([dstate, dPhi.flatten(order="F")], dtype=np.float64)


def eom_stm_factory(mu: np.float64):
    def _f(t, y):
        return eom_stm(t, y, mu)

    return _f


# 3. Calcul de la matrice de monodromie


def compute_monodromy(
    state0: NDArray[np.float64],
    T: np.float64,
    mu: np.float64 = MU_SUN_EARTH,
    n_steps: np.int16 = np.int16(10000),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Intègre le système augmenté sur une période T.

    Returns
    -------
    M     : NDArray[np.float64] (6, 6)     matrice de monodromie Φ(T, 0)
    times : NDArray[np.float64] (N,)
    stms  : NDArray[np.float64] (N, 6, 6)  STM à chaque instant
    """

    y0 = np.concatenate([state0, np.eye(6).flatten(order="F")], dtype=np.float64)
    times, ys = integrate(eom_stm_factory(mu), y0, ZERO, T, T / n_steps)
    stms = ys[:, 6:].reshape(-1, 6, 6, order="F")
    return stms[-1], times, stms


# 3b. Correction différentielle (tir simple) → orbite halo périodique
def _integrate_to_half_period(
    state0: NDArray[np.float64],
    mu: np.float64,
    T_guess: np.float64,
    n_steps: int = 20000,
) -> tuple[np.float64, NDArray[np.float64], NDArray[np.float64]]:
    """
    Intègre le système augmenté [état, Φ] de t=0 jusqu'au PREMIER passage y=0 (t>0).

    La graine est sur le plan xz (y0=0) ; le prochain passage y=0 marque la
    demi-période. Renvoie (t_cross, état(6) au passage, Φ(6,6) au passage).
    """
    f = eom_stm_factory(mu)
    y = np.concatenate([state0, np.eye(6).flatten(order="F")], dtype=np.float64)
    h = np.float64(T_guess / n_steps)
    t = ZERO
    for i in range(n_steps):
        y_prev, t_prev = y.copy(), t
        y = rk4_step(f, t, y, h)
        t = t + h
        # passage par zéro de la composante y (indice 1), après avoir quitté y=0
        if i > 0 and y_prev[1] * y[1] < 0:
            frac = y_prev[1] / (y_prev[1] - y[1])  # interpolation linéaire du zéro
            y_cross = rk4_step(f, t_prev, y_prev, frac * h)
            t_cross = t_prev + frac * h
            return t_cross, y_cross[:6], y_cross[6:].reshape(6, 6, order="F")
    raise RuntimeError("Aucun passage y=0 trouvé dans [0, T_guess].")


def correct_halo(
    state0: NDArray[np.float64],
    mu: np.float64 = MU_SUN_EARTH,
    T_guess: np.float64 = np.float64(3.0),
    tol: np.float64 = np.float64(1e-11),
    max_iter: int = 50,
) -> tuple[NDArray[np.float64], np.float64]:
    """
    Raffine une graine de Richardson en orbite halo PÉRIODIQUE par tir simple.

    On exploite la symétrie des halos par rapport au plan xz : la graine est de la
    forme [x0, 0, z0, 0, vy0, 0] (passage perpendiculaire). On intègre jusqu'au
    prochain passage y=0 et on impose vx=vz=0 à ce passage, en ajustant (x0, vy0)
    à z0 = Az fixé. Newton via la STM, avec correction du temps de vol :

        d[vx; vz] = ( [[Φ41 Φ45],[Φ61 Φ65]] - (1/vy_f)[ax_f; az_f]⊗[Φ21 Φ25] ) [dx0; dvy0]

    Returns
    -------
    state0_corr : (6,)  condition initiale périodique corrigée
    T           : période complète vraie (= 2·t_cross)
    """
    s = np.array(state0, dtype=np.float64).copy()
    err = np.float64(np.inf)
    for _ in range(max_iter):
        t_half, s_c, Phi = _integrate_to_half_period(s, mu, T_guess)
        vx_f, vz_f = s_c[3], s_c[5]
        err = max(abs(vx_f), abs(vz_f))
        if err < tol:
            return s, np.float64(2.0 * t_half)
        ds = eom(t_half, s_c, mu)
        ax_f, az_f = ds[3], ds[5]
        vy_f = s_c[4]
        Mmat = np.array(
            [[Phi[3, 0], Phi[3, 4]], [Phi[5, 0], Phi[5, 4]]], dtype=np.float64
        ) - (1.0 / vy_f) * np.outer(
            np.array([ax_f, az_f]), np.array([Phi[1, 0], Phi[1, 4]])
        )
        d = np.linalg.solve(Mmat, -np.array([vx_f, vz_f], dtype=np.float64))
        s[0] += d[0]
        s[4] += d[1]
        T_guess = np.float64(2.0 * t_half)
    raise RuntimeError(f"Correction différentielle non convergée (err={err:.2e}).")


# 4. Extraction des directions stable et instable
def stable_unstable_eigvecs(
    M: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    np.complex128,
    np.complex128,
]:
    """
    Extrait les vecteurs propres stable et instable de la monodromie.

    Appariement gauche/droit :
    Les vecteurs propres gauches (adjoints) sont les vecteurs propres droits
    de M^T. On les apparie avec les valeurs propres de M par distance minimale
    dans le plan complexe, puis on normalise biorthogonalement :
        v_u_left ← v_u_left / (v_u_left · v_u)
    de sorte que ⟨v_u_left, v_u⟩ = 1.

    Returns
    -------
    v_s, v_u, v_s_left, v_u_left : NDArray (6,)
    lam_s, lam_u : complex
    """
    eigvals_R, V_R = np.linalg.eig(M)  # droits  : M  v = λ v
    eigvals_L, V_L = np.linalg.eig(M.T)  # gauches : M^T w = λ w

    mods = np.abs(eigvals_R, dtype=np.float64)

    # Valeur propre la plus petite en module → stable
    # Valeur propre la plus grande en module → instable
    # On exclut les paires complexes en cherchant parmi les réelles
    real_mask = np.abs(eigvals_R.imag, dtype=np.float64) < 1e-6 * np.abs(
        eigvals_R.real + 1e-30, dtype=np.float64
    )  #

    if real_mask.sum() >= 2:
        real_idx = np.where(real_mask)[0]
        idx_s = real_idx[np.argmin(mods[real_idx])]
        idx_u = real_idx[np.argmax(mods[real_idx])]
    else:
        idx_s = int(np.argmin(mods))
        idx_u = int(np.argmax(mods))

    lam_s = eigvals_R[idx_s]
    lam_u = eigvals_R[idx_u]

    # Appariement des vecteurs gauches : trouver dans eigvals_L celui le plus
    # proche de lam_s et lam_u respectivement.
    idx_s_L = int(np.argmin(np.abs(eigvals_L - lam_s)))
    idx_u_L = int(np.argmin(np.abs(eigvals_L - lam_u)))

    v_s_vec = V_R[:, idx_s].real
    v_u_vec = V_R[:, idx_u].real
    v_s_left_vec = V_L[:, idx_s_L].real
    v_u_left_vec = V_L[:, idx_u_L].real

    # Normalisation L2
    v_s_vec /= np.linalg.norm(v_s_vec)
    v_u_vec /= np.linalg.norm(v_u_vec)

    # Normalisation biorthogonale : ⟨v_u_left, v_u⟩ = 1
    dot_u = np.dot(v_u_left_vec, v_u_vec)
    if abs(dot_u) > 1e-14:
        v_u_left_vec /= dot_u
    else:
        v_u_left_vec /= np.linalg.norm(v_u_left_vec)

    dot_s = np.dot(v_s_left_vec, v_s_vec)
    if abs(dot_s) > 1e-14:
        v_s_left_vec /= dot_s
    else:
        v_s_left_vec /= np.linalg.norm(v_s_left_vec)

    return v_s_vec, v_u_vec, v_s_left_vec, v_u_left_vec, lam_s, lam_u


def print_monodromy_summary(M: NDArray[np.float64], mu: np.float64 = MU_SUN_EARTH):
    eigvals = np.linalg.eigvals(M)
    mods = np.abs(eigvals)
    v_s, _, _, _, lam_s, lam_u = stable_unstable_eigvecs(M)

    print(v_s)
    print("\n[Monodromie]")
    print(f"  Valeurs propres |λ| : {sorted(mods.tolist())}")
    print(f"  λ_stable   = {lam_s:.6f}  (|λ_s| = {abs(lam_s):.4e})")
    print(f"  λ_instable = {lam_u:.6f}  (|λ_u| = {abs(lam_u):.4e})")
    print(f"  Produit |λ_s|·|λ_u| = {abs(lam_s)*abs(lam_u):.6f}  (attendu ≈ 1)")
