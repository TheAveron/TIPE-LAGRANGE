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
from .equations import MU_SUN_EARTH, eom
from core.integrator import integrate

# 1. Jacobien analytique des équations CR3BP


def jacobian(state: np.ndarray, mu: float) -> np.ndarray:
    """
    Jacobien A = ∂f/∂x des équations CR3BP, évalué en state.

    Structure par blocs :
        A = [ 0₃   I₃ ]
            [ Ω    C  ]

    Ω = matrice des dérivées secondes du pseudo-potentiel U*.
    C = [[0, 2, 0], [-2, 0, 0], [0, 0, 0]] (termes de Coriolis).

    Parameters
    ----------
    state : array (6,)  [x, y, z, vx, vy, vz]
    mu    : float

    Returns
    -------
    A : np.ndarray, shape (6, 6)
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

    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = np.array([[Uxx, Uxy, Uxz], [Uxy, Uyy, Uyz], [Uxz, Uyz, Uzz]])
    A[3:, 3:] = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
    return A


# 2. Équations du système augmenté [état, STM]


def eom_stm(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """
    Dérivée du vecteur augmenté [x(6), vec(Φ)(36)].

    y[:6]  = état CR3BP
    y[6:]  = STM Φ vectorisée colonne par colonne (ordre Fortran)
    """
    state = y[:6]
    Phi = y[6:].reshape(6, 6, order="F")
    dstate = eom(t, state, mu)
    dPhi = jacobian(state, mu) @ Phi
    return np.concatenate([dstate, dPhi.flatten(order="F")])


def eom_stm_factory(mu: float):
    def _f(t, y):
        return eom_stm(t, y, mu)

    return _f


# 3. Calcul de la matrice de monodromie


def compute_monodromy(
    state0: np.ndarray,
    T: float,
    mu: float = MU_SUN_EARTH,
    n_steps: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Intègre le système augmenté sur une période T.

    Returns
    -------
    M     : np.ndarray (6, 6)     matrice de monodromie Φ(T, 0)
    times : np.ndarray (N,)
    stms  : np.ndarray (N, 6, 6)  STM à chaque instant
    """
    y0 = np.concatenate([state0, np.eye(6).flatten(order="F")])
    times, ys = integrate(eom_stm_factory(mu), y0, 0.0, T, T / n_steps)
    stms = ys[:, 6:].reshape(-1, 6, 6, order="F")
    return stms[-1], times, stms


# 4. Extraction des directions stable et instable


def stable_unstable_eigvecs(
    M: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex, complex]:
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
    v_s, v_u, v_s_left, v_u_left : np.ndarray (6,)
    lam_s, lam_u : complex
    """
    eigvals_R, V_R = np.linalg.eig(M)  # droits  : M  v = λ v
    eigvals_L, V_L = np.linalg.eig(M.T)  # gauches : M^T w = λ w

    mods = np.abs(eigvals_R)

    # Valeur propre la plus petite en module → stable
    # Valeur propre la plus grande en module → instable
    # On exclut les paires complexes en cherchant parmi les réelles
    real_mask = np.abs(eigvals_R.imag) < 1e-6 * np.abs(eigvals_R.real + 1e-30)  # type: ignore

    if real_mask.sum() >= 2:
        real_idx = np.where(real_mask)[0]
        idx_s = real_idx[np.argmin(mods[real_idx])]
        idx_u = real_idx[np.argmax(mods[real_idx])]
    else:
        idx_s = int(np.argmin(mods))
        idx_u = int(np.argmax(mods))

    lam_s = eigvals_R[idx_s]
    lam_u = eigvals_R[idx_u]

    v_s = eigvals_R[idx_s]  # valeur propre stable (pour appariement)
    v_u_val = eigvals_R[idx_u]

    # Appariement des vecteurs gauches : trouver dans eigvals_L celui le plus
    # proche de lam_s et lam_u respectivement.
    idx_s_L = int(np.argmin(np.abs(eigvals_L - lam_s)))
    idx_u_L = int(np.argmin(np.abs(eigvals_L - lam_u)))

    v_s_vec = V_R[:, idx_s].real  # type: ignore
    v_u_vec = V_R[:, idx_u].real  # type: ignore
    v_s_left_vec = V_L[:, idx_s_L].real  # type: ignore
    v_u_left_vec = V_L[:, idx_u_L].real  # type: ignore

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


def print_monodromy_summary(M: np.ndarray, mu: float = MU_SUN_EARTH):
    eigvals = np.linalg.eigvals(M)
    mods = np.abs(eigvals)
    v_s, _, _, _, lam_s, lam_u = stable_unstable_eigvecs(M)

    print(v_s)
    print("\n[Monodromie]")
    print(f"  Valeurs propres |λ| : {sorted(mods.tolist())}")
    print(f"  λ_stable   = {lam_s:.6f}  (|λ_s| = {abs(lam_s):.4e})")
    print(f"  λ_instable = {lam_u:.6f}  (|λ_u| = {abs(lam_u):.4e})")
    print(f"  Produit |λ_s|·|λ_u| = {abs(lam_s)*abs(lam_u):.6f}  (attendu ≈ 1)")
