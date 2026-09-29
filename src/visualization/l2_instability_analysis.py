"""
l2_instability_analysis.py
==========================
Analyse physique et numérique de l'instabilité du point de Lagrange L2
dans le problème restreint des trois corps circulaire (CR3BP) Soleil-Terre.

Quatre modules graphiques :
  1. Surface du potentiel effectif U*(x,y) et localisation des points de Lagrange
     → L2 est un col (point-selle) : ∂²U*/∂x² > 0 mais det(Hess) < 0.
  2. Spectre du linéarisé A = ∂f/∂x en L1, L2, L4, L5
     → présence d'une valeur propre réelle positive en L2 ↔ instabilité de Lyapunov.
  3. Variétés stable et instable au voisinage de L2
     → divergence exponentielle le long de la direction instable v_u.
  4. Croissance de ‖δx(t)‖ pour des perturbations initiales ε·v_u (exposant de Lyapunov)
     → comparaison croissance exponentielle théorique vs numérique.

Conventions :
  - μ = m_Terre / (m_Soleil + m_Terre) ≈ 3.04e-6  (valeur JPL)
  - Soleil en x = -μ,  Terre en x = 1-μ  (repère tournant adim.)
  - Potentiel effectif : U*(x,y) = (1-μ)/d₁ + μ/d₂ + (x²+y²)/2
    où C = 2U* - v²  (constante de Jacobi)

Références :
  Szebehely (1967), Koon et al. (2011), Gómez et al. (2001).
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from scipy.linalg import eigvals
from scipy.optimize import brentq

# Constantes


MU = np.float64(3.040423389123456e-6)  # paramètre de masse Soleil-Terre (JPL)
T_STAR_DAYS = np.float64(
    365.25 / (2 * np.pi)
)  # unité de temps adim. → jours (≈ 58.15 j)

ZERO = np.float64(0)

# Style global
BG = "#FFFFFF"
FG = "#0D0D1A"
GRID = "#1E1E3A"
C_L2 = "#FF6B6B"
C_L1 = "#FFA07A"
C_L4 = "#90EE90"
C_STB = "#00BFFF"
C_UNS = "#FF4444"
C_POT = "#DDA0DD"


def _apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": FG,
            "axes.labelcolor": FG,
            "xtick.color": FG,
            "ytick.color": FG,
            "text.color": FG,
            "grid.color": GRID,
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": FG,
            "font.family": "monospace",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
        }
    )


_apply_style()


# 1.  Fonctions CR3BP de base
def distances(
    x: np.float64, y: np.float64, z: np.float64, mu: np.float64
) -> tuple[np.float64, np.float64]:
    """
    Distances adim. du spacecraft au Soleil (d) et à la Terre (r).

    d = ||spacecraft - Soleil||,  Soleil en (-μ, 0, 0)
    r = ||spacecraft - Terre||,   Terre  en (1-μ, 0, 0)
    """
    d = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)
    return d, r


def U_eff(x: np.float64, y: np.float64, mu=MU) -> np.float64:
    """Potentiel effectif U*(x,y,z=0) = (1-μ)/d₁ + μ/d₂ + (x²+y²)/2."""
    d1, d2 = distances(x, y, ZERO, mu)
    return (1 - mu) / d1 + mu / d2 + 0.5 * (x**2 + y**2)


def eom(t: np.float64, state: NDArray[np.float64], mu=MU) -> NDArray[np.float64]:
    """Équations de mouvement CR3BP (vecteur d'état [x,y,z,vx,vy,vz])."""
    x, y, z, vx, vy, vz = state
    d1, d2 = distances(x, y, z, mu)
    d1_3, d2_3 = d1**3, d2**3
    dUx = x - (1 - mu) * (x + mu) / d1_3 - mu * (x - 1 + mu) / d2_3
    dUy = y * (1 - (1 - mu) / d1_3 - mu / d2_3)
    dUz = z * (-(1 - mu) / d1_3 - mu / d2_3)
    return np.array([vx, vy, vz, dUx + 2 * vy, dUy - 2 * vx, dUz], dtype=np.float64)


def rk4(
    f: Callable[[np.float64, NDArray[np.float64]], NDArray[np.float64]],
    t: np.float64,
    y: NDArray[np.float64],
    h: np.float64,
) -> NDArray[np.float64]:
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_rk4(
    f: Callable[[np.float64, NDArray[np.float64]], NDArray[np.float64]],
    y0: NDArray[np.float64],
    t0: np.float64,
    t_end: np.float64,
    h: np.float64,
):
    n = int(np.ceil((t_end - t0) / h))
    t = t0
    y = y0.copy()
    traj = [y.copy()]
    for _ in range(n):
        h_eff = min(h, t_end - t)
        y = rk4(f, t, y, h_eff)
        t += h_eff
        traj.append(y.copy())
    return np.array(traj, dtype=np.float64)


# 2. Points de Lagrange colinéaires L1, L2, L3


def _gamma_L1(mu: np.float64):

    def eq(g: np.float64):
        return (
            g**5 - (3 - mu) * g**4 + (3 - 2 * mu) * g**3 - mu * g**2 + 2 * mu * g - mu
        )

    g0 = (mu / 3) ** (1 / 3)
    return np.float64(brentq(eq, g0 * 0.5, g0 * 1.5))


def _gamma_L2(mu: np.float64):

    def eq(g: np.float64):
        return (
            g**5 + (3 - mu) * g**4 + (3 - 2 * mu) * g**3 - mu * g**2 - 2 * mu * g - mu
        )

    g0 = (mu / 3) ** (1 / 3)
    return np.float64(brentq(eq, g0 * 0.5, g0 * 1.5))


def _gamma_L3(mu: np.float64):
    # L3 côté Soleil : x_L3 ≈ -(1 - 7μ/12)
    def eq(g: np.float64):
        return (
            g**5
            + (2 + mu) * g**4
            + (1 + 2 * mu) * g**3
            - (1 - mu) * g**2
            - 2 * (1 - mu) * g
            - (1 - mu)
        )

    return np.float64(brentq(eq, 0.5, 1.5))


def lagrange_points(mu=MU):
    """Retourne les 5 points de Lagrange sous forme (x, y)."""
    g1 = _gamma_L1(mu)
    g2 = _gamma_L2(mu)
    g3 = _gamma_L3(mu)
    L1 = np.array([1 - mu - g1, 0.0], dtype=np.float64)
    L2 = np.array([1 - mu + g2, 0.0], dtype=np.float64)
    L3 = np.array([-(1 + g3), 0.0], dtype=np.float64)
    L4 = np.array([0.5 - mu, np.sqrt(3) / 2], dtype=np.float64)
    L5 = np.array([0.5 - mu, -np.sqrt(3) / 2], dtype=np.float64)
    return {"L1": L1, "L2": L2, "L3": L3, "L4": L4, "L5": L5}


# 3. Jacobien du CR3BP (linéarisé en un point quelconque)


def jacobian_cr3bp(x: np.float64, y: np.float64, z: np.float64, mu=MU):
    """
    Jacobien A = ∂f/∂X de la forme f(X) = [ṙ, r̈] avec X = [r, ṙ].

    Bloc U (dérivées secondes du potentiel) :
      U_xx, U_yy, U_zz, U_xy, U_xz, U_yz

    A = [ 0₃    I₃  ]
        [ U     Cor ]  avec Cor = [[0,2,0],[-2,0,0],[0,0,0]]
    """
    d1sq = (x + mu) ** 2 + y**2 + z**2
    d2sq = (x - 1 + mu) ** 2 + y**2 + z**2
    d1_3, d1_5 = d1sq**1.5, d1sq**2.5
    d2_3, d2_5 = d2sq**1.5, d2sq**2.5

    Uxx = (
        1
        - (1 - mu) / d1_3
        + 3 * (1 - mu) * (x + mu) ** 2 / d1_5
        - mu / d2_3
        + 3 * mu * (x - 1 + mu) ** 2 / d2_5
    )
    Uyy = (
        1
        - (1 - mu) / d1_3
        + 3 * (1 - mu) * y**2 / d1_5
        - mu / d2_3
        + 3 * mu * y**2 / d2_5
    )
    Uzz = (
        -(1 - mu) / d1_3 + 3 * (1 - mu) * z**2 / d1_5 - mu / d2_3 + 3 * mu * z**2 / d2_5
    )
    Uxy = 3 * (1 - mu) * (x + mu) * y / d1_5 + 3 * mu * (x - 1 + mu) * y / d2_5
    Uxz = 3 * (1 - mu) * (x + mu) * z / d1_5 + 3 * mu * (x - 1 + mu) * z / d2_5
    Uyz = 3 * (1 - mu) * y * z / d1_5 + 3 * mu * y * z / d2_5

    A = np.zeros((6, 6), dtype=np.float64)
    A[:3, 3:] = np.eye(3, dtype=np.float64)
    A[3:, :3] = np.array(
        [[Uxx, Uxy, Uxz], [Uxy, Uyy, Uyz], [Uxz, Uyz, Uzz]], dtype=np.float64
    )
    A[3:, 3:] = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]], dtype=np.float64)
    return A


def hessian_U_eff(x: np.float64, y: np.float64, mu=MU):
    """
    Hessien de U*(x,y) en z=0 restreint au plan (x,y) → matrice 2x2.
    Sert à caractériser le point critique (col vs minimum/maximum).
    """
    A = jacobian_cr3bp(x, y, ZERO, mu)
    return A[3:5, :2]  # [Uxx Uxy ; Uxy Uyy] avec les corrections centrifuges


#  GRAPHE 1 — Surface du potentiel effectif U*(x,y)
def plot_effective_potential(mu=MU, save_path: str | None = None):
    """
    Carte 2D de U*(x,y) avec contours d'énergie de Jacobi (ZVC)
    et localisation des points de Lagrange.

    Points clés :
    - L4, L5 sont des maxima locaux de U* → stables (si μ < μ_Routh ≈ 0.0385).
    - L1, L2, L3 sont des cols de U* → instables.
    - Le col en L2 est moins profond qu'en L1 (L2 côté anti-Soleil).

    La nature de col est confirmée par le Hessien :
      det(Hess U*|_L2) < 0  ↔  point-selle.
    """
    print("\n[1/4] Surface du potentiel effectif...")
    lp = lagrange_points(mu)

    # Grille fine autour de la région L2 et globale
    Nx, Ny = 800, 600
    x_arr = np.linspace(-1.02, 1.02, Nx, dtype=np.float64)
    y_arr = np.linspace(-1.02, 1.02, Ny, dtype=np.float64)
    X, Y = np.meshgrid(x_arr, y_arr)

    # Potentiel vectorisé (éviter les singularités au voisinage des corps)
    D1: NDArray[np.float64] = np.sqrt((X + mu) ** 2 + Y**2, dtype=np.float64)
    D2: NDArray[np.float64] = np.sqrt((X - 1 + mu) ** 2 + Y**2, dtype=np.float64)
    eps = 1e-1
    D1 = np.where(D1 < eps, eps, D1)
    D2 = np.where(D2 < eps, eps, D2)
    Omega = (1 - mu) / D1 + mu / D2 + 0.5 * (X**2 + Y**2)

    # Valeurs de U* aux points de Lagrange (= niveaux des ZVC)
    omega_L = {k: U_eff(v[0], v[1], mu) for k, v in lp.items()}
    C_L = {
        k: 2 * omega_L[k] for k, v in lp.items()
    }  # constante de Jacobi au point concerné

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        "Potentiel effectif U*(x,y) — Points de Lagrange comme points critiques",
        fontsize=12,
        y=0.98,
    )

    # ── Sous-graphe gauche : heatmap + contours ZVC ──
    ax1 = fig.add_subplot(1, 2, 1)
    vmin, vmax = omega_L["L2"] * 0.9998, omega_L["L2"] * 1.0003
    im = ax1.imshow(
        Omega,
        origin="lower",
        aspect="auto",
        extent=(x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]),
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("U*(x,y)")

    # Courbes de Jacobi aux niveaux des points colinéaires
    levels_plot = np.linspace(
        omega_L["L2"] * 0.9999, omega_L["L2"] * 1.0002, 20, dtype=np.float64
    )
    cs = ax1.contour(
        X, Y, Omega, levels=levels_plot, colors=FG, alpha=0.3, linewidths=0.5
    )

    # Niveaux critiques L1, L2 en traits distincts
    for k, col, ls in [("L2", C_L2, "-"), ("L1", C_L1, "--")]:
        if omega_L[k] >= vmin and omega_L[k] <= vmax:
            ax1.contour(
                X,
                Y,
                Omega,
                levels=[omega_L[k]],
                colors=[col],
                linewidths=1.2,
                linestyles=ls,
            )

    # Marquage des points de Lagrange dans la fenêtre
    for k, pos in lp.items():
        if x_arr[0] <= pos[0] <= x_arr[-1] and y_arr[0] <= pos[1] <= y_arr[-1]:
            col = C_L2 if k == "L2" else C_L1 if k == "L1" else C_L4
            ax1.plot(pos[0], pos[1], "o", color=col, ms=7, zorder=5)
            ax1.annotate(
                k,
                tuple(pos),
                textcoords="offset points",
                xytext=(6, 5),
                color=col,
                fontsize=9,
                fontweight="bold",
            )

    # Marquage de la Terre (dans la fenêtre)
    ax1.plot(1 - mu, 0, "o", color="#3A9BD5", ms=5, label="Terre", zorder=6)

    ax1.set_xlabel("x  [adim.]")
    ax1.set_ylabel("y  [adim.]")
    ax1.set_title("Zoom région L1–L2  (contours = niveaux d'énergie)")
    ax1.grid(True)
    ax1.legend(fontsize=8)

    # ── Sous-graphe droit : coupe 1D U*(x, y=0) ──
    ax2 = fig.add_subplot(1, 2, 2)
    x1d = np.linspace(0.98, 1.04, 3000, dtype=np.float64)
    # Éviter les singularités
    x1d_safe = x1d[np.abs(x1d - (1 - mu)) > 1e-4]
    omega1d = np.array([U_eff(xi, ZERO, mu) for xi in x1d_safe], dtype=np.float64)

    ax2.plot(x1d_safe, omega1d, color=C_POT, lw=1.5, label="U*(x, 0)")

    # Annotations des extrema L1, L2
    for k, col in [("L1", C_L1), ("L2", C_L2)]:
        xk = lp[k][0]
        if x1d_safe[0] <= xk <= x1d_safe[-1]:
            ok = U_eff(xk, ZERO, mu)
            ax2.plot(xk, ok, "o", color=col, ms=8, zorder=5)
            ax2.annotate(
                f"{k}\nU* = {ok:.6f}\ncol (instable)",
                xy=(xk, ok),
                xytext=(0, 18),
                textcoords="offset points",
                color=col,
                fontsize=8,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
            )

    # Hessien en L2 : afficher le signe de det
    xL2, yL2 = lp["L2"]
    H = hessian_U_eff(xL2, yL2, mu)
    detH: np.float64 = np.linalg.det(H)
    Uxx_L2: np.float64 = H[0, 0]
    Uyy_L2: np.float64 = H[1, 1]
    ax2.set_xlabel("x  [adim.]")
    ax2.set_ylabel("U*(x, 0)")
    ax2.set_title(
        f"Coupe y=0 — Hessien en L2 :\n"
        f"U_xx={Uxx_L2:.3f}  U_yy={Uyy_L2:.3f}  det={detH:.4f} < 0  =>  col"
    )
    ax2.grid(True)
    ax2.legend()

    # Flèche illustrant le col : direction instable (+x) et stable (+y) schématiques
    ymax = omega1d.max()
    ymin = omega1d.min()
    yrange = ymax - ymin

    print(f"  U*(L2) = {U_eff(xL2, yL2):.8f}")
    print(f"  Hessien U* en L2 : Uxx={Uxx_L2:.4f}, Uyy={Uyy_L2:.4f}, det={detH:.6f}")
    print(
        f"  => det < 0  <=>  point-selle (col)  <=>  instabilite topologique garantie"
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return lp


#  GRAPHE 2 — Spectre du linéarisé aux points de Lagrange
def plot_eigenvalue_spectrum(
    mu=MU,
    lp: dict[str, NDArray[np.float64]] | None = None,
    save_path: str | None = None,
):
    """
    Valeurs propres du Jacobien A(X_eq) pour chaque point de Lagrange.

    Pour un équilibre d'un système hamiltonien, les valeurs propres viennent
    par paires ±λ (structure symplectique). Trois cas :
      • ±iU (purement imaginaires) → centre → stabilité linéaire
      • ±σ  (réelles)              → selle   → instabilité exponentielle
      • ±(σ±iU) complexes          → spirale selle → instabilité

    L1, L2 : une paire réelle (±σ), deux paires imaginaires → point-selle.
    L4, L5 : trois paires imaginaires si μ < μ_Routh.

    L'exposant σ = Re(λ_max) donne le taux d'instabilité.
    Temps de doublement : τ = ln2 / σ (en unités adim. = × T_star_days en jours).
    """
    print("\n[2/4] Spectre du linéarisé...")
    if lp is None:
        lp = lagrange_points(mu)

    points_to_plot = ["L1", "L2", "L4", "L5"]
    colors_map = {"L1": C_L1, "L2": C_L2, "L4": C_L4, "L5": C_L4}

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        "Spectre du Jacobien A = ∂f/∂X aux équilibres du CR3BP", fontsize=12, y=0.98
    )

    # ── Plan complexe : valeurs propres de chaque point ──
    ax_spec = fig.add_subplot(1, 2, 1)
    ax_spec.axhline(0, color=FG, lw=0.5, alpha=0.4)
    ax_spec.axvline(0, color=FG, lw=0.5, alpha=0.4)
    # Demi-plan instable
    ax_spec.axvspan(0, 2, alpha=0.07, color="#FF4444", label="Re(λ) > 0 : instable")
    ax_spec.axvspan(-2, 0, alpha=0.07, color="#00FF88", label="Re(λ) ≤ 0 : stable ?")

    eigen_summary: dict[str, list[np.complex128]] = {}
    for name in points_to_plot:
        pos = lp[name]
        A = jacobian_cr3bp(pos[0], pos[1], ZERO, mu)
        eigs = eigvals(A)
        col = colors_map[name]

        # Trier par partie réelle décroissante
        eigs_sorted: list[np.complex128] = sorted(
            eigs, key=lambda z: z.real, reverse=True
        )
        eigen_summary[name] = eigs_sorted

        re = [z.real for z in eigs_sorted]
        im = [z.imag for z in eigs_sorted]
        ax_spec.scatter(
            re, im, color=col, s=80, zorder=5, label=f"{name}  (σ_max={max(re):+.4f})"
        )

        # Annoter la valeur propre instable (Re > 0)
        for z in eigs_sorted:
            if z.real > 1e-8:
                ax_spec.annotate(
                    f"  λ={z.real:.4f}{z.imag:+.4f}i",
                    xy=(z.real, z.imag),
                    color=col,
                    fontsize=8,
                )

    ax_spec.set_xlim(-2.0, 2.0)
    ax_spec.set_xlabel("Re(λ)")
    ax_spec.set_ylabel("Im(λ)")
    ax_spec.set_title("Plan complexe — Les λ de L1, L2 ont Re > 0")
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True)

    # ── Tableau comparatif des exposants ──
    ax_tab = fig.add_subplot(1, 2, 2)
    ax_tab.axis("off")

    headers = ["Point", "λ_instable", "σ=Re(λ)", "Temps doublement (j)", "Nature"]
    rows: list[tuple[str, str, str, str, str]] = []
    for name in points_to_plot:
        eigs = eigen_summary[name]
        sigma_max = max(z.real for z in eigs)
        lam_inst = max(eigs, key=lambda z: z.real)
        if sigma_max > 1e-8:
            tau_days = np.log(2) / sigma_max * T_STAR_DAYS
            nature = "INSTABLE (selle)"
            col = "#FF6B6B"
        else:
            tau_days = np.float64("inf")
            nature = "stable (centre)"
            col = "#90EE90"
        rows.append(
            (
                name,
                f"{lam_inst:.4f}",
                f"{sigma_max:+.6f}",
                f"{tau_days:.1f}" if tau_days < 1e6 else "∞",
                nature,
            )
        )
        print(
            f"  {name:3s} | σ_max = {sigma_max:+.6f}  |  "
            f"τ_double = {tau_days:.1f} j"
            if tau_days < 1e6
            else f"  {name:3s} | σ_max = {sigma_max:+.6f}  |  stable"
        )

    table = ax_tab.table(
        cellText=rows, colLabels=headers, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.2)

    # Colorier les lignes instables
    for i, row in enumerate(rows):
        for j in range(len(headers)):
            cell = table[i + 1, j]
            # cell.set_facecolor("#2A0A0A" if "INST" in row[4] else "#0A2A0A")
            cell.set_text_props(color=FG)
        table[i + 1, 0].set_text_props(
            color=C_L2 if rows[i][0] == "L2" else C_L1 if rows[i][0] == "L1" else C_L4,
            fontweight="bold",
        )
    for j in range(len(headers)):
        # table[0, j].set_facecolor("#1A1A2E")
        table[0, j].set_text_props(color=FG, fontweight="bold")

    ax_tab.set_title(
        "Exposants et temps caractéristiques\n"
        "(τ = ln2/σ en unités adim.  ×  T* = 58.15 j)",
        pad=12,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return eigen_summary


#  GRAPHE 3 — Variétés stable et instable de L2


def plot_stable_unstable_manifolds(
    mu=MU,
    lp: dict[str, NDArray[np.float64]] | None = None,
    save_path: str | None = None,
):
    """
    Intégration des variétés stable (W^s) et instable (W^u) de L2.

    Méthode :
    - Calculer v_s, v_u (vecteurs propres de A(L2)) correspondant aux
      valeurs propres réelles ±σ.
    - Initialiser à X_L2 ± ε·v_s  et  X_L2 ± ε·v_u.
    - Intégrer vers l'avant (W^u) ou l'arrière (W^s, équivaut à intégrer
      vers l'avant avec f → -f).
    - W^s(L2) : trajectoires qui convergent vers L2 dans le passé.
    - W^u(L2) : trajectoires qui s'éloignent de L2.

    L'épaisseur ε est choisie petite (≈ 10 km en adim.) pour rester dans le
    domaine de validité de la linéarisation.
    """
    print("\n[3/4] Variétés stable et instable...")
    if lp is None:
        lp = lagrange_points(mu)

    xL2, yL2 = lp["L2"]
    A = jacobian_cr3bp(xL2, yL2, ZERO, mu)
    eigs, vecs = np.linalg.eig(A)

    # Identifier les vecteurs propres réels stable (σ<0) et instable (σ>0)
    real_mask = np.abs(eigs.imag) < 1e-8 * (np.abs(eigs.real) + 1e-30)
    real_idx = np.where(real_mask)[0]

    # Instable : valeur propre réelle la plus POSITIVE (Re > 0)
    # Stable   : valeur propre réelle la plus NÉGATIVE (Re < 0)
    idx_u = real_idx[np.argmax(eigs.real[real_idx])]  # λ > 0
    idx_s = real_idx[np.argmin(eigs.real[real_idx])]  # λ < 0

    v_u = vecs[:, idx_u].real  #
    v_s = vecs[:, idx_s].real  #
    v_u /= np.linalg.norm(v_u)
    v_s /= np.linalg.norm(v_s)

    lam_u = eigs[idx_u].real  # positif
    lam_s = eigs[idx_s].real  # négatif

    print(f"  λ_instable = {lam_u:+.6f},  λ_stable = {lam_s:+.6f}")
    print(
        f"  Produit |λ_u·λ_s| = {abs(lam_u*lam_s):.6f}  (attendu ≠ 1 : linéarisé, pas monodromie)"
    )

    # Perturbation initiale : ~100 km en adim.
    eps = 100e3 / 1.496e11

    X0_L2 = np.array([xL2, yL2, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Durée d'intégration : ~3 révolutions halo ≈ 3 * 180 jours
    T_int = np.float64(3.0 * 2 * np.pi)  # adim.
    h = T_int / 10000

    def f_fwd(t, s):
        return eom(t, s, mu)

    def f_bwd(t, s):
        return -eom(t, s, mu)

    trajectories: dict[str, NDArray[np.float64]] = {}

    # W^u : intégrer en avant depuis X0 ± ε·v_u
    for sign, label in [(+1, "Wu_plus"), (-1, "Wu_minus")]:
        y0 = X0_L2 + sign * eps * v_u
        traj = integrate_rk4(f_fwd, y0, ZERO, T_int, h)
        trajectories[label] = traj

    # W^s : intégrer en avant depuis X0 ± ε·v_s  AVEC f inversé
    #        (équivalent à remonter le temps : les trajectoires convergeaient vers L2)
    for sign, label in [(+1, "Ws_plus"), (-1, "Ws_minus")]:
        y0 = X0_L2 + sign * eps * v_s
        traj = integrate_rk4(f_bwd, y0, ZERO, T_int, h)
        trajectories[label] = traj

    # ── Figure ──
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        "Variétés stable (W^s) et instable (W^u) de L2\n" "(repère tournant, z=0)",
        fontsize=12,
        y=0.98,
    )

    KM = np.float64(1.496e8)  # adim → km

    ax1 = fig.add_subplot(1, 2, 1)  # vue globale XY
    ax2 = fig.add_subplot(1, 2, 2)  # zoom L2

    for ax, (xlim, ylim), title in [
        (ax1, ((-0.02, 0.04), (-0.025, 0.025)), "Vue globale (XY)"),
        (ax2, ((-2000, 2000), (-1500, 1500)), "Zoom L2 [km]"),
    ]:
        is_km = "km" in title

        def coords(traj: NDArray[np.float64]) -> tuple[np.float64, np.float64]:
            x = traj[:, 0] - xL2
            y = traj[:, 1] - yL2
            if is_km:
                return x * KM, y * KM
            return x, y

        for key, col, ls, lbl in [
            ("Wu_plus", C_UNS, "-", "W^u (+)"),
            ("Wu_minus", C_UNS, "--", "W^u (-)"),
            ("Ws_plus", C_STB, "-", "W^s (+)"),
            ("Ws_minus", C_STB, "--", "W^s (-)"),
        ]:
            xc, yc = coords(trajectories[key])
            ax.plot(xc, yc, color=col, lw=0.9, ls=ls, alpha=0.85, label=lbl)

        # L2
        ax.plot(0, 0, "o", color=C_L2, ms=8, zorder=6, label="L2")

        # Vecteurs propres
        scale = 1500 if is_km else 0.008
        for v, col, lbl in [(v_u, C_UNS, "v_u"), (v_s, C_STB, "v_s")]:
            ax.annotate(
                "",
                xy=(v[0] * scale, v[1] * scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
            )
            ax.annotate(
                "",
                xy=(-v[0] * scale, -v[1] * scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
            )

        ax.set_xlabel("ΔX  [km]" if is_km else "ΔX  [adim.]")
        ax.set_ylabel("ΔY  [km]" if is_km else "ΔY  [adim.]")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(fontsize=8)
        if is_km:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return v_u, v_s, lam_u, lam_s


#  GRAPHE 4 — Croissance exponentielle de ‖δx(t)‖


def plot_lyapunov_divergence(
    mu=MU,
    lp: dict[str, NDArray[np.float64]] | None = None,
    v_u: NDArray[np.float64] | None = None,
    lam_u: np.float64 | None = None,
    save_path: str | None = None,
):
    """
    Croissance de ‖δx(t)‖ pour des perturbations initiales ε·v_u de L2.

    Deux approches parallèles :
    1. Intégration non-linéaire complète : deux trajectoires séparées de ε
       initalement dans la direction instable → mesure directe de la divergence.
    2. Intégration linéarisée : δẋ = A(X_L2)·δx → δx(t) = exp(A·t)·δx(0)
       → croissance théorique ~ε·exp(λ_u·t).

    Compare les deux pour vérifier la cohérence et illustrer la limite de
    validité de l'approximation linéaire.

    Plusieurs amplitudes de perturbation montrent la dépendance en ε.
    """
    print("\n[4/4] Croissance exponentielle (exposant de Lyapunov)...")
    if lp is None:
        lp = lagrange_points(mu)

    xL2, yL2 = lp["L2"]
    X0_L2 = np.array([xL2, yL2, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if v_u is None:
        A_mat = jacobian_cr3bp(xL2, yL2, ZERO, mu)
        eigs, vecs = np.linalg.eig(A_mat)
        real_idx = np.where(np.abs(eigs.imag) < 1e-8)[0]
        idx_u = real_idx[np.argmax(eigs.real[real_idx])]  # λ > 0
        v_u = vecs[:, idx_u].real
        v_u /= np.linalg.norm(v_u)
        lam_u = eigs[idx_u].real

    assert lam_u is not None

    A_mat = jacobian_cr3bp(xL2, yL2, ZERO, mu)

    # Durée : ~2 révolutions halo (jusqu'à ce que la non-linéarité domine)
    T_int = np.float64(2.0 * 2 * np.pi)
    h = T_int / 8000
    t_arr = np.arange(0, T_int + h, h, dtype=np.float64)
    t_days = t_arr * T_STAR_DAYS

    # Amplitudes : de ~1 km à ~10 000 km (en adim.)
    epsilons_km = np.array([1.0, 100.0, 1000.0, 10000.0], dtype=np.float64)
    epsilons = np.array([e * 1e3 / 1.496e11 for e in epsilons_km], dtype=np.float64)
    KM = np.float64(1.496e8)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Instabilité de L2 : croissance exponentielle de ‖δx‖\n"
        f"Taux théorique λ_u = {lam_u:.5f} adim.  "
        f"→  τ_double = {np.log(2)/lam_u * T_STAR_DAYS:.1f} jours",
        fontsize=12,
        y=0.99,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_log = fig.add_subplot(gs[0, :])  # ‖δx‖ en log pour toutes les ε
    ax_lin = fig.add_subplot(gs[1, 0])  # comparaison linéaire vs non-linéaire
    ax_comp = fig.add_subplot(gs[1, 1])  # exposant de Lyapunov local

    colors_eps = plt.cm.plasma(np.linspace(0.15, 0.85, len(epsilons), dtype=np.float64))

    # ── Intégration non-linéaire ──
    def f_fwd(t: np.float64, s: NDArray[np.float64]):
        return eom(t, s, mu)

    for eps, eps_km, col in zip(epsilons, epsilons_km, colors_eps):
        # Trajectoire perturbée
        y0_pert = X0_L2 + eps * v_u
        traj_pert = integrate_rk4(f_fwd, y0_pert, ZERO, T_int, h)

        # Trajectoire de référence (initialisée en L2 exactement)
        traj_ref = integrate_rk4(f_fwd, X0_L2, ZERO, T_int, h)

        n = min(len(traj_pert), len(traj_ref), len(t_arr))
        delta_norm = np.linalg.norm(traj_pert[:n, :3] - traj_ref[:n, :3], axis=1) * KM

        # Éviter log(0)
        mask = delta_norm > 1e-30
        ax_log.semilogy(
            t_days[mask],
            delta_norm[mask],
            color=col,
            lw=0.9,
            label=f"ε = {eps_km:.0f} km",
        )

    # Courbe théorique linéaire (pour ε=100 km)
    eps_ref = np.float64(100e3 / 1.496e11)
    delta_lin = eps_ref * np.exp(lam_u * t_arr) * KM
    ax_log.semilogy(
        t_days,
        delta_lin,
        color=FG,
        lw=1.2,
        ls="--",
        label=f"ε·exp(λ_u·t)  théorique\nλ_u={lam_u:.4f}",
        alpha=0.7,
    )

    ax_log.set_xlabel("Temps [jours]")
    ax_log.set_ylabel("‖δx‖  [km]")
    ax_log.set_title("Croissance de la divergence depuis L2  (échelle log)")
    ax_log.grid(True, which="both")
    ax_log.legend(fontsize=8, ncol=3)

    # ── Comparaison non-linéaire vs linéaire pour ε=100 km ──
    eps_ref_km = 100.0
    eps_ref = eps_ref_km * 1e3 / 1.496e11
    y0_pert = X0_L2 + eps_ref * v_u
    traj_pert = integrate_rk4(f_fwd, y0_pert, ZERO, T_int, h)
    traj_ref = integrate_rk4(f_fwd, X0_L2, ZERO, T_int, h)
    n = min(len(traj_pert), len(traj_ref), len(t_arr))
    delta_nonlin = np.linalg.norm(traj_pert[:n, :3] - traj_ref[:n, :3], axis=1) * KM
    delta_linear = eps_ref * np.exp(lam_u * t_arr[:n]) * KM
    ax_lin.semilogy(t_days[:n], delta_nonlin, color=C_UNS, lw=1.2, label="Non-linéaire")
    ax_lin.semilogy(
        t_days[:n],
        delta_linear,
        color=FG,
        lw=1.0,
        ls="--",
        label=f"Linéaire : ε·e^(λ_u t)",
    )

    # Marquer la fin du régime linéaire (divergence > 10 × ε_ref)
    threshold = 10 * eps_ref_km
    cross_idx = (
        np.argmax(delta_nonlin > threshold)
        if np.any(delta_nonlin > threshold)
        else None
    )
    if cross_idx:
        ax_lin.axvline(
            t_days[cross_idx],
            color="#FFD700",
            lw=0.8,
            ls=":",
            label=f"Fin régime linéaire\n(‖δx‖ > 10ε ≈ {threshold:.0f} km)",
        )

    ax_lin.set_xlabel("Temps [jours]")
    ax_lin.set_ylabel("‖δx‖  [km]")
    ax_lin.set_title(f"ε={eps_ref_km:.0f} km — Linéaire vs Non-linéaire")
    ax_lin.grid(True, which="both")
    ax_lin.legend(fontsize=8)

    # ── Exposant de Lyapunov local λ_loc(t) = d/dt ln(‖δx‖) ──
    # Estimé par différence finie sur la norme en log
    log_delta = np.log(np.maximum(delta_nonlin, 1e-40))
    # Fenêtre glissante pour lisser
    window = 50
    lam_local = np.gradient(log_delta, t_arr[:n]) / (
        2 * np.pi / T_STAR_DAYS * T_STAR_DAYS
    )
    # Convertir en adim.
    lam_local_adim = np.gradient(log_delta, t_arr[:n])

    ax_comp.plot(
        t_days[:n],
        lam_local_adim,
        color=C_POT,
        lw=0.8,
        alpha=0.7,
        label="λ_loc(t) = d/dt ln‖δx‖",
    )
    ax_comp.axhline(
        lam_u, color=FG, lw=1.2, ls="--", label=f"λ_u théorique = {lam_u:.5f}"
    )
    ax_comp.axhline(0, color=GRID, lw=0.5)
    ax_comp.set_xlabel("Temps [jours]")
    ax_comp.set_ylabel("Exposant local  [adim.]")
    ax_comp.set_title(
        "Exposant de Lyapunov local\n(converge vers λ_u en régime linéaire)"
    )
    ax_comp.set_ylim(-lam_u * 3, lam_u * 5)
    ax_comp.grid(True)
    ax_comp.legend(fontsize=8)

    print(
        f"  λ_u = {lam_u:.6f} adim.  →  τ_double = {np.log(2)/lam_u * T_STAR_DAYS:.1f} jours"
    )
    print(
        f"  Après {T_int * T_STAR_DAYS:.0f} jours, une perturbation de 100 km "
        f"→  {100*np.exp(lam_u * T_int):.2e} km"
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


#  Main


if __name__ == "__main__":
    print("=" * 64)
    print("  Analyse de l'instabilité du point de Lagrange L2")
    print("  CR3BP Soleil-Terre  |  μ = {:.4e}".format(MU))
    print("=" * 64)

    lp = plot_effective_potential(MU, save_path=None)

    eigen_summary = plot_eigenvalue_spectrum(MU, lp, save_path=None)

    v_u, v_s, lam_u, lam_s = plot_stable_unstable_manifolds(MU, lp, save_path=None)

    plot_lyapunov_divergence(MU, lp, v_u, lam_u, save_path=None)

    print("\n" + "=" * 64)
    print("  Récapitulatif physique")
    print("=" * 64)
    print(f"  μ (Soleil-Terre)      = {MU:.4e}")
    lp_vals = lagrange_points(MU)
    print(f"  x(L2)                 = {lp_vals['L2'][0]:.8f}  adim.")
    print(f"  U*(L2)               = {U_eff(lp_vals['L2'][0], ZERO):.8f}")
    A_L2 = jacobian_cr3bp(lp_vals["L2"][0], ZERO, ZERO, MU)
    H = hessian_U_eff(lp_vals["L2"][0], ZERO, MU)
    print(f"  det(Hess U*|L2)      = {np.linalg.det(H):.6f}  < 0  => col (point-selle)")
    eigs_L2 = eigvals(A_L2)
    sigma = max(z.real for z in eigs_L2)
    print(f"  lambda_max (Jacobien L2)  = {sigma:+.6f}  > 0  => instable")
    print(f"  Temps de doublement  = {np.log(2)/sigma * T_STAR_DAYS:.1f} jours")
    print(f"  Sans station-keeping, une perturbation de 100 km")
    print(
        f"  atteint ~1 UA en aprox. {np.log(1.496e8/100)/sigma * T_STAR_DAYS:.0f} jours"
    )
