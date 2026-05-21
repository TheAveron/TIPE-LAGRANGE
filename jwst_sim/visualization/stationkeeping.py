"""
stationkeeping.py — Graphes spécifiques à la simulation avec corrections EVSK.
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import set_style, COLORS, annotate_extrema


def plot_sk_trajectory(sim_sk, save_path: str | None = None):
    """
    Trajectoires comparées : avec corrections / sans corrections / référence.
    Vue 3D + projections 2D dans le repère tournant centré sur L2.
    """
    set_style()
    KM = 1.496e8  # adim → km

    from cr3bp.lagrange import lagrange_L2

    L2 = lagrange_L2(sim_sk.mu)

    pos_sk = (sim_sk.positions - L2) * KM
    pos_free = (sim_sk.positions_free - L2) * KM
    pos_ref = (sim_sk.positions_ref - L2) * KM
    t_norm = sim_sk.times / sim_sk.times[-1]

    v_s = np.asarray(sim_sk.v_s_list)[:, :3]
    v_s = v_s / np.linalg.norm(v_s, axis=1, keepdims=True)

    step = max(len(pos_sk) // 200, 1)
    idx = np.arange(0, len(pos_sk), step)

    # Positions des manœuvres
    man_pos = (
        np.array(
            [(sim_sk._ref_state(m.t_adim)[:3] - L2) * KM for m in sim_sk.maneuvers]
        )
        if sim_sk.maneuvers
        else None
    )

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("JWST — Station-Keeping EVSK (repère tournant, centré sur L2)")

    # --- 3D ---
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax3.set_facecolor(COLORS["bg"])
    ax3.plot(
        pos_ref[:, 0],
        pos_ref[:, 1],
        pos_ref[:, 2],
        color=COLORS["L2"],
        lw=0.8,
        alpha=0.5,
        label="Référence",
    )
    ax3.scatter(
        pos_sk[:, 0],
        pos_sk[:, 1],
        pos_sk[:, 2],
        c=t_norm,
        cmap="plasma",
        s=0.4,  # type: ignore
        label="Avec SK",
    )

    ax3.plot(
        pos_free[:, 0],
        pos_free[:, 1],
        pos_free[:, 2],
        color="#888888",
        lw=0.5,
        alpha=0.4,
        label="Sans SK",
    )
    if man_pos is not None:
        ax3.scatter(
            man_pos[:, 0],
            man_pos[:, 1],
            man_pos[:, 2],  # type: ignore
            color="white",
            s=20,
            zorder=6,
            marker="x",
            label="Manœuvre",
        )
    ax3.set_xlabel("ΔX [km]")
    ax3.set_ylabel("ΔY [km]")
    ax3.set_zlabel("ΔZ [km]")
    ax3.set_title("Vue 3D")
    ax3.legend(fontsize=7)

    # --- XY ---
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(pos_ref[:, 0], pos_ref[:, 1], color=COLORS["L2"], lw=0.8, alpha=0.5)
    ax_xy.scatter(pos_sk[:, 0], pos_sk[:, 1], c=t_norm, cmap="plasma", s=0.4)
    ax_xy.plot(pos_free[:, 0], pos_free[:, 1], color="#888888", lw=0.5, alpha=0.4)
    if man_pos is not None:
        ax_xy.scatter(
            man_pos[:, 0], man_pos[:, 1], color="white", s=20, marker="x", zorder=6
        )
    ax_xy.set_xlabel("ΔX [km]")
    ax_xy.set_ylabel("ΔY [km]")
    ax_xy.set_title("Plan XY")
    ax_xy.grid(True)
    ax_xy.set_aspect("equal")

    # --- Erreur de position ---
    ax_err = fig.add_subplot(2, 2, 3)
    err_sk = sim_sk.position_errors
    err_free = np.linalg.norm(pos_free - pos_ref, axis=1)
    t_days = sim_sk.times_days
    ax_err.semilogy(t_days, err_sk, color=COLORS["jwst"], lw=1.0, label="Avec SK")
    ax_err.semilogy(
        t_days, err_free, color="#888888", lw=0.8, alpha=0.7, label="Sans SK"
    )
    if sim_sk.maneuvers:
        for m in sim_sk.maneuvers:
            ax_err.axvline(m.t_days, color="white", lw=0.4, alpha=0.3)
    ax_err.set_xlabel("Temps [jours]")
    ax_err.set_ylabel("Erreur position [km]")
    ax_err.set_title("Erreur par rapport à la référence")
    ax_err.grid(True)
    ax_err.legend()

    # --- YZ ---
    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.axis("equal")
    ax_yz.plot(pos_ref[:, 1], pos_ref[:, 2], color=COLORS["L2"], lw=0.8, alpha=0.5)
    ax_yz.scatter(pos_sk[:, 1], pos_sk[:, 2], c=t_norm, cmap="plasma", s=0.4)
    ax_yz.plot(pos_free[:, 1], pos_free[:, 2], color="#888888", lw=0.5, alpha=0.4)
    ax_yz.set_xlabel("ΔY [km]")
    ax_yz.set_ylabel("ΔZ [km]")
    ax_yz.set_title("Plan YZ")
    ax_yz.grid(True)
    ax_yz.set_aspect("equal")

    scale = 200e3  # ajuster visuellement

    for m, v in zip(sim_sk.maneuvers, sim_sk.v_s_list):
        # retrouver l'indice temporel correspondant
        i = np.argmin(np.abs(sim_sk.times - m.t_adim))

        v = v[:3]
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n

        ax3.quiver(
            pos_sk[i, 0],
            pos_sk[i, 1],
            pos_sk[i, 2],
            v[0],
            v[1],
            v[2],
            length=int(scale),
            color="cyan",
            alpha=0.8,
            linewidth=1.0,
        )

        ax_xy.quiver(
            pos_sk[i, 0],
            pos_sk[i, 1],
            v[0],
            v[1],
            color="cyan",
            alpha=0.8,
            linewidth=1.0,
        )

        ax_yz.quiver(
            pos_sk[i, 1],
            pos_sk[i, 2],
            v[1],
            v[2],
            color="cyan",
            alpha=0.8,
            linewidth=0.4,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_delta_v_history(sim_sk, save_path: str | None = None):
    """
    Historique des ΔV : magnitude par manœuvre et cumulé.
    """
    set_style()
    if not sim_sk.maneuvers:
        print("Aucune manœuvre enregistrée.")
        return

    t_man = sim_sk.maneuver_times_days
    dvs = sim_sk.dv_norms_ms
    cumul = np.cumsum(dvs)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("EVSK — Historique des manœuvres de station-keeping")

    ax = axes[0]
    ax.bar(
        t_man,
        dvs,
        width=1.5,
        color=COLORS["speed"],
        alpha=0.8,
        label="|ΔV| par manœuvre",
    )
    ax.axhline(
        dvs.mean(),
        color=COLORS["text"],
        lw=0.8,
        ls="--",
        label=f"Moyenne = {dvs.mean():.4f} m/s",
    )
    ax.set_ylabel("|ΔV| [m/s]")
    ax.set_title(
        f"Budget total : {sim_sk.total_dv_ms:.3f} m/s  " f"(JWST réel : ~2.5 m/s/an)"
    )
    ax.grid(True)
    ax.legend()
    annotate_extrema(ax, t_man, dvs, n=1)

    ax2 = axes[1]
    ax2.plot(t_man, cumul, color=COLORS["energy"], lw=1.2, marker="o", ms=3)
    ax2.set_xlabel("Temps [jours]")
    ax2.set_ylabel("ΔV cumulé [m/s]")
    ax2.set_title("Budget ΔV cumulé")
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sk_jacobi(sim_sk, save_path: str | None = None):
    """
    Constante de Jacobi avec les instants de manœuvre marqués.
    Les manœuvres brisent la conservation de C (ΔC = ΔV · dC/dv ≠ 0 en général).
    """
    set_style()
    t = sim_sk.times_days
    C = sim_sk.jacobi

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Station-Keeping — Constante de Jacobi (avec sauts aux manœuvres)")

    ax.plot(t, C, color=COLORS["jacobi"], lw=0.8)
    ax.axhline(
        C[0], color=COLORS["text"], lw=0.5, ls="--", alpha=0.4, label=f"C₀ = {C[0]:.6f}"
    )
    for m in sim_sk.maneuvers:
        ax.axvline(m.t_days, color=COLORS["speed"], lw=0.5, alpha=0.5)
    ax.set_xlabel("Temps [jours]")
    ax.set_ylabel("C (constante de Jacobi)")
    ax.set_title("Les traits verticaux indiquent les manœuvres")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
