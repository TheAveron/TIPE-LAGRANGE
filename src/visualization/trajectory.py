"""
trajectory.py — Graphes de trajectoires 2D et 3D.

Fournit des fonctions pour les deux modules (CR3BP et inertiel).
"""

import matplotlib.pyplot as plt
from cr3bp.simulation import CR3BPSimulation
from inertial.simulation import InertialSimulation
from mpl_toolkits.mplot3d import \
    Axes3D  # noqa: F401  (enregistrement projection 3d)

from .utils import COLORS, add_colorbar_time, set_style

# CR3BP


def plot_cr3bp_trajectory(sim: CR3BPSimulation, save_path: str | None = None):
    """
    Trajectoire 3D + projections 2D dans le repère tournant centré sur L2.

    Parameters
    ----------
    sim : CR3BPSimulation  (après run())
    """
    assert sim.times is not None

    set_style()
    pos = sim.positions
    L2 = sim.L2_position()

    # Coordonnées relatives à L2 [km]
    KM = 1.495_978_707e8  # 1 unité adim. → km  (1 UA en km)
    dpos = (pos - L2) * KM

    t_norm = sim.times / sim.times[-1]  # [0, 1] pour la colormap

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("JWST — Orbite halo autour de L2 (CR3BP, repère tournant)", y=0.98)

    # --- 3D ---
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax3.set_facecolor(COLORS["bg"])
    sc = ax3.scatter(
        dpos[:, 0], dpos[:, 1], dpos[:, 2], c=t_norm, cmap="plasma", s=0.5, alpha=0.8
    )
    ax3.scatter(0, 0, 0, color=COLORS["L2"], s=60, zorder=5, label="L2")
    ax3.set_xlabel("ΔX [km]")
    ax3.set_ylabel("ΔY [km]")
    ax3.set_zlabel("ΔZ [km]")
    ax3.set_title("Vue 3D")
    ax3.legend(fontsize=8)

    # --- XY ---
    ax_xy = fig.add_subplot(2, 2, 2)
    sc2 = ax_xy.scatter(dpos[:, 0], dpos[:, 1], c=t_norm, cmap="plasma", s=0.5)
    ax_xy.scatter(0, 0, color=COLORS["L2"], s=60, zorder=5, label="L2")
    ax_xy.set_xlabel("ΔX [km]")
    ax_xy.set_ylabel("ΔY [km]")
    ax_xy.set_title("Plan XY (écliptique)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True)
    ax_xy.legend(fontsize=8)
    add_colorbar_time(fig, ax_xy, sc2, label=f"Temps [0 → {sim.times_days[-1]:.0f} j]")

    # --- XZ ---
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.axis("equal")
    ax_xz.scatter(dpos[:, 0], dpos[:, 2], c=t_norm, cmap="plasma", s=0.5)
    ax_xz.scatter(0, 0, color=COLORS["L2"], s=60, zorder=5)
    ax_xz.set_xlabel("ΔX [km]")
    ax_xz.set_ylabel("ΔZ [km]")
    ax_xz.set_title("Plan XZ (hors-plan)")
    ax_xz.set_aspect("equal")
    ax_xz.grid(True)

    # --- YZ ---
    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.axis("equal")
    ax_yz.scatter(dpos[:, 1], dpos[:, 2], c=t_norm, cmap="plasma", s=0.5)
    ax_yz.scatter(0, 0, color=COLORS["L2"], s=60, zorder=5)
    ax_yz.set_xlabel("ΔY [km]")
    ax_yz.set_ylabel("ΔZ [km]")
    ax_yz.set_title("Plan YZ")
    ax_yz.set_aspect("equal")
    ax_yz.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# Inertiel
def plot_inertial_trajectory(sim: InertialSimulation, save_path: str | None = None):
    """
    Trajectoire JWST + orbite Terre dans le référentiel inertiel [UA].

    Parameters
    ----------
    sim : InertialSimulation  (après run())
    """
    assert sim.times is not None

    set_style()
    UA = 1.495_978_707e11  # m → UA

    pos_jwst = sim.positions / UA
    pos_earth = sim.earth_positions() / UA
    t_norm = sim.times / sim.times[-1]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("JWST — Trajectoire dans le référentiel inertiel J2000", y=0.98)

    # --- 3D ---
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax3.set_facecolor(COLORS["bg"])
    ax3.scatter(
        pos_jwst[:, 0],
        pos_jwst[:, 1],
        pos_jwst[:, 2],
        c=t_norm,
        cmap="plasma",
        s=0.5,
        alpha=0.8,
        label="JWST",
    )
    ax3.plot(
        pos_earth[:, 0],
        pos_earth[:, 1],
        pos_earth[:, 2],
        color=COLORS["earth"],
        lw=0.8,
        alpha=0.6,
        label="Terre",
    )
    # ax3.scatter(0, 0, 0, color=COLORS["sun"], s=80, zorder=5, label="Soleil")
    ax3.set_xlabel("X [UA]")
    ax3.set_ylabel("Y [UA]")
    ax3.set_zlabel("Z [UA]")
    ax3.set_title("Vue 3D")
    ax3.legend(fontsize=8)

    # --- XY ---
    ax_xy = fig.add_subplot(2, 2, 2)
    sc = ax_xy.scatter(
        pos_jwst[:, 0], pos_jwst[:, 1], c=t_norm, cmap="plasma", s=0.5, label="JWST"
    )
    ax_xy.plot(
        pos_earth[:, 0],
        pos_earth[:, 1],
        color=COLORS["earth"],
        lw=0.8,
        alpha=0.6,
        label="Terre",
    )
    # ax_xy.scatter(0, 0, color=COLORS["sun"], s=80, zorder=5, label="Soleil")
    ax_xy.set_xlabel("X [UA]")
    ax_xy.set_ylabel("Y [UA]")
    ax_xy.set_title("Plan XY (écliptique)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True)
    ax_xy.legend(fontsize=8)
    add_colorbar_time(fig, ax_xy, sc, label=f"Temps [0 → {sim.times_days[-1]:.0f} j]")

    # --- Zoom autour de L2 (XY) ---
    ax_zoom = fig.add_subplot(2, 2, 3)
    ax_zoom.scatter(pos_jwst[:, 0], pos_jwst[:, 1], c=t_norm, cmap="plasma", s=0.5)
    # Centre approximatif de L2 (≈ Terre + 0.01 UA dans la direction radiale)
    ax_zoom.plot(
        pos_earth[:, 0], pos_earth[:, 1], color=COLORS["earth"], lw=0.8, alpha=0.4
    )
    ax_zoom.set_xlabel("X [UA]")
    ax_zoom.set_ylabel("Y [UA]")
    ax_zoom.set_title("Zoom région L2")
    ax_zoom.grid(True)
    # Centrage automatique sur le barycentre de la trajectoire JWST
    cx, cy = pos_jwst[:, 0].mean(), pos_jwst[:, 1].mean()
    half = 0.025
    ax_zoom.set_xlim(cx - half, cx + half)
    ax_zoom.set_ylim(cy - half, cy + half)
    ax_zoom.set_aspect("equal")

    # --- XZ ---
    ax_xz = fig.add_subplot(2, 2, 4)
    ax_xz.axis("equal")
    ax_xz.scatter(pos_jwst[:, 0], pos_jwst[:, 2], c=t_norm, cmap="plasma", s=0.5)
    ax_xz.set_xlabel("X [UA]")
    ax_xz.set_ylabel("Z [UA]")
    ax_xz.set_title("Plan XZ (inclinaison)")
    ax_xz.grid(True)
    ax_xz.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
