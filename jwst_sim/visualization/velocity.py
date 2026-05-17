"""
velocity.py — Graphes de vitesse du JWST.
"""

import matplotlib.pyplot as plt
import numpy as np

from .utils import COLORS, annotate_extrema, set_style


def plot_cr3bp_velocity(sim, save_path: str | None = None):
    """
    Norme et composantes de la vitesse adim. (CR3BP).

    Parameters
    ----------
    sim : CR3BPSimulation  (après run())
    """
    set_style()
    t = sim.times_days
    vel = sim.velocities
    spd = sim.speeds

    # Facteur de conversion adim. → km/s
    # v* = l*/t* = 1 UA / (T_terre/2π) ≈ 29.78 km/s (vitesse orbitale Terre)
    V_STAR_KMS = 29.784_691  # km/s

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("CR3BP — Vitesse du JWST (repère tournant)", y=0.98)

    ax = axes[0]
    ax.plot(t, spd * V_STAR_KMS, color=COLORS["speed"], lw=1.0, label="|v|")
    ax.set_ylabel("|v| [km/s]")
    ax.set_title("Norme de la vitesse")
    ax.grid(True)
    ax.legend()
    annotate_extrema(ax, t, spd * V_STAR_KMS)

    ax2 = axes[1]
    for i, (comp, col) in enumerate(
        zip(["vx", "vy", "vz"], ["#FF6B6B", "#90EE90", "#87CEEB"])
    ):
        ax2.plot(t, vel[:, i] * V_STAR_KMS, color=col, lw=0.9, label=comp)
    ax2.axhline(0, color=COLORS["text"], lw=0.4, ls="--", alpha=0.3)
    ax2.set_xlabel("Temps [jours]")
    ax2.set_ylabel("v [km/s]")
    ax2.set_title("Composantes de la vitesse")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_inertial_velocity(sim, save_path: str | None = None):
    """
    Norme et composantes de la vitesse SI [km/s] (inertiel).

    Parameters
    ----------
    sim : InertialSimulation  (après run())
    """
    set_style()
    t = sim.times_days
    vel = sim.velocities / 1000.0  # m/s → km/s
    spd = sim.speeds / 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("Inertiel — Vitesse du JWST (référentiel J2000)", y=0.98)

    ax = axes[0]
    ax.plot(t, spd, color=COLORS["speed"], lw=1.0, label="|v|")
    ax.set_ylabel("|v| [km/s]")
    ax.set_title("Norme de la vitesse")
    ax.grid(True)
    ax.legend()
    annotate_extrema(ax, t, spd)

    ax2 = axes[1]
    for i, (comp, col) in enumerate(
        zip(["vx", "vy", "vz"], ["#FF6B6B", "#90EE90", "#87CEEB"])
    ):
        ax2.plot(t, vel[:, i], color=col, lw=0.9, label=comp)
    ax2.axhline(0, color=COLORS["text"], lw=0.4, ls="--", alpha=0.3)
    ax2.set_xlabel("Temps [jours]")
    ax2.set_ylabel("v [km/s]")
    ax2.set_title("Composantes de la vitesse")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_velocity_comparison(sim_cr3bp, sim_inertial, save_path: str | None = None):
    """
    Vitesse scalaire JWST : CR3BP vs inertiel sur le même graphe.
    Les deux sont convertis en km/s.
    """
    set_style()
    V_STAR_KMS = 29.784_691

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Comparaison des vitesses JWST : CR3BP vs Inertiel")

    ax.plot(
        sim_cr3bp.times_days,
        sim_cr3bp.speeds * V_STAR_KMS,
        color=COLORS["jacobi"],
        label="CR3BP (repère tournant)",
        lw=1.0,
    )
    ax.plot(
        sim_inertial.times_days,
        sim_inertial.speeds / 1000.0,
        color=COLORS["speed"],
        label="Inertiel J2000",
        lw=1.0,
        ls="--",
    )

    ax.set_xlabel("Temps [jours]")
    ax.set_ylabel("|v| [km/s]")
    ax.grid(True)
    ax.legend()
    ax.set_title("Note : vitesses dans des repères différents — comparaison indicative")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
