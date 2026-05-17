"""
energy.py — Graphes d'énergie / constante de Jacobi.

- CR3BP     : constante de Jacobi C(t), dérive relative ΔC/C₀
- Inertiel  : énergie mécanique spécifique E(t), dérive relative ΔE/E₀
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import set_style, COLORS, annotate_extrema, relative_drift


def plot_jacobi(sim, save_path: str | None = None):
    """
    Constante de Jacobi au cours du temps (CR3BP).

    Parameters
    ----------
    sim : CR3BPSimulation  (après run())
    """
    set_style()
    t = sim.times_days
    C = sim.jacobi
    drift = relative_drift(C)

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle("CR3BP — Conservation de la constante de Jacobi", y=0.98)

    # Valeur absolue
    ax = axes[0]
    ax.plot(t, C, color=COLORS["jacobi"], lw=1.0)
    ax.axhline(
        C[0], color=COLORS["text"], lw=0.6, ls="--", alpha=0.5, label=f"C₀ = {C[0]:.6f}"
    )
    ax.set_ylabel("C (constante de Jacobi)")
    ax.set_title(f"Dérive relative : {100*drift:.4f} %")
    ax.grid(True)
    ax.legend()
    annotate_extrema(ax, t, C)

    # Dérive absolue
    ax2 = axes[1]
    ax2.plot(t, C - C[0], color=COLORS["jacobi"], lw=0.9)
    ax2.axhline(0, color=COLORS["text"], lw=0.5, ls="--", alpha=0.4)
    ax2.set_xlabel("Temps [jours]")
    ax2.set_ylabel("ΔC = C(t) − C₀")
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_energy(sim, save_path: str | None = None):
    """
    Énergie mécanique spécifique au cours du temps (référentiel inertiel).

    Parameters
    ----------
    sim : InertialSimulation  (après run())
    """
    set_style()
    t = sim.times_days
    E = sim.energy
    drift = relative_drift(E)

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle("Inertiel — Conservation de l'énergie mécanique spécifique", y=0.98)

    ax = axes[0]
    ax.plot(t, E, color=COLORS["energy"], lw=1.0)
    ax.axhline(
        E[0],
        color=COLORS["text"],
        lw=0.6,
        ls="--",
        alpha=0.5,
        label=f"E₀ = {E[0]:.4e} J/kg",
    )
    ax.set_ylabel("E spécifique [J/kg]")
    ax.set_title(f"Dérive relative : {100*drift:.4f} %")
    ax.grid(True)
    ax.legend()
    annotate_extrema(ax, t, E)

    ax2 = axes[1]
    ax2.plot(t, E - E[0], color=COLORS["energy"], lw=0.9)
    ax2.axhline(0, color=COLORS["text"], lw=0.5, ls="--", alpha=0.4)
    ax2.set_xlabel("Temps [jours]")
    ax2.set_ylabel("ΔE = E(t) − E₀ [J/kg]")
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_energy_comparison(sim_cr3bp, sim_inertial, save_path: str | None = None):
    """
    Comparaison des dérives de conservation entre les deux simulations.
    Affiche ΔC/C₀ et ΔE/E₀ sur le même graphe (en %).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Comparaison des dérives numériques CR3BP vs Inertiel")

    C = sim_cr3bp.jacobi
    E = sim_inertial.energy
    t_cr3 = sim_cr3bp.times_days
    t_in = sim_inertial.times_days

    ax.plot(
        t_cr3,
        100 * (C - C[0]) / abs(C[0]),
        color=COLORS["jacobi"],
        label="ΔC/C₀ (CR3BP)",
        lw=1.0,
    )
    ax.plot(
        t_in,
        100 * (E - E[0]) / abs(E[0]),
        color=COLORS["energy"],
        label="ΔE/E₀ (Inertiel)",
        lw=1.0,
    )
    ax.axhline(0, color=COLORS["text"], lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Temps [jours]")
    ax.set_ylabel("Dérive relative [%]")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
