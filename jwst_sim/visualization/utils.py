"""
utils.py — Style, couleurs et fonctions communes pour les graphes.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Palette
COLORS = {
    "jwst": "#00BFFF",  # bleu ciel
    "earth": "#3A9BD5",  # bleu Terre
    "sun": "#FFD700",  # or Soleil
    "L2": "#FF6B6B",  # rouge L2
    "energy": "#90EE90",  # vert énergie
    "jacobi": "#DDA0DD",  # violet Jacobi
    "speed": "#FFA07A",  # orange vitesse
    "grid": "#2A2A3E",  # "#636374",
    "bg": "#FFFFFF",  # "#E0E0E0",
    "text": "#0D0D1A",
}


# Thème global
def set_style():
    """Applique le thème à tous les graphes."""
    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "legend.facecolor": "#FFFFFF",  # "#1A1A2E",
            "legend.edgecolor": COLORS["text"],
            "font.family": "monospace",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "lines.linewidth": 1.2,
        }
    )


# Helpers


def make_fig(nrows: int = 1, ncols: int = 1, **kwargs) -> tuple:
    """Crée une figure avec le thème appliqué."""
    set_style()
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    return fig, axes


def add_colorbar_time(fig, ax, scatter, label: str = "Temps"):
    """Ajoute une colorbar temporelle à un scatter plot."""
    cb = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label(label, color=COLORS["text"])
    cb.ax.yaxis.set_tick_params(color=COLORS["text"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=COLORS["text"])


def relative_drift(arr: NDArray[np.float64]) -> np.float64:
    """Drift relatif d'une quantité censée être conservée."""
    return (arr.max() - arr.min()) / abs(arr[0])


def annotate_extrema(
    ax, x: NDArray[np.float64], y: NDArray[np.float64], label: str = "", n: int = 1
):
    """Annote les n extrema globaux (max et min) sur un axe."""
    idx_max = np.argmax(y)
    idx_min = np.argmin(y)
    for idx, tag in [(idx_max, "max"), (idx_min, "min")]:
        ax.annotate(
            f"{label}{tag}={y[idx]:.3e}",
            xy=(x[idx], y[idx]),
            xytext=(10, 10),
            textcoords="offset points",
            color=COLORS["text"],
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=0.8),
        )
