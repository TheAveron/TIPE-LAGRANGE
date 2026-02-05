"""
Module de visualisation des points de Lagrange.

Ce module génère des graphiques pour comprendre :
    - Positions des 5 points de Lagrange
    - Contours du potentiel effectif
    - Courbes de vitesse nulle (Hill's regions)
    - Stabilité locale (champs de vecteurs)
    - Comparaison Soleil-Terre vs Terre-Lune

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple, Optional

from src.simulation.lagrange_points import (
    LagrangePointCalculator,
    LagrangePoint,
    LagrangePointInfo,
)
from src.simulation.constants import Constants

# Configuration matplotlib globale
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


class LagrangeVisualizer:
    """
    Classe pour visualiser les points de Lagrange et la dynamique du CRTBP.

    Cette classe génère différents types de graphiques :
    1. Position des 5 points de Lagrange
    2. Contours du potentiel effectif Ω(x,y)
    3. Courbes de vitesse nulle (pour différentes valeurs de C)
    4. Champs de vecteurs (stabilité locale)
    5. Comparaison entre systèmes (Soleil-Terre, Terre-Lune)
    """

    def __init__(
        self,
        mu: float,
        system_name: str = "Système",
        distance_unit: float = Constants.AU,
        normalized: bool = True,
    ):
        """
        Initialise le visualiseur.

        Args:
            mu: Paramètre de masse μ = m₂/(m₁+m₂)
            system_name: Nom du système (ex: "Soleil-Terre")
            distance_unit: Unité de distance physique (défaut: AU)
            normalized: Si True, utilise coordonnées normalisées

        Note:
            En coordonnées normalisées :
                - Distance primaire-secondaire = 1
                - Primaire 1 en x = -μ
                - Primaire 2 en x = 1-μ
        """
        self.mu = mu
        self.system_name = system_name
        self.distance_unit = distance_unit
        self.normalized = normalized

        self.calculator = LagrangePointCalculator(
            self.mu, self.distance_unit, self.normalized
        )

        self.lagrange_points = self.calculator.compute_all_lagrange_points()

        self.x1 = -mu
        self.x2 = 1.0 - mu

    def effective_potential(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcule le potentiel effectif Ω(x,y) dans le référentiel tournant.

        Dans le CRTBP, le potentiel effectif est :

            Ω(x,y,z) = (x² + y²)/2 + (1-μ)/r₁ + μ/r₂ + μ(1-μ)/2

        où :
            r₁ = √[(x+μ)² + y² + z²]  # distance au primaire 1
            r₂ = √[(x-1+μ)² + y² + z²]  # distance au primaire 2

        Pour le plan z=0 :
            Ω(x,y) = (x² + y²)/2 + (1-μ)/r₁ + μ/r₂ + μ(1-μ)/2

        Interprétation physique :
            - Premier terme : potentiel centrifuge
            - Deuxième terme : potentiel gravitationnel du primaire 1
            - Troisième terme : potentiel gravitationnel du primaire 2
            - Dernier terme : constante (pour que Ω=0 au barycentre)

        Args:
            x, y: Grilles de coordonnées (arrays 2D)

        Returns:
            Valeurs de Ω sur la grille
        """
        r1 = np.sqrt((x + self.mu) ** 2 + y**2)
        r2 = np.sqrt((x - 1.0 + self.mu) ** 2 + y**2)

        r1 = np.maximum(r1, 1e-6)
        r2 = np.maximum(r2, 1e-6)

        omega = (
            0.5 * (x**2 + y**2)  # Centrifuge
            + (1.0 - self.mu) / r1  # Gravité primaire 1
            + self.mu / r2  # Gravité primaire 2
            + 0.5 * self.mu * (1.0 - self.mu)  # Constante
        )

        return omega

    def jacobi_constant_curve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcule la constante de Jacobi C = 2Ω(x,y) - v².

        Pour v=0 (courbes de vitesse nulle) :
            C = 2Ω(x,y)

        Ces courbes délimitent les régions accessibles/interdites.

        Args:
            x, y: Grilles de coordonnées

        Returns:
            Valeurs de C sur la grille
        """
        return 2.0 * self.effective_potential(x, y)

    def plot_lagrange_points(
        self, figsize: Tuple[float, float] = (12, 10), save_path: Optional[str] = None
    ) -> plt.Figure:  # type: ignore
        """
        Graphique de synthèse : positions des 5 points de Lagrange.

        Affiche :
            - Les 2 primaires (tailles proportionnelles aux masses)
            - Les 5 points de Lagrange (couleurs selon stabilité)
            - Contours du potentiel effectif
            - Courbes de vitesse nulle (Hill's regions)

        Args:
            figsize: Dimensions de la figure
            save_path: Chemin pour sauvegarder (optionnel)

        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)

        x_range = np.linspace(-1.5, 1.5, 400)
        y_range = np.linspace(-1.5, 1.5, 400)
        X, Y = np.meshgrid(x_range, y_range)

        Omega = self.effective_potential(X, Y)
        Omega_safe = np.clip(Omega, np.percentile(Omega, 1), np.percentile(Omega, 99))

        contour = ax.contourf(
            X, Y, Omega_safe, levels=30, cmap="viridis", alpha=0.6, extend="both"
        )
        contour_lines = ax.contour(
            X, Y, Omega_safe, levels=15, colors="white", alpha=0.3, linewidths=0.5
        )

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Potentiel effectif Ω(x,y)", rotation=270, labelpad=20)

        # Primaires
        # Taille proportionnelle à la masse
        r1_plot = 0.05 * (1.0 - self.mu) ** 0.3
        r2_plot = 0.05 * self.mu**0.3

        primary1 = Circle(
            (self.x1, 0),
            r1_plot,
            color="yellow",
            ec="orange",
            linewidth=2,
            zorder=10,
            label=f"Primaire 1 (m={1-self.mu:.4f})",
        )
        primary2 = Circle(
            (self.x2, 0),
            r2_plot,
            color="blue",
            ec="darkblue",
            linewidth=2,
            zorder=10,
            label=f"Primaire 2 (m={self.mu:.6f})",
        )
        ax.add_patch(primary1)
        ax.add_patch(primary2)

        # Points de Lagrange
        colors = {
            LagrangePoint.L1: "red",
            LagrangePoint.L2: "red",
            LagrangePoint.L3: "red",
            LagrangePoint.L4: "green",
            LagrangePoint.L5: "green",
        }

        markers = {
            LagrangePoint.L1: "X",
            LagrangePoint.L2: "X",
            LagrangePoint.L3: "X",
            LagrangePoint.L4: "o",
            LagrangePoint.L5: "o",
        }

        for point, info in self.lagrange_points.items():
            pos = info.position
            if not self.normalized:
                # Convertir en coordonnées normalisées pour l'affichage
                pos = pos / self.distance_unit

            ax.plot(
                pos[0],
                pos[1],
                marker=markers[point],
                markersize=12,
                color=colors[point],
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=15,
                label=f"{point.value} ({info.stability.value})",
            )

            # Annotation
            ax.annotate(
                point.value,
                xy=(pos[0], pos[1]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                zorder=20,
            )

        # Courbes de vitesse nulle (Hill's regions)
        C_values = [
            self.lagrange_points[LagrangePoint.L1].jacobi_constant,
            self.lagrange_points[LagrangePoint.L2].jacobi_constant,
            self.lagrange_points[LagrangePoint.L3].jacobi_constant,
        ]

        C_grid = self.jacobi_constant_curve(X, Y)

        for i, C_val in enumerate(C_values):
            ax.contour(
                X,
                Y,
                C_grid,
                levels=[C_val],
                colors=["cyan", "magenta", "lime"][i],
                linewidths=2,
                linestyles="--",
                alpha=0.8,
            )

        ax.set_xlabel("x (normalisé)", fontsize=12)
        ax.set_ylabel("y (normalisé)", fontsize=12)
        ax.set_title(
            f"Points de Lagrange : {self.system_name}\n" f"μ = {self.mu:.6e}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Graphique sauvegardé : {save_path}")

        return fig

    def plot_hills_regions(
        self,
        C_values: Optional[List[float]] = None,
        figsize: Tuple[float, float] = (14, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:  # type: ignore
        """
        Graphique des courbes de Hill (régions accessibles/interdites).

        Pour une valeur donnée de C, les régions où C > 2Ω sont interdites
        (vitesse requise serait imaginaire).

        Les courbes C = 2Ω délimitent donc les "Hill's regions".

        Évolution avec C :
            - C très grand : région accessible très petite autour des primaires
            - C = C_L1 : ouverture entre les primaires
            - C = C_L2 : ouverture au-delà du secondaire
            - C = C_L3 : ouverture au-delà du primaire
            - C petit : presque tout l'espace accessible

        Args:
            C_values: Valeurs de C à tracer (si None, valeurs automatiques)
            figsize: Dimensions de la figure
            save_path: Chemin de sauvegarde

        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)

        x_range = np.linspace(-1.8, 1.8, 500)
        y_range = np.linspace(-1.5, 1.5, 500)
        X, Y = np.meshgrid(x_range, y_range)

        C_grid = self.jacobi_constant_curve(X, Y)

        if C_values is None:
            C_L1 = self.lagrange_points[LagrangePoint.L1].jacobi_constant
            C_L2 = self.lagrange_points[LagrangePoint.L2].jacobi_constant
            C_L3 = self.lagrange_points[LagrangePoint.L3].jacobi_constant
            C_L4 = self.lagrange_points[LagrangePoint.L4].jacobi_constant

            C_values = [
                C_L3,
                C_L1,
                C_L2,
                C_L4,
                C_L2 + 0.5,
            ]

        im = ax.imshow(
            C_grid,
            extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],  # type: ignore
            origin="lower",
            cmap="coolwarm",
            alpha=0.5,
            aspect="auto",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Constante de Jacobi C = 2Ω", rotation=270, labelpad=20)

        # Courbes de niveau pour chaque C
        colors = plt.cm.viridis(np.linspace(0, 1, len(C_values)))  # type: ignore

        for i, C_val in enumerate(C_values):
            contour = ax.contour(
                X,
                Y,
                C_grid,
                levels=[C_val],
                colors=[colors[i]],
                linewidths=2.5,
                linestyles="-",
            )

            ax.clabel(contour, inline=True, fontsize=9, fmt=f"C={C_val:.3f}")

        ax.plot(
            self.x1,
            0,
            "o",
            markersize=15,
            color="gold",
            markeredgecolor="orange",
            markeredgewidth=2,
            label="Primaire 1",
            zorder=10,
        )
        ax.plot(
            self.x2,
            0,
            "o",
            markersize=10,
            color="dodgerblue",
            markeredgecolor="navy",
            markeredgewidth=2,
            label="Primaire 2",
            zorder=10,
        )

        for point in [LagrangePoint.L1, LagrangePoint.L2, LagrangePoint.L3]:
            pos = self.lagrange_points[point].position
            if not self.normalized:
                pos = pos / self.distance_unit
            ax.plot(pos[0], pos[1], "rX", markersize=12, markeredgewidth=2, zorder=15)

        for point in [LagrangePoint.L4, LagrangePoint.L5]:
            pos = self.lagrange_points[point].position
            if not self.normalized:
                pos = pos / self.distance_unit
            ax.plot(pos[0], pos[1], "go", markersize=10, markeredgewidth=2, zorder=15)

        ax.set_xlabel("x (normalisé)", fontsize=12)
        ax.set_ylabel("y (normalisé)", fontsize=12)
        ax.set_title(
            f"Courbes de Hill (régions accessibles)\n{self.system_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(loc="upper right", fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Graphique sauvegardé : {save_path}")

        return fig

    def plot_stability_field(
        self,
        point: LagrangePoint,
        window_size: float = 0.3,
        figsize: Tuple[float, float] = (10, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:  # type: ignore
        """
        Champ de vecteurs autour d'un point de Lagrange pour visualiser la stabilité.

        On calcule la dérivée temporelle de (x, y) :
            dx/dt = vx
            dy/dt = vy
            dvx/dt = 2ωvy + ∂Ω/∂x
            dvy/dt = -2ωvx + ∂Ω/∂y

        Pour une particule au repos (vx=vy=0), seules les forces gravitationnelles
        et centrifuges agissent :
            dvx/dt = ∂Ω/∂x
            dvy/dt = ∂Ω/∂y

        Le champ de vecteurs (∂Ω/∂x, ∂Ω/∂y) montre :
            - Point stable : vecteurs convergent vers le point
            - Point instable : vecteurs divergent

        Args:
            point: Point de Lagrange à analyser
            window_size: Taille de la fenêtre autour du point
            figsize: Dimensions de la figure
            save_path: Chemin de sauvegarde

        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)

        info = self.lagrange_points[point]
        pos = info.position
        if not self.normalized:
            pos = pos / self.distance_unit

        x0, y0 = pos[0], pos[1]

        n_points = 20
        x_range = np.linspace(x0 - window_size, x0 + window_size, n_points)
        y_range = np.linspace(y0 - window_size, y0 + window_size, n_points)
        X, Y = np.meshgrid(x_range, y_range)

        # Calcul des dérivées de Ω par différences finies
        dx = x_range[1] - x_range[0]
        dy = y_range[1] - y_range[0]

        # Potentiel effectif
        Omega = self.effective_potential(X, Y)

        # Gradient (approximation par différences centrées)
        dOmega_dx = np.zeros_like(X)
        dOmega_dy = np.zeros_like(Y)

        # Intérieur de la grille
        dOmega_dx[:, 1:-1] = (Omega[:, 2:] - Omega[:, :-2]) / (2 * dx)
        dOmega_dy[1:-1, :] = (Omega[2:, :] - Omega[:-2, :]) / (2 * dy)

        # Bords (différences avant/arrière)
        dOmega_dx[:, 0] = (Omega[:, 1] - Omega[:, 0]) / dx
        dOmega_dx[:, -1] = (Omega[:, -1] - Omega[:, -2]) / dx
        dOmega_dy[0, :] = (Omega[1, :] - Omega[0, :]) / dy
        dOmega_dy[-1, :] = (Omega[-1, :] - Omega[-2, :]) / dy

        # Champ de vecteurs
        # Pour une masse au repos, l'accélération est donnée par le gradient de Ω
        U = dOmega_dx
        V = dOmega_dy

        # Normalisation pour la visualisation
        magnitude = np.sqrt(U**2 + V**2)
        magnitude = np.where(magnitude > 0, magnitude, 1)  # Éviter division par 0

        # Fond : potentiel effectif
        contour = ax.contourf(X, Y, Omega, levels=20, cmap="RdYlBu_r", alpha=0.6)
        plt.colorbar(contour, ax=ax, label="Potentiel effectif Ω")

        quiver = ax.quiver(
            X,
            Y,
            U / magnitude,
            V / magnitude,
            magnitude,
            cmap="plasma",
            scale=20,
            width=0.003,
            alpha=0.8,
        )

        # Point de Lagrange
        ax.plot(
            x0,
            y0,
            "ro",
            markersize=15,
            markeredgecolor="darkred",
            markeredgewidth=2,
            label=f"{point.value}",
            zorder=10,
        )

        # Primaires (si dans la fenêtre)
        if abs(self.x1 - x0) < window_size * 1.2:
            ax.plot(
                self.x1,
                0,
                "yo",
                markersize=12,
                markeredgecolor="orange",
                markeredgewidth=2,
                label="Primaire 1",
                zorder=10,
            )

        if abs(self.x2 - x0) < window_size * 1.2:
            ax.plot(
                self.x2,
                0,
                "bo",
                markersize=10,
                markeredgecolor="darkblue",
                markeredgewidth=2,
                label="Primaire 2",
                zorder=10,
            )

        ax.set_xlabel("x (normalisé)", fontsize=12)
        ax.set_ylabel("y (normalisé)", fontsize=12)
        ax.set_title(
            f"Champ de stabilité autour de {point.value}\n"
            f"Stabilité : {info.stability.value}\n"
            f"{self.system_name}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(loc="best", fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Graphique sauvegardé : {save_path}")

        return fig

    def plot_comparison_systems(
        self,
        other_system: "LagrangeVisualizer",
        figsize: Tuple[float, float] = (16, 7),
        save_path: Optional[str] = None,
    ) -> plt.Figure:  # type: ignore
        """
        Compare deux systèmes côte à côte (ex: Soleil-Terre vs Terre-Lune).

        Args:
            other_system: Autre système à comparer
            figsize: Dimensions de la figure
            save_path: Chemin de sauvegarde

        Returns:
            Figure matplotlib avec 2 sous-graphiques
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Système 1
        self._plot_system_summary(ax1, self)

        # Système 2
        self._plot_system_summary(ax2, other_system)

        fig.suptitle(
            f"Comparaison : {self.system_name} vs {other_system.system_name}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Graphique sauvegardé : {save_path}")

        return fig

    def _plot_system_summary(self, ax: plt.Axes, system: "LagrangeVisualizer"):  # type: ignore
        """Graphique résumé pour un système (méthode auxiliaire)."""
        # Grille
        x_range = np.linspace(-1.5, 1.5, 300)
        y_range = np.linspace(-1.2, 1.2, 300)
        X, Y = np.meshgrid(x_range, y_range)

        # Potentiel
        Omega = system.effective_potential(X, Y)
        Omega_safe = np.clip(Omega, np.percentile(Omega, 5), np.percentile(Omega, 95))

        # Contours
        ax.contourf(X, Y, Omega_safe, levels=20, cmap="viridis", alpha=0.5)

        # Primaires
        ax.plot(
            system.x1,
            0,
            "yo",
            markersize=10,
            markeredgecolor="orange",
            markeredgewidth=2,
            zorder=10,
        )
        ax.plot(
            system.x2,
            0,
            "bo",
            markersize=8,
            markeredgecolor="darkblue",
            markeredgewidth=2,
            zorder=10,
        )

        # Points de Lagrange
        for point, info in system.lagrange_points.items():
            pos = info.position
            if not system.normalized:
                pos = pos / system.distance_unit

            color = (
                "red"
                if point in [LagrangePoint.L1, LagrangePoint.L2, LagrangePoint.L3]
                else "green"
            )
            marker = "X" if color == "red" else "o"

            ax.plot(
                pos[0],
                pos[1],
                marker,
                markersize=8,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1,
                zorder=15,
            )
            ax.annotate(
                point.value,
                xy=(pos[0], pos[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        ax.set_title(f"{system.system_name}\nμ = {system.mu:.6e}", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)


# ========== FONCTION PRINCIPALE POUR GÉNÉRER TOUS LES GRAPHIQUES ==========


def generate_all_plots(output_dir: str = "./figures_lagrange"):
    """
    Génère tous les graphiques pour comprendre les points de Lagrange.

    Crée :
        1. Vue d'ensemble Soleil-Terre
        2. Courbes de Hill
        3. Champs de stabilité pour L1, L2, L4
        4. Comparaison Soleil-Terre vs Terre-Lune

    Args:
        output_dir: Répertoire de sortie pour les figures
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GÉNÉRATION DES GRAPHIQUES : POINTS DE LAGRANGE")
    print("=" * 60)

    # Système Soleil-Terre
    print("\n1. Système Soleil-Terre...")
    sun_earth = LagrangeVisualizer(
        mu=Constants.MU_RATIO_SUN_EARTH, system_name="Soleil-Terre", normalized=True
    )

    # Vue d'ensemble
    sun_earth.plot_lagrange_points().savefig(
        os.path.join(output_dir, "lagrange_sun_earth_overview.png")
    )
    plt.show()

    # Courbes de Hill
    sun_earth.plot_hills_regions().savefig(
        os.path.join(output_dir, "lagrange_sun_earth_hills_regions.png")
    )
    plt.show()

    # Champs de stabilité
    for point in [LagrangePoint.L1, LagrangePoint.L2, LagrangePoint.L4]:
        sun_earth.plot_stability_field(point).savefig(
            os.path.join(output_dir, f"lagrange_sun_earth_stability_{point.value}.png")
        )

    plt.show()

    # Système Terre-Lune
    print("\n2. Système Terre-Lune...")
    earth_moon = LagrangeVisualizer(
        mu=Constants.MU_RATIO_EARTH_MOON, system_name="Terre-Lune", normalized=True
    )

    # Comparaison Soleil-Terre vs Terre-Lune
    sun_earth.plot_comparison_systems(earth_moon).savefig(
        os.path.join(output_dir, "lagrange_comparison_sun_earth_earth_moon.png")
    )

    plt.show()

    print("\nTous les graphiques ont été générés et sauvegardés dans :", output_dir)


if __name__ == "__main__":
    generate_all_plots(output_dir="./figures_lagrange")
