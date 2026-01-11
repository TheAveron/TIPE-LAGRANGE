from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from constants import Constants
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from simulator import Simulator
from state import State


class Visualizer:
    """Classe pour la visualisation 3D des trajectoires"""

    def __init__(self, simulator: Simulator):
        self.simulator = simulator

    def plot_trajectory_3d(
        self,
        history: List[Tuple[float, State]],
        title: str = "Trajectoire du satellite",
        show_lagrange: bool = True,
        show_orbits: bool = True,
        reference_frame: str = "inertial",
    ):
        """
        Affiche la trajectoire 3D du satellite

        Args:
            history: Historique de la simulation
            title: Titre du graphique
            show_lagrange: Afficher les points de Lagrange
            show_orbits: Afficher l'orbite terrestre
            reference_frame: "inertial" ou "rotating"
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Extraction des positions
        times = np.array([t for t, _ in history])
        positions = np.array(
            [[s.position.x, s.position.y, s.position.z] for _, s in history]
        )

        # Conversion en km pour lisibilité
        positions_km = positions / 1e3

        # Si référentiel tournant, rotation inverse
        if reference_frame == "rotating":
            positions_km = self._to_rotating_frame(positions_km, times)

        # Trajectoire du satellite
        ax.plot(
            positions_km[:, 0],
            positions_km[:, 1],
            positions_km[:, 2],
            "b-",
            linewidth=2,
            label="Satellite",
            alpha=0.7,
        )

        # Position initiale et finale
        ax.scatter(
            positions_km[0, 0],
            positions_km[0, 1],
            positions_km[0, 2],
            c="green",
            s=100,  # type: ignore
            marker="o",
            label="Début",
            zorder=5,
        )
        ax.scatter(
            positions_km[-1, 0],
            positions_km[-1, 1],
            positions_km[-1, 2],
            c="red",
            s=100,  # type: ignore
            marker="X",
            label="Fin",
            zorder=5,
        )

        # Soleil (à l'origine)
        ax.scatter(
            0,
            0,
            0,
            c="yellow",
            s=300,  # type: ignore
            marker="o",
            edgecolors="orange",
            linewidths=2,
            label="Soleil",
            zorder=10,
        )

        # Orbite de la Terre
        if show_orbits:
            if reference_frame == "inertial":
                theta = np.linspace(0, 2 * np.pi, 100)
                r_earth_km = Constants.R_EARTH_ORBIT / 1e3
                orbit_x = r_earth_km * np.cos(theta)
                orbit_y = r_earth_km * np.sin(theta)
                orbit_z = np.zeros_like(theta)
                ax.plot(
                    orbit_x,
                    orbit_y,
                    orbit_z,
                    "g--",
                    linewidth=1,
                    alpha=0.3,
                    label="Orbite Terre",
                )

        # Position de la Terre au début et à la fin
        earth_start = self.simulator.earth.get_state_at_time(times[0])
        earth_end = self.simulator.earth.get_state_at_time(times[-1])

        earth_pos_start = (
            np.array(
                [earth_start.position.x, earth_start.position.y, earth_start.position.z]
            )
            / 1e3
        )
        earth_pos_end = (
            np.array([earth_end.position.x, earth_end.position.y, earth_end.position.z])
            / 1e3
        )

        if reference_frame == "rotating":
            earth_pos_start = self._to_rotating_frame(
                earth_pos_start.reshape(1, 3), np.array([times[0]])
            )[0]
            earth_pos_end = self._to_rotating_frame(
                earth_pos_end.reshape(1, 3), np.array([times[-1]])
            )[0]

        ax.scatter(
            earth_pos_start[0],
            earth_pos_start[1],
            earth_pos_start[2],
            c="blue",
            s=200,  # type: ignore
            marker="o",
            label="Terre (début)",
            zorder=8,
        )
        ax.scatter(
            earth_pos_end[0],
            earth_pos_end[1],
            earth_pos_end[2],
            c="cyan",
            s=200,  # type: ignore
            marker="o",
            label="Terre (fin)",
            alpha=0.6,
            zorder=8,
        )

        # Points de Lagrange
        if show_lagrange and reference_frame == "rotating":
            for name, point in self.simulator.lagrange_points.items():
                pos_km = np.array([point.x, point.y, point.z]) / 1e3
                color = "red" if name == "L2" else "orange"
                size = 150 if name == "L2" else 80
                ax.scatter(
                    pos_km[0],
                    pos_km[1],
                    pos_km[2],
                    c=color,
                    s=size,  # type: ignore
                    marker="*",
                    label=name,
                    zorder=9,
                    edgecolors="black",
                    linewidths=1,
                )

        # Configuration des axes
        ax.set_xlabel("X [km]", fontsize=12)
        ax.set_ylabel("Y [km]", fontsize=12)
        ax.set_zlabel("Z [km]", fontsize=12)  # type: ignore
        ax.set_title(
            f"{title}\nRéférentiel: {reference_frame}", fontsize=14, fontweight="bold"
        )

        # Légende
        ax.legend(loc="upper left", fontsize=10)

        # Grille
        ax.grid(True, alpha=0.3)

        # Aspect ratio égal
        self._set_axes_equal(ax)

        plt.tight_layout()
        return fig, ax

    def plot_dual_view(
        self,
        history: List[Tuple[float, State]],
        title: str = "Comparaison référentiels",
    ):
        """
        Affiche deux vues côte à côte : inertiel et tournant
        """
        fig = plt.figure(figsize=(18, 8))

        # Vue inertielle
        ax1 = fig.add_subplot(121, projection="3d")
        self._plot_single_view(
            ax1, history, "inertial", "Référentiel héliocentrique inertiel"
        )

        # Vue tournante
        ax2 = fig.add_subplot(122, projection="3d")
        self._plot_single_view(
            ax2, history, "rotating", "Référentiel tournant Soleil-Terre"
        )

        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        return fig, (ax1, ax2)

    def _plot_single_view(self, ax, history, reference_frame, subtitle):
        """Helper pour tracer une vue unique"""
        times = np.array([t for t, _ in history])
        positions = np.array(
            [[s.position.x, s.position.y, s.position.z] for _, s in history]
        )
        positions_km = positions / 1e3

        if reference_frame == "rotating":
            positions_km = self._to_rotating_frame(positions_km, times)

        # Trajectoire
        ax.plot(
            positions_km[:, 0],
            positions_km[:, 1],
            positions_km[:, 2],
            "b-",
            linewidth=2,
            alpha=0.7,
        )

        # Début/Fin
        ax.scatter(
            positions_km[0, 0],
            positions_km[0, 1],
            positions_km[0, 2],
            c="green",
            s=100,
            marker="o",
            zorder=5,
        )
        ax.scatter(
            positions_km[-1, 0],
            positions_km[-1, 1],
            positions_km[-1, 2],
            c="red",
            s=100,
            marker="X",
            zorder=5,
        )

        # Soleil
        ax.scatter(
            0,
            0,
            0,
            c="yellow",
            s=300,
            marker="o",
            edgecolors="orange",
            linewidths=2,
            zorder=10,
        )

        # Terre
        earth_pos = self.simulator.earth.get_state_at_time(times[0])
        earth_km = (
            np.array([earth_pos.position.x, earth_pos.position.y, earth_pos.position.z])
            / 1e3
        )
        if reference_frame == "rotating":
            earth_km = self._to_rotating_frame(
                earth_km.reshape(1, 3), np.array([times[0]])
            )[0]
        ax.scatter(
            earth_km[0], earth_km[1], earth_km[2], c="blue", s=200, marker="o", zorder=8
        )

        # Points de Lagrange (seulement en tournant)
        if reference_frame == "rotating":
            for name, point in self.simulator.lagrange_points.items():
                pos_km = np.array([point.x, point.y, point.z]) / 1e3
                color = "red" if name == "L2" else "orange"
                size = 150 if name == "L2" else 80
                ax.scatter(
                    pos_km[0],
                    pos_km[1],
                    pos_km[2],
                    c=color,
                    s=size,
                    marker="*",
                    zorder=9,
                    edgecolors="black",
                    linewidths=1,
                )

        ax.set_xlabel("X [km]", fontsize=10)
        ax.set_ylabel("Y [km]", fontsize=10)
        ax.set_zlabel("Z [km]", fontsize=10)
        ax.set_title(subtitle, fontsize=12)
        ax.grid(True, alpha=0.3)
        self._set_axes_equal(ax)

    def _to_rotating_frame(
        self, positions: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """
        Convertit des positions du référentiel inertiel au tournant

        Args:
            positions: Array (N, 3) de positions en référentiel inertiel
            times: Array (N,) de temps correspondants

        Returns:
            Positions dans le référentiel tournant
        """
        rotated = np.zeros_like(positions)

        for i, t in enumerate(times):
            angle = -Constants.OMEGA * t  # Rotation inverse
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Matrice de rotation autour de z
            R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

            rotated[i] = R @ positions[i]

        return rotated

    def _set_axes_equal(self, ax):
        """Fixe les limites des axes pour avoir un aspect ratio égal"""
        limits = np.array(
            [
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ]
        )

        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    def plot_energy_conservation(self, history: List[Tuple[float, State]]):
        """
        Affiche l'évolution de l'énergie pour vérifier la conservation
        """
        times = np.array([t for t, _ in history]) / (24 * 3600)  # En jours
        energies = [self.simulator.get_energy(state, t) for t, state in history]

        # Normalisation par rapport à l'énergie initiale
        E0 = energies[0]
        relative_error = [(E - E0) / abs(E0) * 100 for E in energies]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Énergie absolue
        ax1.plot(times, energies, "b-", linewidth=2)
        ax1.set_xlabel("Temps [jours]", fontsize=12)
        ax1.set_ylabel("Énergie spécifique [J/kg]", fontsize=12)
        ax1.set_title(
            "Évolution de l'énergie mécanique", fontsize=14, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)

        # Erreur relative
        ax2.plot(times, relative_error, "r-", linewidth=2)
        ax2.set_xlabel("Temps [jours]", fontsize=12)
        ax2.set_ylabel("Erreur relative [%]", fontsize=12)
        ax2.set_title("Conservation de l'énergie", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Statistiques
        max_error = max(abs(min(relative_error)), abs(max(relative_error)))
        ax2.text(
            0.02,
            0.98,
            f"Erreur max: {max_error:.2e}%",
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return fig, (ax1, ax2)
