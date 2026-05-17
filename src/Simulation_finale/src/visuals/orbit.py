from dataclasses import dataclass
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Importing from your existing modules
from src.Simulations.constants import Constants, JWSTParameters
from src.Simulations.coordinates import StateVector, create_state_vector
from src.Simulations.orbit_generator import OrbitGenerator


@dataclass(frozen=True)
class TrajectoryData:
    """Immutable container for orbital trajectory results."""

    time: np.ndarray
    states: StateVector
    l2_pos_m: float


class JWSTOrbitVisualizer:
    """OOP Visualizer for JWST orbits around L2."""

    def __init__(self, generator: OrbitGenerator):
        self.gen = generator
        self.l2_pos = self.gen.lp_info.position[0] * Constants.AU

    def _calculate_x_offset(self, ay_m: float) -> float:
        """
        Functional: Calculates the linear X-offset (Ax) for a Halo orbit.
        Based on the coupling constant k for Sun-Earth L2 (~3.22).
        """
        k_coupling = 3.229
        return ay_m / k_coupling

    def get_corrected_initial_state(self):
        """Generates the state with the corrected X-offset from L2."""
        ay = 0 * JWSTParameters.ORBIT_AMPLITUDE_Y
        az = -JWSTParameters.ORBIT_AMPLITUDE_Z
        ax = self._calculate_x_offset(ay)

        nu = 2.086
        v_star = Constants.AU * Constants.OMEGA_EARTH

        state_phys = np.array(
            [
                self.l2_pos + ax,
                0.0,
                az,
                0.0,
                (ay / Constants.AU) * nu * v_star,
                0.0,
            ],
            dtype=np.float64,
        )
        return state_phys

    def run_simulation(self, duration_days: int = 30) -> TrajectoryData:
        """Propagates the orbit using your existing CRTBP model."""
        # Generate the JWST nominal orbit using the OrbitGenerator
        orbit = self.gen.generate_jwst_nominal_orbit()
        state_phys = orbit.to_physical().state
        print(f"Initial state (physical): {state_phys}")
        print(f"Orbit type: {orbit.orbit_type}")
        print(f"L2 position: {self.l2_pos}")

        # Use your internal RK4 step or solve_ivp for propagation
        # (Simplified integration logic for visualization purposes)
        t_norm = np.linspace(
            0, (duration_days * 86400) * Constants.OMEGA_EARTH, 10000, dtype=np.float64
        )

        # This uses your Dynamics Model from CRTBP_model_dynamics.py
        # For brevity, assume 'states' is the integrated result array
        states = self._integrate(state_phys, t_norm)

        return TrajectoryData(
            time=t_norm / Constants.OMEGA_EARTH / 86400,
            states=states,
            l2_pos_m=self.l2_pos,
        )

    def _integrate(
        self, start_state_phys: StateVector, t_norm: np.ndarray
    ) -> np.ndarray:
        """
        Integrates the trajectory using scipy.integrate.solve_ivp.

        Args:
            start_state_phys: Initial state in physical units [m, m/s]
            t_norm: Array of normalized time points for evaluation

        Returns:
            Array of states in physical units [m, m/s]
        """
        # 1. Normalize the state for the CRTBP engine
        l_star = Constants.AU
        v_star = l_star * Constants.OMEGA_EARTH

        state_norm = create_state_vector()
        state_norm[:3] = start_state_phys[:3] / l_star
        state_norm[3:6] = start_state_phys[3:6] / v_star

        # 2. Setup and run solve_ivp using your model's equations_of_motion
        sol = solve_ivp(
            fun=self.gen.crtbp.equations_of_motion,
            t_span=(t_norm[0], t_norm[-1]),
            y0=state_norm,
            t_eval=t_norm,
            method="RK45",
            rtol=1e-12,
            atol=1e-12,
        )

        # 3. Denormalize the results back to physical units
        states_phys = np.zeros(sol.y.T.shape)
        states_phys[:, :3] = sol.y.T[:, :3] * l_star
        states_phys[:, 3:6] = sol.y.T[:, 3:6] * v_star

        return states_phys

    def plot_projections(self, data: TrajectoryData):
        """Functional: Generates the 3 main mission projections."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("JWST Mission Geometry - Relative to L2", fontsize=14)

        # Extract relative positions in km
        x_rel = (data.states[:, 0] - data.l2_pos_m) / 1000
        y_km = data.states[:, 1] / 1000
        z_km = data.states[:, 2] / 1000

        titles = ["Top View (XY)", "Side View (XZ)", "Front View (YZ)"]
        coords = [(x_rel, y_km), (x_rel, z_km), (y_km, z_km)]
        labels = [
            ("$\Delta X$ (km)", "$Y$ (km)"),  # type: ignore
            ("$\Delta X$ (km)", "$Z$ (km)"),  # type: ignore
            ("$Y$ (km)", "$Z$ (km)"),
        ]

        for ax, title, (h, v), (xl, yl) in zip(axes, titles, coords, labels):
            ax.plot(h, v, color="royalblue", lw=1.5)
            ax.scatter(0, 0, color="red", marker="x", s=50, label="L2")
            ax.set_title(title)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.show()

    def plot_3D_trajectory(self, data: TrajectoryData):

        x_rel_km: np.ndarray = (data.states[:, 0] - data.l2_pos_m) / 1000.0
        y_km: np.ndarray = data.states[:, 1] / 1000.0
        z_km: np.ndarray = data.states[:, 2] / 1000.0

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        n_points: int = len(x_rel_km)
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, n_points - 1, dtype=np.float64))  # type: ignore

        for i in range(n_points - 1):
            ax.plot(
                x_rel_km[i : i + 2],
                y_km[i : i + 2],
                z_km[i : i + 2],
                color=colors[i],
                linewidth=1.2,
                alpha=0.8,
            )

        ax.scatter(
            0,
            0,
            0,
            color="red",
            marker="x",
            s=150,
            linewidths=3,
            label="L2 Lagrange Point",
            zorder=10,
        )

        ax.scatter(
            x_rel_km[0],
            y_km[0],
            z_km[0],
            color="green",
            marker="o",
            s=100,
            label="Initial Position",
            edgecolors="darkgreen",
            linewidths=2,
            zorder=9,
        )

        # Position finale
        ax.scatter(
            x_rel_km[-1],
            y_km[-1],
            z_km[-1],
            color="orange",
            marker="s",
            s=100,
            label="Final Position",
            edgecolors="darkorange",
            linewidths=2,
            zorder=9,
        )

        x_range: float = np.ptp(x_rel_km)  # ptp = peak to peak (max - min)
        y_range: float = np.ptp(y_km)
        z_range: float = np.ptp(z_km)

        max_range: float = max(x_range, y_range, z_range)
        x_mid: float = (np.max(x_rel_km) + np.min(x_rel_km)) / 2
        y_mid: float = (np.max(y_km) + np.min(y_km)) / 2
        z_mid: float = (np.max(z_km) + np.min(z_km)) / 2

        # Application des limites symétriques
        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

        ax.set_xlabel(r"$\Delta X$ (km)" + "\n(Earth - L2)", fontsize=11, labelpad=10)
        ax.set_ylabel(r"$Y$ (km)" + "\n(Orbital component)", fontsize=11, labelpad=10)
        ax.set_zlabel(r"$Z$ (km)" + "\n(Normal to ecliptic)", fontsize=11, labelpad=10)

        ax.set_title(
            "JWST 3D Trajectory in L2-Centered Rotating Frame\n"
            + f"Orbit type: Lissajous/Halo | Duration: {n_points} integration steps",
            fontsize=13,
            fontweight="bold",
            pad=20,
        )

        ax.grid(True, linestyle="--", alpha=0.3)

        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

        ax.view_init(elev=30, azim=45)

        """
        if np.min(z_km) < 0 < np.max(z_km):
            xx, yy = np.meshgrid(
                np.linspace(x_mid - max_range / 2, x_mid + max_range / 2, 10),
                np.linspace(y_mid - max_range / 2, y_mid + max_range / 2, 10),
            )
            ax.plot_surface(
                xx,
                yy,
                np.zeros_like(xx),
                alpha=0.1,
                color="gray",
                edgecolors="gray",
                linewidth=0.5,
            )
        """

        plt.tight_layout()
        plt.show()


# Execution entry point
if __name__ == "__main__":
    gen = OrbitGenerator()
    visualizer = JWSTOrbitVisualizer(gen)
    sim_data = visualizer.run_simulation()
    visualizer.plot_projections(sim_data)
