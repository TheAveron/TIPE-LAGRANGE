from typing import List, Tuple

import matplotlib.pyplot as plt
from constants import Constants
from simulator import Simulator
from state import State
from vector import Vector3D
from visualizer import Visualizer

if __name__ == "__main__":
    """Script principal pour tester le simulateur et visualiseur
    # Création du simulateur
    sim = Simulator()

    # Test 1 : Vérification de l'orbite terrestre
    print("=== Test 1 : Orbite de la Terre ===")
    print(f"Période théorique : {Constants.T_YEAR / (24*3600):.2f} jours")
    print(f"Rayon orbital : {Constants.R_EARTH_ORBIT / 1e9:.3f} millions de km")
    print(f"Vitesse orbitale : {Constants.V_EARTH / 1e3:.2f} km/s")

    # Test 2 : Points de Lagrange
    print("\n=== Test 2 : Points de Lagrange ===")
    L2_distance = sim.lagrange_points["L2"].x - Constants.R_EARTH_ORBIT
    print(f"Distance Terre-L2 : {L2_distance / 1e9:.3f} millions de km")
    print(f"Position L2 : {sim.lagrange_points['L2']}")

    # Test 3 : Simulation d'un satellite proche de L2
    print("\n=== Test 3 : Simulation satellite près de L2 ===")

    # État initial : légèrement décalé de L2
    initial_pos = sim.lagrange_points["L2"] + Vector3D(1e8, 0, 0)  # +100,000 km
    initial_vel = Vector3D(0, Constants.V_EARTH, 0)  # Vitesse de la Terre
    initial_state = State(initial_pos, initial_vel)

    # Simulation sur 30 jours
    t_start = 0
    t_end = 30 * 24 * 3600  # 30 jours en secondes
    dt = 3600  # 1 heure

    print(f"Durée de simulation : {t_end / (24*3600):.0f} jours")
    print(f"Pas de temps : {dt / 3600:.1f} heures")
    print("Propagation en cours...")

    history = sim.propagate(initial_state, t_start, t_end, dt)

    print(f"Nombre de points calculés : {len(history)}")

    # Vérification conservation de l'énergie
    energies = [sim.get_energy(state, t) for t, state in history]
    energy_variation = (max(energies) - min(energies)) / abs(energies[0])
    print(f"Variation relative d'énergie : {energy_variation:.2e}")

    print("\nSimulation terminée avec succès !")
    print("Structure de base opérationnelle.")
    """


if __name__ == "__main__":
    # Création du simulateur
    sim = Simulator()
    vis = Visualizer(sim)

    # Test : Simulation satellite près de L2
    print("=== Simulation satellite près de L2 ===")

    # État initial : orbite de halo autour de L2
    # Petite perturbation en position et vitesse
    initial_pos = sim.lagrange_points["L2"] + Vector3D(1e8, 5e7, 3e7)  # km
    initial_vel = Vector3D(-50, Constants.V_EARTH + 10, 5)  # m/s
    initial_state = State(initial_pos, initial_vel)

    # Simulation sur 180 jours
    t_start = 0
    t_end = 2 * 180 * 24 * 3600  # 180 jours
    dt = 3600  # 1 heure

    print(f"Durée: {t_end / (24*3600):.0f} jours, Pas: {dt / 3600:.1f} h")
    print("Propagation en cours...")

    history = sim.propagate(initial_state, t_start, t_end, dt)
    print(f"✓ {len(history)} points calculés")

    # Visualisations
    print("\nGénération des graphiques...")

    # Vue 3D dans le référentiel tournant
    fig1, ax1 = vis.plot_trajectory_3d(
        history,
        title="Trajectoire autour de L2 (référentiel tournant)",
        reference_frame="rotating",
    )

    # Vue 3D dans le référentiel inertiel
    fig2, ax2 = vis.plot_trajectory_3d(
        history,
        title="Trajectoire autour de L2 (référentiel inertiel)",
        reference_frame="inertial",
        show_lagrange=False,
    )

    # Comparaison double vue
    fig3, (ax3, ax4) = vis.plot_dual_view(history, title="Comparaison des référentiels")

    # Conservation de l'énergie
    fig4, (ax5, ax6) = vis.plot_energy_conservation(history)

    plt.show()

    print("\n✓ Visualisations générées avec succès !")
