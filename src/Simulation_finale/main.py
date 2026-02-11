import sys

from src import *
from src.simulation.CRTBP_model_dynamics import CRTBP3Body
from src.simulation.dynamics_conf import DynamicsConfig, DynamicsModel
from src.simulation.orbit_generator import OrbitGenerator
from src.simulation.station_keeping import (StationKeepingStrategy,
                                            integrate_with_station_keeping)

if __name__ == "__main__":
    # constant_validation()
    # validate_transformations()
    # test_crtbp_model()
    # test_high_fidelity_model()
    # test_crtbp_vs_ephemeris_comparison()
    # test_lagrange_points()

    # generate_all_plots()

    # all_passed = run_all_tests()

    # Exemple workflow
    # example_complete_workflow()

    # sys.exit(0 if all_passed else 1)

    gen = OrbitGenerator()
    visualizer = JWSTOrbitVisualizer(gen)
    sim_data = visualizer.run_simulation(duration_days=180)
    visualizer.plot_3D_trajectory(sim_data)
    # pass

    # Générer orbite nominale
    # orbit_nominal = gen.generate_jwst_nominal_orbit()

    # Convertir en physique
    # orbit_phys = orbit_nominal.to_physical()
    # print(orbit_phys)

    """
    config = DynamicsConfig(model=DynamicsModel.CRTBP)
    crtbp = CRTBP3Body(config, normalized=True)

    # Simuler 1 an avec station-keeping
    states, maneuvers = integrate_with_station_keeping(
        crtbp_model=crtbp,
        initial_state=orbit_phys.state,
        reference_orbit=orbit_nominal,
        duration=86400,  # 1 jour
        strategy=StationKeepingStrategy.JWST_OPERATIONAL,
    )

    # Afficher résultats
    print(f"Nombre de manœuvres : {len(maneuvers)}")
    print(f"ΔV total : {sum(m.magnitude for m in maneuvers):.3f} m/s")
    """
