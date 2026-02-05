from src import *
from src.simulation.orbit_generator import OrbitGenerator
import sys


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
    pass
