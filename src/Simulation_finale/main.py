import sys

import numpy as np

from src import *
from src.Models.base_dynamics import DynamicsConfig, DynamicsModel
from src.Simulations.calcul_pos_lagrange import LagrangePoint
from src.Simulations.constants import Constants
from src.Simulations.CRTBP3_dynamics import CRTBP3Body
from src.Simulations.differential_corrector import DifferentialCorrector
from src.Simulations.orbit_generator import (
    OrbitGenerator,
    OrbitInitialConditions,
    OrbitType,
)
from src.Simulations.station_keeping import (
    StationKeepingStrategy,
    integrate_with_station_keeping,
)
from src.Simulations.target_point import TargetPointController
from src.visuals.orbit import TrajectoryData

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
    duration_days = 180

    gen = OrbitGenerator()
    orbit_linear = gen.generate_jwst_nominal_orbit()
    state_linear_norm = orbit_linear.to_normalized().state

    config = DynamicsConfig(model=DynamicsModel.CRTBP)
    crtbp = CRTBP3Body(config, normalized=True)

    corrector = DifferentialCorrector(crtbp, tol=1e-10)

    assert orbit_linear.period
    T_half_approx = orbit_linear.period * Constants.OMEGA_EARTH / 2  # normalisé

    state_corrected_norm, period_norm = corrector.correct(
        state_linear_norm, T_half_approx
    )

    # Reconstruire l'objet orbite corrigée
    orbit_corrected = OrbitInitialConditions(
        state=state_corrected_norm,
        orbit_type=OrbitType.QUASI_HALO,
        period=period_norm / Constants.OMEGA_EARTH,
        amplitudes=orbit_linear.amplitudes,
        jacobi_constant=crtbp.jacobi_constant(state_corrected_norm),
        lagrange_point=LagrangePoint.L2,
        generation_method="differential_corrector",
        is_physical=False,
    )

    # 3. Simuler avec Target Point Method
    crtbp_phys = CRTBP3Body(config, normalized=False)
    tpm = TargetPointController(crtbp_phys, orbit_corrected.to_physical())

    states, maneuvers = integrate_with_station_keeping(
        crtbp_model=crtbp_phys,
        initial_state=orbit_corrected.to_physical().state,
        reference_orbit=orbit_corrected,
        duration=86400 * 180,
        strategy=StationKeepingStrategy.TARGET_POINT,
        controller_override=tpm,  # à ajouter dans integrate_with_station_keeping
    )
