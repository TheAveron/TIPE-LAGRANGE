"""
main.py — Point d'entrée principal de la simulation JWST.

Lance les deux simulations (CR3BP et inertielle) et affiche les graphes.
"""

import numpy as np
from cr3bp import CR3BPSimulation
from cr3bp.stationkeeping import StationKeepingSimulation
from inertial import InertialSimulation
from visualization import (
    plot_cr3bp_trajectory,
    plot_cr3bp_velocity,
    plot_delta_v_history,
    plot_energy,
    plot_energy_comparison,
    plot_inertial_trajectory,
    plot_inertial_velocity,
    plot_jacobi,
    plot_sk_jacobi,
    plot_sk_trajectory,
    plot_velocity_comparison,
)

# Paramètres communs

AZ_ADIM = np.float64(0.00240)  # 0.00279  # amplitude hors-plan JWST ≈ 418 000 km
N_REVOLUTIONS = np.int16(3)  # nombre de révolutions halo à simuler
N_STEPS = np.int16(10000)  # pas RK4 par révolution (augmenter pour plus de précision)

# 1. Simulation CR3BP

print("=" * 60)
print("  MODULE 1 — CR3BP (repère tournant non-dimensionnalisé)")
print("=" * 60)

sim_cr3bp = CR3BPSimulation(
    Az=AZ_ADIM,
    n_revolutions=N_REVOLUTIONS,
    n_steps_per_rev=N_STEPS,
    northern=True,
    phi=np.float64(0),
)
sim_cr3bp.run()

# 2. Station keeping simulation


sim_stationkeeping = StationKeepingSimulation(
    Az=AZ_ADIM,
    n_revolutions=N_REVOLUTIONS,
    n_steps_per_rev=N_STEPS,
)

sim_stationkeeping.run()

# 3. Simulation inertielle J2000

print("\n" + "=" * 60)
print("  MODULE 2 — Inertiel (référentiel J2000)")
print("=" * 60)

assert sim_cr3bp.state0 is not None and sim_cr3bp.T_halo
assert sim_stationkeeping.state0_ref is not None and sim_stationkeeping.T_halo

sim_inertial = InertialSimulation(
    state0_cr3bp=sim_stationkeeping.state0_ref,
    n_revolutions=N_REVOLUTIONS,
    T_halo_adim=sim_stationkeeping.T_halo,
    n_steps_per_rev=N_STEPS,
)
sim_inertial.run()
# 3. Visualisations

print("\nAffichage des graphes...")


# Station keeping
plot_sk_trajectory(sim_stationkeeping)
plot_sk_jacobi(sim_stationkeeping)
plot_delta_v_history(sim_stationkeeping)

# Trajectoires
plot_cr3bp_trajectory(sim_cr3bp)
plot_inertial_trajectory(sim_inertial)

# Énergie / Jacobi
plot_jacobi(sim_cr3bp)
plot_energy(sim_inertial)
plot_energy_comparison(sim_cr3bp, sim_inertial)

# Vitesses
plot_cr3bp_velocity(sim_cr3bp)
plot_inertial_velocity(sim_inertial)
plot_velocity_comparison(sim_cr3bp, sim_inertial)
