import time
import rebound
import numpy as np
import matplotlib.pyplot as plt



# Constants for the Sun-Earth system
G = 4 * np.pi**2  # AU^3 / (yr^2 * Msun), gravitational constant
c = 63239.7263  # Speed of light in AU/yr
M_sun = 1.0  # Mass of the Sun in solar masses
M_earth = 3.003e-6  # Earth mass in solar masses
a_earth = 1.0  # Earth's semi-major axis (1 AU)
mu = M_earth / (M_sun + M_earth)  # Reduced mass

# Approximate Lagrange point locations
L1 = (a_earth * (1 - (mu / 3) ** (1 / 3)), 0, 0)
L2 = (a_earth * (1 + (mu / 3) ** (1 / 3)), 0, 0)
L3 = (-a_earth * (1 + 5 * mu / 12), 0, 0)
L4 = (a_earth * np.cos(np.pi / 3), a_earth * np.sin(np.pi / 3), 0)
L5 = (a_earth * np.cos(np.pi / 3), -a_earth * np.sin(np.pi / 3), 0)

lagrange_points = [L1, L2, L3, L4, L5]

import numpy as np

def general_relativity_correction(sim):
    """Applies 1PN GR corrections to accelerations with NumPy vectorization."""
    ps = sim.particles  # Get particles list
    N = len(ps)
    c2_inv = 1 / c**2  # Store 1/c² for later use
    
    # Gather positions and velocities in arrays
    positions = np.array([[p.x, p.y, p.z] for p in ps])
    velocities = np.array([[p.vx, p.vy, p.vz] for p in ps])
    masses = np.array([p.m for p in ps])

    # Calculate the relative positions and distances (pairwise)
    diff_positions = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape: (N, N, 3)
    r2 = np.sum(diff_positions**2, axis=2)  # r² (distance squared between all pairs)
    r = np.sqrt(r2)  # r (distance between all pairs)
    inv_r = 1 / r  # 1/r
    inv_r2 = inv_r**2  # 1/r²
    inv_r3 = inv_r2 * inv_r  # 1/r³

    # Compute squared velocities for all particles
    v1_sq = np.sum(velocities**2, axis=1)  # shape: (N,)

    # Compute the accelerations using vectorized operations
    # Initialize accelerations array
    accelerations = np.zeros_like(positions)

    # Precompute Newtonian acceleration factors: G * m / r³
    Gm_r3 = G * masses[:, np.newaxis] * inv_r3  # shape: (N, N)
    dot_product = np.sum(diff_positions * velocities[:, np.newaxis, :], axis=2)  # shape: (N, N)
    
    # Calculate the 1PN correction factor for each pair
    factor = ((4 * G * masses[:, np.newaxis] * inv_r - v1_sq[:, np.newaxis]) * c2_inv + 
              4 * (dot_product**2) * (inv_r2 * c2_inv))  # shape: (N, N)

    # Apply the correction to accelerations
    accelerations += np.sum(Gm_r3 * diff_positions * (1 + factor)[:, :, np.newaxis], axis=1)

    # Update particle accelerations
    for i, p in enumerate(ps):
        p.ax += accelerations[i, 0]
        p.ay += accelerations[i, 1]
        p.az += accelerations[i, 2]


# Define a non-vectorized GR correction (original, for comparison)
def general_relativity_correction_non_vectorized(sim):
    ps = sim.particles
    N = len(ps)
    c2_inv = 1 / c**2
    for i in range(N):
        p1 = ps[i]
        if p1.m == 0:
            continue  # Skip test particles
        ax, ay, az = 0, 0, 0
        v1_sq = p1.vx**2 + p1.vy**2 + p1.vz**2
        for j in range(N):
            if i == j:
                continue
            p2 = ps[j]
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dz = p2.z - p1.z
            r2 = dx**2 + dy**2 + dz**2
            r = np.sqrt(r2)
            inv_r = 1 / r
            inv_r2 = inv_r**2
            inv_r3 = inv_r2 * inv_r
            ax_N = G * p2.m * inv_r3 * dx
            ay_N = G * p2.m * inv_r3 * dy
            az_N = G * p2.m * inv_r3 * dz
            dot_product = dx * p1.vx + dy * p1.vy + dz * p1.vz
            factor = ((4 * G * p2.m * inv_r - v1_sq) * c2_inv + 4 * (dot_product**2) * (inv_r2 * c2_inv))
            ax += ax_N * (1 + factor)
            ay += ay_N * (1 + factor)
            az += az_N * (1 + factor)
        p1.ax += ax
        p1.ay += ay
        p1.az += az

# Set up a simple simulation to measure performance
def run_simulation(sim, correction_func):
    sim.additional_forces = correction_func
    sim.integrator = "ias15"
    sim.move_to_com()
    
    times = np.linspace(0, 1, 500)  # Simulate for 100 years, 500 time steps
    for t in times:
        sim.integrate(t)

# Measure the time for the non-vectorized version
def measure_time():
    # Test with a small system first
    N = 50  # Number of particles
    sim = rebound.Simulation()
    sim.add(m=1.0)  # Sun
    sim.add(m=3e-6, a=1.0)  # Earth
    sim.add(m=0)  # Add test particles to Lagrange points

    for _ in range(N-2):  # Adding more test particles
        sim.add(m=0)

    def non_vec_correction(re_sim):
        general_relativity_correction_non_vectorized(sim)
    def vec_correction(re_sim):
        general_relativity_correction(sim)


    # Time for non-vectorized
    start_time = time.perf_counter()
    run_simulation(sim, non_vec_correction)
    non_vec_time = time.perf_counter() - start_time

    # Time for vectorized
    start_time = time.perf_counter()
    run_simulation(sim, vec_correction)
    vec_time = time.perf_counter() - start_time

    return non_vec_time, vec_time

# Test speed with different numbers of particles
N_values = [50, 100, 200, 400, 800]
non_vec_times = []
vec_times = []

for N in N_values:
    non_vec_time, vec_time = measure_time()
    non_vec_times.append(non_vec_time)
    vec_times.append(vec_time)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(N_values, non_vec_times, label="Non-Vectorized", marker='o', linestyle='-', color='r')
plt.plot(N_values, vec_times, label="Vectorized", marker='o', linestyle='-', color='b')
plt.xlabel("Number of Particles")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time Comparison: Non-Vectorized vs Vectorized")
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.show()
