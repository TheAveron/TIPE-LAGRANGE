import numpy as np

G = 4 * np.pi**2


def newtonian(sim):
    """Applies Newtonian gravity corrections to accelerations without General Relativity effects."""
    ps = sim.particles  # Get particles list

    # Gather positions and velocities in arrays
    positions = np.array([[p.x, p.y, p.z] for p in ps])
    velocities = np.array([[p.vx, p.vy, p.vz] for p in ps])
    masses = np.array([p.m for p in ps])

    # Compute pairwise position differences
    diff_positions = (
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    )  # shape: (N, N, 3)
    r2 = np.sum(diff_positions**2, axis=2)  # r² (distance squared between all pairs)

    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    r2 = np.maximum(r2, epsilon)  # Ensure r² is never zero
    r = np.sqrt(r2)  # r (distance between all pairs)

    inv_r3 = 1 / r**3  # 1/r³

    # Initialize accelerations array
    accelerations = np.zeros_like(positions)

    # Precompute Newtonian acceleration factors: G * m / r³
    Gm_r3 = G * masses[:, np.newaxis] * inv_r3  # shape: (N, N)

    # Apply Newtonian gravity accelerations
    accelerations += np.sum(Gm_r3[:, :, np.newaxis] * diff_positions, axis=1)

    # Update particle accelerations
    for i, p in enumerate(ps):
        p.ax += accelerations[i, 0]
        p.ay += accelerations[i, 1]
        p.az += accelerations[i, 2]
