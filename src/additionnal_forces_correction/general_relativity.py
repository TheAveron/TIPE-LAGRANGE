import numpy as np

G = 4 * np.pi**2
C = 63239.7263
C2_INV = 1 / C**2


# Define Post-Newtonian (1PN) correction function
def general_relativity(sim):
    """Applies 1PN GR corrections to accelerations."""
    ps = sim.particles  # Get particles list
    N = len(ps)

    for i in range(N):
        p1 = ps[i]
        if p1.m == 0:
            continue  # Skip test particles

        ax, ay, az = 0, 0, 0
        v1_sq = p1.vx**2 + p1.vy**2 + p1.vz**2  # v₁² (squared velocity of p1)

        for j in range(N):
            if i == j:
                continue
            p2 = ps[j]

            # Compute relative position & distance
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dz = p2.z - p1.z
            r2 = max(dx**2 + dy**2 + dz**2, 1e-12)

            r = np.sqrt(r2)
            inv_r = 1 / r  # Store 1/r for reuse
            inv_r2 = inv_r**2  # 1/r²
            inv_r3 = inv_r2 * inv_r  # 1/r³

            # Precompute Newtonian acceleration
            Gm_r3 = G * p2.m * inv_r3  # (G*m/r³)
            ax_N = Gm_r3 * dx
            ay_N = Gm_r3 * dy
            az_N = Gm_r3 * dz

            # Compute velocity-related terms
            v2_sq = p2.vx**2 + p2.vy**2 + p2.vz**2  # v₂² (squared velocity of p2)
            dot_product1 = dx * p1.vx + dy * p1.vy + dz * p1.vz  # (r · v1)
            dot_product2 = dx * p2.vx + dy * p2.vy + dz * p2.vz  # (r · v2)

            # Symmetric relativistic correction factor
            factor = 1 + (
                (4 * G * p2.m * inv_r - v1_sq - v2_sq) * C2_INV
                + 4 * dot_product1 * dot_product2 * inv_r2 * C2_INV
            )

            # Apply relativistic correction factor
            ax += ax_N * factor
            ay += ay_N * factor
            az += az_N * factor

        # Apply final corrected acceleration to particle
        p1.ax += ax
        p1.ay += ay
        p1.az += az
