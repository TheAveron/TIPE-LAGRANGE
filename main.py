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

# Set up the simulation
sim = rebound.Simulation()
sim.units = ("AU", "yr", "Msun")

# Add the Sun
sim.add(m=M_sun, x=0, y=0, z=0, vx=0, vy=0, vz=0)

# Add the Earth
sim.add(m=M_earth, x=a_earth, y=0, z=0, vx=0, vy=np.sqrt(G * M_sun / a_earth), vz=0)

# Add test particles at the Lagrange points
for L in lagrange_points:
    sim.add(m=0, x=L[0], y=L[1], z=L[2], vx=0, vy=np.sqrt(G * M_sun / a_earth), vz=0)


# Define Post-Newtonian (1PN) correction function
def general_relativity_correction(reb_sim):
    """Applies 1PN GR corrections to accelerations."""
    ps = sim.particles  # Get particles list
    N = len(ps)
    c2_inv = 1 / c**2  # Store 1/c² to avoid division in loop

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
            r2 = dx**2 + dy**2 + dz**2  # r²
            r = np.sqrt(r2)  # r
            inv_r = 1 / r  # Store 1/r for reuse
            inv_r2 = inv_r**2  # 1/r²
            inv_r3 = inv_r2 * inv_r  # 1/r³

            # Precompute Newtonian acceleration
            Gm_r3 = G * p2.m * inv_r3  # (G*m/r³)
            ax_N = Gm_r3 * dx
            ay_N = Gm_r3 * dy
            az_N = Gm_r3 * dz

            # Compute velocity-related terms
            dot_product = dx * p1.vx + dy * p1.vy + dz * p1.vz  # (r · v)
            factor = ((4 * G * p2.m * inv_r - v1_sq) * c2_inv + 4 * (dot_product**2) * (inv_r2 * c2_inv))

            # Apply relativistic correction factor
            ax += ax_N * (1 + factor)
            ay += ay_N * (1 + factor)
            az += az_N * (1 + factor)

        # Apply final corrected acceleration to particle
        p1.ax += ax
        p1.ay += ay
        p1.az += az


# Attach the force function to Rebound
sim.additional_forces = general_relativity_correction
sim.integrator = "ias15"  # High-accuracy integrator

# Integrate and track positions
times = np.linspace(0, 1, 500)  # 10 years, 500 steps
positions = {i: [] for i in range(len(sim.particles))}
for t in times:
    sim.integrate(t)
    for i, p in enumerate(sim.particles):
        positions[i].append((p.x, p.y))

# Plot the results
plt.figure(figsize=(8, 8))
for i in range(2, len(sim.particles)):  # Ignore Sun and Earth
    pos = np.array(positions[i])
    plt.plot(pos[:, 0], pos[:, 1], label=f"L{i-1}")

# Mark Sun and Earth
plt.scatter(0, 0, color="yellow", s=200, label="Sun")
plt.scatter(a_earth, 0, color="blue", s=100, label="Earth")

plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.title("Earth-Sun System with Lagrange Points (GR Corrected)")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()
