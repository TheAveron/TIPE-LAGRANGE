from turtle import color
import rebound
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import general_relativity, newtonian

# Constants for the Sun-Earth system
G = 4 * np.pi**2  # AU^3 / (yr^2 * Msun), gravitational constant
C = 63239.7263  # Speed of light in AU/yr
M_SUN = 1.0  # Mass of the Sun in solar masses
M_EARTH = 3.003e-6  # Earth mass in solar masses
A_EARTH = 1.0  # Earth's semi-major axis (1 AU)
MU = M_EARTH / (M_SUN + M_EARTH)  # Reduced mass

# Variations of constants for better running time
C2_INV = 1 / C**2  # Store 1/c² for later use

def mapper(object: tuple):
    return tuple(map(float, object))

# Approximate Lagrange point locations
L1 = mapper((A_EARTH * (1 - (MU / 3) ** (1 / 3)), 0, 0))
L2 = mapper((A_EARTH * (1 + (MU / 3) ** (1 / 3)), 0, 0))
L3 = mapper((-A_EARTH * (1 + 5 * MU / 12), 0, 0))
L4 = mapper((A_EARTH * np.cos(np.pi / 3), A_EARTH * np.sin(np.pi / 3), 0))
L5 = mapper((A_EARTH * np.cos(np.pi / 3), -A_EARTH * np.sin(np.pi / 3), 0))

lagrange_points = [L1, L2, L3, L4, L5]

particules_number = 7

def particule_creation(sim: rebound.Simulation):
    # Add the Sun
    sim.add(m=M_SUN, x=0, y=0, z=0, vx=0, vy=0, vz=0)
    # Add the Earth
    sim.add(m=M_EARTH, a=A_EARTH, primary=sim.particles[0])

    # Add test particles at the Lagrange points
    for L in lagrange_points:
        sim.add(x=L[0], y=L[1], z=L[2], vx=0, vy=np.sqrt(G * M_SUN / A_EARTH), vz=0)

def graph(positions: dict):
    # Plot the results
    plt.figure(figsize=(10, 10))
    for i in range(2, particules_number):  # Ignore Sun and Earth
        pos = np.array(positions[i])
        plt.plot(pos[:, 0], pos[:, 1], label=f"L{i-1}")

    # Mark Sun and Earth
    plt.scatter(0, 0, color="yellow", s=200, label="Sun")
    plt.scatter(A_EARTH, 0, color="blue", s=100, label="Earth")

    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.title("Earth-Sun System with Lagrange Points (GR Corrected)")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()

def rebound_plot(sim):
    rebound.OrbitPlot(sim, particles=[1], color=True).fig.savefig(f"plots/Lagrange/Earthview.png")
    rebound.OrbitPlot(sim, color=True).fig.savefig(f"plots/Lagrange/systemview.png")
    for i in range(2, particules_number):
        rebound.OrbitPlot(sim, particles=[1 , i], primary=0, color=True).fig.savefig(f"plots/Lagrange/L{i-1}.png")


def main(sim: rebound.Simulation):
    def relativity_correction(reb_sim):
        general_relativity(sim)

    def newtonian_correction(reb_sim):
        newtonian(sim)

    # Attach the force function to Rebound
    sim.additional_forces = relativity_correction
    sim.integrator = "ias15"  # High-accuracy integrator

    # Integrate and track positions
    times = np.linspace(0, 1, 100000)  # 1 years, 500 steps
    positions = {i: [] for i in range(particules_number)}
    for t in tqdm(times):
        sim.integrate(t)
        for i, p in enumerate(sim.particles):
            positions[i].append((p.x, p.y))

    return positions

if __name__ == "__main__":
    # Set up the siMUlation
    sim = rebound.Simulation()
    sim.move_to_com()
    sim.units = ("AU", "yr", "Msun")

    particule_creation(sim)
    rebound.OrbitPlot(sim, color=True).fig.savefig(f"plots/Lagrange/startingsystemview.png")

    positions = main(sim)
    

    graph(positions)
    rebound_plot(sim)
