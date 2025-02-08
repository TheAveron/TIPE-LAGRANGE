import rebound
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fsolve
import reboundx

from src import general_relativity, newtonian

# Constants for the Sun-Earth system
G = 4 * np.pi**2  # AU^3 / (yr^2 * Msun), gravitational constant
C = reboundx.c  # Speed of light in AU/yr
M_SUN = 1.0  # Mass of the Sun in solar masses
M_EARTH = 3.003e-6  # Earth mass in solar masses
A_EARTH = 1.0  # Earth's semi-major axis (1 AU)
MU = M_EARTH / (M_SUN + M_EARTH)  # Reduced mass

# Variations of constants for better running time
C2_INV = 1 / C**2  # Store 1/c² for later use
MUminus = 1 - MU
sqrt3 = np.sqrt(3) / 2


def mapper(object: tuple):
    return tuple(map(float, object))


"""
# Approximate Lagrange point locations
L1 = mapper((A_EARTH * (1 - (MU / 3) ** (1 / 3)), 0, 0))
L2 = mapper((A_EARTH * (1 + (MU / 3) ** (1 / 3)), 0, 0))
L3 = mapper((-A_EARTH * (1 + 5 * MU / 12), 0, 0))
L4 = mapper((A_EARTH * np.cos(np.pi / 3), A_EARTH * np.sin(np.pi / 3), 0))
L5 = mapper((A_EARTH * np.cos(np.pi / 3), -A_EARTH * np.sin(np.pi / 3), 0))

lagrange_points = [L1, L2, L3, L4, L5]
"""

def lagrange_eq(x, L_type):
    term1 = MU / (x - MU) ** 2
    term2 = MUminus / (x + MUminus) ** 2
    if L_type == "L1":
        return x - MUminus - term1 + term2
    elif L_type == "L2":
        return x - MUminus - term1 - term2
    elif L_type == "L3":
        return x + MU - term1 - term2

d_L1g = fsolve(lagrange_eq, 0.99, args=("L1"))[0]
d_L2g = fsolve(lagrange_eq, 1.01, args=("L2"))[0]
d_L3g = fsolve(lagrange_eq, -1.01, args=("L3"))[0]

particules_number = 7


def particule_creation(sim: rebound.Simulation):
    # Add the Sun
    sim.add(m=M_SUN, x=0, y=0, z=0, vx=0, vy=0, vz=0)
    # Add the Earth

    
    vitesse = np.sqrt(G * M_SUN / A_EARTH)
    sim.add(m=M_EARTH, x=A_EARTH, y=0, z=0, vx=0, vy=vitesse, vz=0)

    # Add test particles at the Lagrange points
    """for L in lagrange_points:
        sim.add(x=L[0], y=L[1], z=L[2], vx=0, vy=np.sqrt(G * M_SUN / A_EARTH), vz=0)
    """

    dx, dy, dz = A_EARTH, 0, 0

    r = np.sqrt(dx**2 + dy**2 + dz**2)
    
    bary_x = MU * A_EARTH
    bary_y = 0
    bary_z = 0

    
    angle_L4 = np.pi / 3  # Add 60 degrees (pi/3 radians)

    vitesse_cos= vitesse * np.cos(angle_L4)
    vitesse_sin = vitesse * np.sin(angle_L4)
    L4: tuple[float, float, float] = (
        bary_x + r * np.cos(angle_L4),
        bary_y + r * np.sin(angle_L4),
        bary_z
    )

    #sim.add(m = 1e-30, x=L4[0], y = L4[1], z = L4[2], vx = vitesse*np.sin(np.pi /3), vy = vitesse*np.cos(np.pi /3), vz = 0)


def compute_lagrange_points(
    sun_pos, earth_pos
) -> list[tuple[float, float, float]]:
    """
    Computes the positions of the five Lagrange points dynamically based on the Sun and Earth's positions.

    Parameters:
    sun   -> Rebound particle representing the Sun. 
    earth -> Rebound particle representing the Earth.

    Returns:
    Dictionary containing positions of L1, L2, L3, L4, and L5.
    """
    MU = M_EARTH / (M_SUN + M_EARTH)
    MUminus = 1 - MU

    # Extract positions and masses
    x1, y1, z1 = sun_pos
    x2, y2, z2 = earth_pos

    # some global computation to aoid redundency
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1

    # Compute the Earth-Sun distance
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    angle = np.arctan2(dy, dx)  # Angle of the Earth-Sun vector in radians

    # Center of Mass (Barycenter)
    bary_x = MUminus * x1 + MU * x2
    bary_y = MUminus * y1 + MU * y2
    bary_z = MUminus * z1 + MU * z2

    #
    cos = np.cos(angle)
    sin = np.sin(angle)

    # Transform to real positions in space
    L1: tuple[float, float, float] = (bary_x + (r + 0.01) * cos, bary_y + (r + 0.01) * sin, bary_z)
    L2: tuple[float, float, float] = (bary_x + (r - 0.01) * cos, bary_y + (r - 0.01) * sin, bary_z)

    angle_L3 = angle + np.pi
    L3: tuple[float, float, float] = (
        bary_x + r * np.cos(angle_L3), 
        bary_y + r * np.sin(angle_L3), 
        bary_z
        )

    # Rotation matrix for L4 (60 degrees counterclockwise)
    angle_L4 = angle + np.pi / 3  # Add 60 degrees (pi/3 radians)
    L4: tuple[float, float, float] = (
        bary_x + r * np.cos(angle_L4),
        bary_y + r * np.sin(angle_L4),
        bary_z
    )

    # Rotation matrix for L5 (-60 degrees clockwise)
    angle_L5 = angle - np.pi / 3  # Subtract 60 degrees (-pi/3 radians)
    L5: tuple[float, float, float] = (
        bary_x + r * np.cos(angle_L5),
        bary_y + r * np.sin(angle_L5),
        bary_z,
    )

    return [L1, L2, L3, L4, L5]


def graph(positions: dict):
    # Plot the results
    plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection="3d")
    # print(positions)

    colors = ["red", "green", "orange", "purple", "cyan"]
    for ind, color in enumerate(colors[2:]):
        pos = np.array(positions[ind + 1])
        plt.plot(pos[:, 0], pos[:, 1], color=color, label=f"L{ind+1}")

    # Earth
    pos_earth = np.array(positions[1])
    plt.plot(pos_earth[1:, 0], pos_earth[1:, 1], color="blue", label=f"Earth")

    # Sun
    plt.scatter(0, 0, color="yellow", s=200, label="Sun")

    """
    # L4
    #print(positions[5])
    pos = np.array(positions[5])
    plt.plot(pos[:, 0], pos[:, 1], color=colors[3], label=f"L4")
    """

    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.title("Earth-Sun System with Lagrange Points (GR Corrected)")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()


def rebound_plot(sim):
    rebound.OrbitPlot(sim, particles=[1], color=True).fig.savefig(
        f"plots/Lagrange/Earthview.png"
    )
    rebound.OrbitPlot(sim, color=True).fig.savefig(f"plots/Lagrange/systemview.png")
    for i in range(2, particules_number):
        rebound.OrbitPlot(sim, particles=[1, i], primary=0, color=True).fig.savefig(
            f"plots/Lagrange/object_{i}.png"
    )


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
    positions: dict[int, list[tuple[float, float, float]]] = {
        i: [] for i in range(particules_number)
    }
    for t in tqdm(times):
        sim.integrate(t)

        sun = sim.particles[0]
        earth = sim.particles[1]

        sun_coord = (sun.x, sun.y, sun.z)
        earth_coord = (earth.x, earth.y, earth.z)

        positions[0].append(sun_coord)
        positions[1].append(earth_coord)

        L_points = compute_lagrange_points(sun_coord, earth_coord)
        for i, p in enumerate(L_points):
            positions[i + 2].append(p)

    return positions


if __name__ == "__main__":
    # Set up the siMUlation
    sim = rebound.Simulation()
    sim.move_to_com()
    sim.units = ("AU", "yr", "Msun")

    rebx = reboundx.Extras(sim)
    gr = rebx.load_force("gr")
    rebx.add_force(gr)

    particule_creation(sim)
    rebound.OrbitPlot(sim, color=True).fig.savefig(f"plots/Lagrange/startingsystemview.png")

    positions = main(sim)

    rebound.OrbitPlot(sim, color=True).fig.savefig(f"plots/Lagrange/systemview.png")

    graph(positions)
    #rebound_plot(sim)
