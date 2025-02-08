import rebound
import dpnp as np
import matplotlib.pyplot as plt
import reboundx.constants
from tqdm import tqdm
from scipy.optimize import fsolve
import reboundx
from numba import njit

from src import general_relativity, newtonian

# Constants for the Sun-Earth system
G = 4 * np.pi**2  # AU^3 / (yr^2 * Msun), gravitational constant
C = reboundx.constants.C # Speed of light in AU/yr
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

    sim.add(m = 1e-4, a=1, Omega=np.pi/3)

@njit
def compute_lagrange_points(sun_pos, earth_pos) -> list[tuple[float, float, float]]:
    """
    Computes the positions of the five Lagrange points dynamically based on the Sun and Earth's positions.

    Parameters:
    sun_pos   -> Tuple representing the Sun's position (x, y, z).
    earth_pos -> Tuple representing the Earth's position (x, y, z).

    Returns:
    List containing positions of L1, L2, L3, L4, and L5.
    """
    MU = M_EARTH / (M_SUN + M_EARTH)
    MUminus = 1 - MU

    # Convert to NumPy arrays for vectorized operations
    sun_pos = np.array(sun_pos)
    earth_pos = np.array(earth_pos)

    # Vectorized difference for Earth-Sun
    delta = earth_pos - sun_pos
    r = np.linalg.norm(delta)  # Earth-Sun distance
    angle = np.arctan2(delta[1], delta[0])  # Angle of the Earth-Sun vector in radians

    # Barycenter computation
    bary_pos = MUminus * sun_pos + MU * earth_pos

    # Precompute cos and sin values
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # L1 and L2 computations (reusing cos_angle and sin_angle)
    offset = 0.01
    L1 = bary_pos + (r + offset) * np.array([cos_angle, sin_angle, 0])
    L2 = bary_pos + (r - offset) * np.array([cos_angle, sin_angle, 0])

    # L3 computation (180 degrees opposite)
    L3 = bary_pos + r * np.array([-cos_angle, -sin_angle, 0])

    # L4 (60 degrees ahead) and L5 (60 degrees behind)
    angle_offset = np.pi / 3  # 60 degrees in radians
    cos_L4, sin_L4 = np.cos(angle + angle_offset), np.sin(angle + angle_offset)
    cos_L5, sin_L5 = np.cos(angle - angle_offset), np.sin(angle - angle_offset)

    L4 = bary_pos + r * np.array([cos_L4, sin_L4, 0])
    L5 = bary_pos + r * np.array([cos_L5, sin_L5, 0])

    # Convert back to tuples if needed
    return [tuple(L1), tuple(L2), tuple(L3), tuple(L4), tuple(L5)]



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
    plt.savefig("test.png")


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
    #sim.additional_forces = relativity_correction
    sim.integrator = "whfast"#"ias15"  # High-accuracy integrator


    # Integrate and track positions
    times = np.linspace(0, 50000, 1000)  # 1 years, 500 steps
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
    sim.move_to_hel()
    #sim.move_to_com()
    sim.units = ("AU", "yr", "Msun")

    rebx = reboundx.Extras(sim)
    gr = rebx.load_force("gr")
    gr.params['c'] = C
    rebx.add_force(gr)

    particule_creation(sim)
    rebound.OrbitPlot(sim, color=True).fig.savefig(f"plots/Lagrange/startingsystemview.png")

    positions = main(sim)

    rebound.OrbitPlot(sim, color=True).fig.savefig(f"plots/Lagrange/systemview.png")

    #graph(positions)
    #rebound_plot(sim)
