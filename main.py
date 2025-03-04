"""Use to simulate trajectory around the sun"""

import numpy as np
import rebound
import reboundx
import reboundx.constants
from numba import njit
from tqdm import tqdm

# Constants for the Sun-Earth system
G = 4 * np.pi**2  # AU^3 / (yr^2 * Msun), gravitational constant
C = reboundx.constants.C  # Speed of light in AU/yr
M_SUN = 1.0  # Mass of the Sun in solar masses
M_EARTH = 3.003e-6  # Earth mass in solar masses
A_EARTH = 1.0  # Earth's semi-major axis (1 AU)
MU = M_EARTH / (M_SUN + M_EARTH)  # Reduced mass


M_MARS = 0.107 * M_EARTH
A_MARS = 1.52368055

# Variations of constants for better running time
C2_INV = 1 / C**2  # Store 1/c² for later use
MUMINUS = 1 - MU
sqrt3 = np.sqrt(3) / 2


def mapper(item: tuple):
    """map a tuple of float"""
    return tuple(map(float, item))


PARTICLE_NUMBER = 7


def particule_creation(simu: rebound.Simulation):
    """Add the particles to the simluation"""
    # Add the Sun
    simu.add(m=M_SUN, x=0, y=0, z=0, vx=0, vy=0, vz=0)
    # Add the Earth
    vitesse_earth = np.sqrt(G * M_SUN / A_EARTH)
    vitesse_mars = np.sqrt(G * M_SUN / A_MARS)

    simu.add(m=M_EARTH, x=A_EARTH, y=0, z=0, vx=0, vy=vitesse_earth, vz=0)
    simu.add(m=M_MARS, x=A_MARS, y=0, z=0, vy=vitesse_mars, vz=0)

    # sim.add(m=1e-3, a=1, Omega=np.pi / 3)
    # sim.add(m=1e-30, a=1, Omega=(- np.pi / 3 + np.pi / 10))


@njit
def compute_lagrange_points(sun_pos, earth_pos):
    """
    Computes the positions of the five Lagrange points dynamically,
    based on the Sun and Earth's positions.
    """

    # Convert to NumPy arrays for vectorized operations
    sun_pos = np.array(sun_pos, dtype=np.float64)
    earth_pos = np.array(earth_pos, dtype=np.float64)

    # Vectorized difference for Earth-Sun
    delta = earth_pos - sun_pos
    r = np.linalg.norm(delta)  # Earth-Sun distance
    angle = np.arctan2(delta[1], delta[0])  # Angle of the Earth-Sun vector in radians

    # Barycenter computation
    bary_pos = MUMINUS * sun_pos + MU * earth_pos

    # Precompute cos and sin values
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # L1 and L2 computations (reusing cos_angle and sin_angle)
    offset = 0.01
    l1 = bary_pos + (r + offset) * np.array([cos_angle, sin_angle, 0])
    l2 = bary_pos + (r - offset) * np.array([cos_angle, sin_angle, 0])

    # L3 computation (180 degrees opposite)
    l3 = bary_pos + r * np.array([-cos_angle, -sin_angle, 0])

    # L4 (60 degrees ahead) and L5 (60 degrees behind)
    angle_offset = np.pi / 3  # 60 degrees in radians
    cos_L4, sin_L4 = np.cos(angle + angle_offset), np.sin(angle + angle_offset)
    cos_L5, sin_L5 = np.cos(angle - angle_offset), np.sin(angle - angle_offset)

    l4 = bary_pos + r * np.array([cos_L4, sin_L4, 0])
    l5 = bary_pos + r * np.array([cos_L5, sin_L5, 0])

    return [l1, l2, l3, l4, l5]


config1 = """
2 mass at L4 and L5

m=1e-3
setps = 10000
total_time = 100"""


def main(sim: rebound.Simulation):

    steps = 10000
    total_time = 1000

    div = total_time / steps

    times: list[float] = []
    k = 0
    for _ in range(steps):
        k += div
        times.append(round(k, 2))

    # positions = {i: [] for i in range(PARTICLE_NUMBER)}

    # Precompute values that don't change
    # sun = sim.particles[0]
    # earth = sim.particles[1]

    plot = rebound.OrbitPlot(sim, color=True, figsize=(10, 10))

    for t in tqdm(times):
        sim.integrate(t, exact_finish_time=total_time)

        # sun_coord = (sun.x, sun.y, sun.z)
        # earth_coord = (earth.x, earth.y, earth.z)

        # positions[0].append(sun_coord)
        # positions[1].append(earth_coord)

        plot.update()
        plot.fig.savefig(f"plots/Lagrange/timelapse-mars/frame{int(t*100):06d}.png")

        # Lagrange points computation
        # L_points = compute_lagrange_points(sun_coord, earth_coord)
        # for i, p in enumerate(L_points):
        # positions[i + 2].append(tuple(p))

    # return positions


if __name__ == "__main__":

    sim = rebound.Simulation()
    # sim.threads = 8
    sim.move_to_hel()
    sim.units = ("AU", "yr", "Msun")
    sim.integrator = "whfast"

    rebx = reboundx.Extras(sim)
    gr = rebx.load_force("gr")
    gr.params["c"] = C
    rebx.add_force(gr)

    particule_creation(sim)
    rebound.OrbitPlot(sim, color=True, figsize=(10, 10), Narc=256).fig.savefig(
        "plots/Lagrange/startingsystemview.png"
    )

    main(sim)

    rebound.OrbitPlot(sim, color=True, figsize=(10, 10), Narc=256).fig.savefig(
        "plots/Lagrange/systemview.png"
    )
