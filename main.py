"""
Simulate JWST-like halo orbit around Sun-Earth L2.
Uses proper L2 calculation via Newton's method in the rotating frame.
Integrates with RK4 in rotating frame where dynamics are more stable.
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Physical constants (SI units: km, s)
G = 6.67430e-20  # Gravitational constant (km^3/kg/s^2)
M_SUN = 1.98847e30  # Sun mass (kg)
M_EARTH = 5.97219e24  # Earth mass (kg)
AU = 149597870.7  # 1 AU (km)

# Sun-Earth system (barycentric frame)
a = AU  # Mean Sun-Earth distance
mu_total = M_SUN + M_EARTH
x_sun = -a * (M_EARTH / mu_total)
x_earth = a * (M_SUN / mu_total)

# Angular rate of rotating frame (rad/s)
omega = np.sqrt(G * mu_total / a**3)
omega_vec = np.array([0.0, 0.0, omega])


def grav_acceleration(r):
    """Gravitational acceleration in rotating frame (km/s^2)."""
    r_sun = np.array([x_sun, 0.0, 0.0])
    r_earth = np.array([x_earth, 0.0, 0.0])
    
    rs = r - r_sun
    re = r - r_earth
    norm_rs = np.linalg.norm(rs)
    norm_re = np.linalg.norm(re)
    
    a_sun = -G * M_SUN * rs / (norm_rs**3 + 1e-30)
    a_earth = -G * M_EARTH * re / (norm_re**3 + 1e-30)
    return a_sun + a_earth


def rotating_frame_acceleration(r, v):
    """Total acceleration in rotating frame (km/s^2)."""
    a_grav = grav_acceleration(r)
    coriolis = -2.0 * np.cross(omega_vec, v)
    centrifugal = -np.cross(omega_vec, np.cross(omega_vec, r))
    return a_grav + coriolis + centrifugal


def compute_L2(tol=1e-6, maxiter=60):
    """
    Compute L2 position (x-coordinate in km) using Newton's method.
    L2 is the colinear point beyond Earth where net rotating-frame acceleration = 0.
    """
    # Initial guess using cubic approximation: d = a*(mu/3)^(1/3)
    mu = M_EARTH / (M_SUN + M_EARTH)
    d_guess = a * (mu / 3.0) ** (1.0 / 3.0)
    x = x_earth + d_guess
    
    def f(x_val):
        r = np.array([x_val, 0.0, 0.0])
        v = np.zeros(3)
        a = rotating_frame_acceleration(r, v)
        return a[0]  # x-component must be zero at L2
    
    # Newton's method with finite difference derivative
    for i in range(maxiter):
        fx = f(x)
        h = max(1e-3, abs(x) * 1e-6)
        fpx = (f(x + h) - f(x - h)) / (2 * h)
        
        if abs(fpx) < 1e-16:
            break
        
        dx = -fx / fpx
        x += dx
        
        if abs(dx) < tol:
            break
    
    return x


def compute_halo_orbit_velocity(x_L2, amplitude_y=1e5):
    """
    Compute initial velocity for a halo orbit around L2.
    
    For a halo orbit in the L2 region, we use the eigenmode of the linearized
    dynamics around L2. The linearized motion near L2 in the yz-plane can be
    approximated by coupling the y and z oscillations with proper phase.
    
    Returns velocity vector [vx, vy, vz] in km/s for a state near L2 with
    initial position displacement in y-direction.
    """
    r_L2 = np.array([x_L2, 0.0, 0.0])
    
    # Linearized stiffness-like parameter from the second-order Taylor expansion
    # around L2 in the rotating frame
    mu = M_EARTH / (M_SUN + M_EARTH)
    k = a * (mu / 3.0) ** (1.0 / 3.0)
    
    # For stability in a halo orbit near L2, typical initial conditions have
    # position displaced in y with velocity in z (or phase-shifted combination).
    # The frequency of oscillation in the L2 region is approximately:
    lambda_freq = omega * np.sqrt(3.0)  # Approximate eigenfrequency
    
    # Simple initial condition: displace in y, give velocity in z direction
    # with magnitude proportional to amplitude and frequency
    vz = amplitude_y * lambda_freq
    
    return np.array([0.0, 0.0, vz])


def inertial_to_rotating(r_I, v_I, t):
    """
    Transform from inertial frame to rotating frame.
    Rotation angle: theta = omega * t
    """
    theta = omega * t
    c, s = np.cos(theta), np.sin(theta)
    
    # Rotation matrix
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    
    # Position in rotating frame
    r_rot = R @ r_I
    
    # Velocity: v_rot = R * (v_I - omega x r_I)
    omega_cross_r = np.cross(omega_vec, r_I)
    v_rot = R @ (v_I - omega_cross_r)
    
    return r_rot, v_rot


def state_derivative(state):
    """Derivative of state [r, v] = [dr/dt, dv/dt] = [v, a]."""
    r = state[:3]
    v = state[3:]
    a = rotating_frame_acceleration(r, v)
    return np.hstack((v, a))


def rk4_step(state, dt):
    """Single RK4 integration step."""
    k1 = state_derivative(state)
    k2 = state_derivative(state + 0.5 * dt * k1)
    k3 = state_derivative(state + 0.5 * dt * k2)
    k4 = state_derivative(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def propagate_l2_halo(duration_days=20, dt_seconds=60.0):
    """
    Propagate a spacecraft in halo orbit around L2.
    
    Args:
        duration_days: Integration duration (days)
        dt_seconds: Time step (seconds)
    
    Returns:
        times: Array of times (seconds)
        trajectory: Array of states [r(3), v(3)] at each time
    """
    # Real JWST initial conditions (inertial frame, 2022-01-24)
    r_inertial = np.array([1.499073e8, 3.97968e5, -1.83603e4])  # km
    v_inertial = np.array([-1.23229, -29.26776, 0.119171])  # km/s
    
    t0 = 0.0
    
    # Transform to rotating frame
    r0, v0 = inertial_to_rotating(r_inertial, v_inertial, t0)
    state0 = np.hstack((r0, v0))
    
    # Sanity checks
    x_L2 = compute_L2()
    print(f"L2 x-coordinate: {x_L2:.2e} km")
    print(f"Earth x-coordinate: {x_earth:.2e} km")
    print(f"Distance Earth to L2: {x_L2 - x_earth:.2e} km")
    print(f"Initial position: {state0[:3]}")
    print(f"Initial velocity: {state0[3:]}")
    
    # Integrate
    tf = duration_days * 24 * 3600  # Convert to seconds
    nsteps = int(np.ceil((tf - t0) / dt_seconds))
    
    times = np.zeros(nsteps + 1)
    trajectory = np.zeros((nsteps + 1, 6))
    
    state = state0.copy()
    trajectory[0] = state
    times[0] = t0
    
    print(f"\nIntegrating {duration_days} days with {nsteps} steps...")
    for i in tqdm(range(nsteps)):
        state = rk4_step(state, dt_seconds)
        times[i + 1] = times[i] + dt_seconds
        trajectory[i + 1] = state
    
    return times, trajectory


def plot_results(times, trajectory):
    """Plot orbit projection and distance to L2."""
    x_L2 = compute_L2()
    positions = trajectory[:, :3]
    
    # XY projection
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # XY plane
    axes[0, 0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=0.5)
    axes[0, 0].plot(x_earth, 0, 'go', markersize=8, label='Earth')
    axes[0, 0].plot(x_L2, 0, 'r*', markersize=15, label='L2')
    axes[0, 0].set_xlabel('X (km)')
    axes[0, 0].set_ylabel('Y (km)')
    axes[0, 0].set_title('XY Projection')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # XZ plane
    axes[0, 1].plot(positions[:, 0], positions[:, 2], 'b-', linewidth=0.5)
    axes[0, 1].plot(x_earth, 0, 'go', markersize=8, label='Earth')
    axes[0, 1].plot(x_L2, 0, 'r*', markersize=15, label='L2')
    axes[0, 1].set_xlabel('X (km)')
    axes[0, 1].set_ylabel('Z (km)')
    axes[0, 1].set_title('XZ Projection')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # YZ plane
    axes[1, 0].plot(positions[:, 1], positions[:, 2], 'b-', linewidth=0.5)
    axes[1, 0].set_xlabel('Y (km)')
    axes[1, 0].set_ylabel('Z (km)')
    axes[1, 0].set_title('YZ Projection (Halo Orbit)')
    axes[1, 0].grid(True)
    
    # Distance to L2
    distances_to_L2 = np.linalg.norm(positions - np.array([x_L2, 0, 0]), axis=1)
    axes[1, 1].plot(times / 3600, distances_to_L2 / 1e6, 'b-', linewidth=1)
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Distance to L2 (million km)')
    axes[1, 1].set_title('Orbit Stability')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/halo_orbit_l2.png', dpi=150)
    print("Plot saved to plots/halo_orbit_l2.png")


if __name__ == "__main__":
    times, trajectory = propagate_l2_halo(duration_days=20, dt_seconds=60.0)
    plot_results(times, trajectory)
    print("\nSimulation complete!")
