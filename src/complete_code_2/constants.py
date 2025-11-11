import numpy as np


# Gravitational constant (km^3 / kg / s^2)
G = 6.67430e-20  # = 6.67430e-11 m^3/kg/s^2 converted to km^3/kg/s^2

# Masses (kg)
M_sun = 1.98847e30
M_earth = 5.97219e24

# Astronomical unit (km)
AU = 149597870.7

# Mean Sun-Earth distance (km)
a = AU

# Compute barycenter positions along x-axis (two-body)
# If origin is at barycenter, then
# x_sun = -a * (M_earth / (M_sun + M_earth))
# x_earth = +a * (M_sun / (M_sun + M_earth))
mu_total = M_sun + M_earth
x_sun = -a * (M_earth / mu_total)
x_earth = a * (M_sun / mu_total)
x_L2 = x_earth + 1.5e6


# Angular rate of the rotating frame (rad/s) assuming circular orbit
omega = np.sqrt(G * mu_total / a**3)

# Omega vector
omega_vec = np.array([0.0, 0.0, omega])

# Useful small numbers
EPS = 1e-15

# Default integration parameters
DEFAULT_DT = 60.0  # 60 s
