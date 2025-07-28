from math import sqrt

### Constants ###
G = 6.67430e-11  # gravitational constant
C = 2.99e8

# Masses (kg)
M_sun = 1.9885e30
M_earth = 5.972e24
M_moon = 7.347e22

total_mass = M_sun + M_earth + M_moon

# Distances (m)
d_se = 1.496e11  # Sun-Earth
d_em = 3.844e8  # Earth-Moon

# Compute barycenter position along the x-axis (1D simplification for setup)
x_earth = d_se * M_sun / total_mass
x_sun = -d_se * (M_earth + M_moon) / total_mass
x_moon = x_earth - d_em

# Angular velocity of the rotating frame (based on Earth orbit)
omega = sqrt(G * (M_earth + M_sun) / d_se**3)
