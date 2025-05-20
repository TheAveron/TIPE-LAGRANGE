import matplotlib.pyplot as plt
import numpy as np
from profiler import profile
from numba import jit

G = 6.67e-11
Ms = 1.989e30  # Mass of the Sun (kg)
Mt = 5.972e24
Mjwst = 6500
r = 1.5e6
R = 1.5e8
omega = G * (Ms + Mt) / R**3

coefv_sun = - G * Ms * Mjwst
coefv_ter = - G * Mt * Mjwst
coefv_cf = - 0.5 * Mjwst * omega

x = np.linspace(-r * 2, 10e5, 10000)
y = np.linspace(-r, r, 10000)

X, Y = np.meshgrid(x, y)

EP = np.empty_like(X)

@profile
def completing_ep():
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_ij = X[i, j]
            y_ij = Y[i, j]

            d_sun = (x_ij + R)**2 + y_ij ** 2
            d_earth = x_ij ** 2 + y_ij ** 2
            V_sun = coefv_sun * np.sqrt(d_sun)
            V_earth = coefv_ter * np.sqrt(d_earth)

            V_cf = coefv_cf * (x_ij ** 2 + y_ij ** 2)
            EP[i, j] = V_sun + V_earth + V_cf

completing_ep()

fig = plt.figure()
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])

cp = ax.contourf(X, Y, EP)
# ax.clabel(cp, inline=True, fontsize=9)
plt.plot(0, 0, marker="x", color="red", markersize=5, label="Point de Lagrange")

ax.set_title("Contour Plot")
ax.set_xlabel("Axe Terre-Soleil (10e6 m)")
ax.set_ylabel("y (cm)")

#plt.show()