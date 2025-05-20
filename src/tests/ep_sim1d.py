import matplotlib.pyplot as plt
import numpy as np
from profiler import profile

G = 6.67e-11
Ms = 1.989e30  # Mass of the Sun (kg)
Mt = 5.972e24
Mjwst = 6500
r = 1.5e6
R = 1.5e8

K1 = - G * Mt * Mjwst
K2 = - G * Ms * Mjwst
K3 = - 1 / 2 * Mjwst * G * Ms

x = np.linspace(-R, R, 1000)
y = np.linspace(-R, R, 1000)

X, Y = np.meshgrid(x, y)

#@profile
def calcul_EP(X, Y):
    """EP = (
        K1 / np.sqrt((r + X) ** 2 + Y**2)
        + K2 / np.sqrt((R + r + X) ** 2 + Y**2)
        + K3 / (R**3) * ((R + r + X) ** 2 - Y**2)
    )"""

    EP = K1 * np.sqrt((X + R)**2 + Y ** 2) + K2 * np.sqrt(X ** 2 + Y ** 2) + K3 * (X ** 2 + Y ** 2)

    return EP/100

EP = calcul_EP(X, Y)

fig = plt.figure()
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])

cp = ax.contourf(X, Y, EP)
# ax.clabel(cp, inline=True, fontsize=9)
plt.plot(0, 0, marker="x", color="red", markersize=5, label="Point de Lagrange")

ax.set_title("Contour Plot")
ax.set_xlabel("Axe Terre-Soleil (10e6 m)")
ax.set_ylabel("y (m)")
plt.show()
