import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

G = 6.67e-11
Ms = 1.989e30  # Mass of the Sun (kg)
Mt = 5.972e24
Mjwst = 6500
r = 1.5e6
R = 1.5e8

x = np.linspace(-150_000_000, 0, 1000)
y = np.linspace(-1_500_000, 1500000, 1000)

X, Y = np.meshgrid(x, y)
EP = (
    G * Mt * Mjwst / np.sqrt((r + X) ** 2 + Y**2)
    + G * Ms * Mjwst / np.sqrt((R + r + X) ** 2 + Y**2)
    - 1 / 2 * Mjwst * G * Ms / (R**3) * ((R + r + X) ** 2 - y**2)
)

fig = plt.figure()
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])

cp = ax.contourf(X, Y, EP)
#ax.clabel(cp, inline=True, fontsize=9)
ax.set_title('Contour Plot')
ax.set_xlabel('Axe Terre-Soleil (m)')
ax.set_ylabel('y (cm)')
plt.show()
