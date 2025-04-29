import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from math import sqrt


G = 6.67e-11
Ms = 1.989e30  # Mass of the Sun (kg)
Mt = 5.972e24
Mjwst = 6500
r = 1.5e6
R = 1.5e8

x = np.outer(np.linspace(0, 1, 100), np.ones(100))
y = x.copy().T

EP_t = G * Mt * Mjwst / np.sqrt((r + x) ** 2 + y**2)

# EP_t = lambda x, y: G * Mt * Mjwst / np.sqrt((r + x) ** 2 + y ** 2)
EP_s = lambda x, y: G * Ms * Mjwst / sqrt((R + r + x) ** 2 + y**2)


ax = plt.figure().add_subplot(projection="3d")
X, Y, Z = x, y, EP_t

# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
ax.contour(X, Y, Z, zdir="z", offset=-100, cmap="coolwarm")
ax.contour(X, Y, Z, zdir="x", offset=-40, cmap="coolwarm")
ax.contour(X, Y, Z, zdir="y", offset=40, cmap="coolwarm")

ax.set(
    xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100), xlabel="X", ylabel="Y", zlabel="Z"
)

plt.show()
