import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from math import sqrt

plt.ion()

G = 6.67e-11
Ms = 1.989e30  # Mass of the Sun (kg)
Mt = 5.972e24
Mjwst = 6500
r = 1.5e6
R = 1.5e8

EP_t = lambda x, y, z: G * Mt * Mjwst / sqrt((r + x) ** 2 + y ** 2 + z ** 2)
EP_s = lambda x, y, z: G * Ms * Mjwst / sqrt( (R + r + x) ** 2 + y ** 2 + z ** 2)



ax = plt.figure().add_subplot(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)

# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
       xlabel='X', ylabel='Y', zlabel='Z')

while True:
       plt.show()