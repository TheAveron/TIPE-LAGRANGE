import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Constantes
G = 6.67e-11
Ms = 1.989e30  # Masse du Soleil (kg)
Mt = 5.972e24  # Masse de la Terre (kg)
Mjwst = 6500  # Masse du télescope JWST (kg)
r = 1.5e6  # Distance Terre-JWST (m)
R = 1.5e8  # Distance Terre-Soleil (m)

# Grille de calcul
x = np.linspace(-R, R, 500)
y = np.linspace(-R, R, 500)
X, Y = np.meshgrid(x, y)

# Énergie potentielle
K1 = G * Mt * Mjwst
K2 = G * Ms * Mjwst
K3 = 1 / 2 * Mjwst * G * Ms
"""EP = (
    K1 / np.sqrt((r + X) ** 2 + Y**2)
    + K2 / np.sqrt((R + r + X) ** 2 + Y**2)
    + K3 / (R**3) * ((R + r + X) ** 2 - Y**2)
)"""
EP = ( K1 * np.sqrt((X + R)**2 + Y ** 2) + K2 * np.sqrt(X ** 2 + Y ** 2) + K3 * (X ** 2 + Y ** 2))


# Tracé 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, EP, cmap="viridis", edgecolor="none")

# Calcul de la position approximative du point L2
# ax.scatter(0, 0, np.max(EP), color='red', label='Point de Lagrange')

# Étiquettes
ax.set_title("Surface d'énergie potentielle")
ax.set_xlabel("Axe Terre-Soleil (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("Énergie potentielle (J)")

fig.colorbar(surf, shrink=0.5, aspect=5, label="Énergie potentielle")

plt.legend()
plt.tight_layout()

plt.show()
