from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Constantes physiques
G = 6.67430e-11  # Constante gravitationnelle
M_s = 1.989e30  # Masse du Soleil (kg)
M_t = 5.972e24  # Masse de la Terre (kg)
d = 1.496e11  # Distance Terre-Soleil (m)
omega = sqrt(G * (M_s + M_t) / d**3)  # Vitesse angulaire orbitale

# Dans ce référentiel, la Terre est à (0, 0), le Soleil à (-d, 0)
x_t, y_t = 0.0, 0.0
x_s, y_s = -d, 0.0

# Grille centrée autour de la Terre
lim = 3e10  # 3 millions de km plus ou moins
x = np.linspace(-lim, lim, 1000)
y = np.linspace(-lim, lim, 1000)
X, Y = np.meshgrid(x, y)

# Distances au Soleil et et la Terre
r_s = np.sqrt((X - x_s) ** 2 + (Y - y_s) ** 2)
r_t = np.sqrt((X - x_t) ** 2 + (Y - y_t) ** 2)
r = np.sqrt(X**2 + Y**2)

# Potentiel effectif dans le référentiel co-rotatif centré sur Terre
U = EP = -G * M_s / r_s - G * M_t / r_t - 0.5 * omega**2 * r**2
print(U)
# Masquer les points proches de la Terre (par exemple, r_t < 1e7 km)
mask = r_t < 3e7  # Masque les points proches de la Terre (<= 10 millions de km)

# Appliquer le masque sur U
U[mask] = np.nan  # Remplacer par NaN pour �viter l'affichage

dU = np.gradient(U)
dU_x, dU_y = dU
dU_x = -dU_x
dU_y = -dU_y

"""
U_shifted = U - np.min(U) + 1e5  # rendre toutes les valeurs positives, et éviter log(0)
logU = np.log10(U_shifted)
"""
"""
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, EP, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3
)

ax.contour(X, Y, EP, zdir="z", offset=-100, cmap="coolwarm")
ax.contour(X, Y, EP, zdir="x", offset=-2 * lim, cmap="coolwarm")
ax.contour(X, Y, EP, zdir="y", offset=2 * lim, cmap="coolwarm")

# Calcul de la position approximative du point L2
# ax.scatter(0, 0, np.max(EP), color='red', label='Point de Lagrange')

# Étiquettes
ax.set_title("Surface d'énergie potentielle")
ax.set_xlabel("Axe Terre-Soleil (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("Énergie potentielle (J)")

# fig.colorbar(surf, shrink=0.5, aspect=5, label="Énergie potentielle")

# plt.tight_layout()
plt.show()"""

# Affichage du potentiel
plt.figure(figsize=(10, 8))
contour = plt.contourf(
    X / 1e6, Y / 1e6, dU_y, levels=100, cmap="coolwarm"
)  # en milliers de km

plt.colorbar(label="Potentiel effectif (J/kg)")
plt.plot([0], [0], "bo", label="Terre")
plt.plot([-d / 1e6], [0], "yo", label="Soleil")
plt.legend()
plt.xlabel("x (en milliers de km, centré sur Terre)")
plt.ylabel("y (en milliers de km)")
plt.title("Potentiel effectif centré sur la Terre")
plt.grid(True)
plt.axis("equal")
plt.show()
