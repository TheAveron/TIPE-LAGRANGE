"""
barycentre.py — Visualisation du Soleil, de la Terre et de leur barycentre
dans le repère tournant adimensionnel du CR3BP Soleil-Terre.

Conventions CR3BP :
  - Unité de longueur : distance Soleil-Terre = 1 UA ≈ 1.496e8 km
  - μ = m_Terre / (m_Soleil + m_Terre) ≈ 3.04e-6
  - Soleil    en x = -μ       (légèrement à gauche de l'origine)
  - Terre     en x = 1 - μ    (légèrement à gauche de x=1)
  - Barycentre en x = 0       (définition même du repère CR3BP)
"""

import matplotlib.pyplot as plt

MU = 3.040423389123456e-6  # valeur JPL

# Positions adim. dans le repère tournant
x_sun = -MU
x_earth = 1.0 - MU
x_bary = 0.0

# Distances en km pour les annotations
UA_KM = 1.496e8
d_sun_bary = abs(x_sun) * UA_KM
d_earth_bary = x_earth * UA_KM

print(f"μ                    = {MU:.4e}")
print(f"x_Soleil             = {x_sun:.4e}  adim.  ({d_sun_bary:.0f} km du barycentre)")
print(
    f"x_Terre              = {x_earth:.8f}  adim.  ({d_earth_bary:.0f} km du barycentre)"
)
print(f"x_barycentre         = {x_bary}  (origine du repère)")
print(f"Distance Soleil-Terre= {UA_KM:.4e} km = 1 UA")
print(
    f"Vérification : μ·x_T + (1-μ)·x_S = {MU*x_earth + (1-MU)*x_sun:.2e}  (doit être 0)"
)

# ── Figure ──
fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0D0D1A")
ax.set_facecolor("#0D0D1A")

# Axe horizontal (ligne de base)
ax.axhline(0, color="#2A2A3E", lw=0.8, zorder=0)

# Soleil
ax.scatter(
    x_sun, 0, s=1200, color="#FFD700", zorder=5, edgecolors="#FFF4A0", linewidths=1.5
)
ax.annotate(
    f"Soleil\nx = −μ = {x_sun:.2e}",
    xy=(x_sun, 0),
    xytext=(x_sun - 0.1, 0.018),
    color="#FFD700",
    fontsize=9,
    fontfamily="monospace",
    arrowprops=dict(arrowstyle="->", color="#FFD700", lw=0.8),
)

# Terre
ax.scatter(
    x_earth, 0, s=180, color="#3A9BD5", zorder=5, edgecolors="#A0D8F0", linewidths=1.2
)
ax.annotate(
    f"Terre\nx = 1-μ = {x_earth:.6f}",
    xy=(x_earth, 0),
    xytext=(x_earth - 0.15, -0.018),
    color="#3A9BD5",
    fontsize=9,
    fontfamily="monospace",
    arrowprops=dict(arrowstyle="->", color="#3A9BD5", lw=0.8),
)

# Barycentre
ax.scatter(x_bary, 0, s=60, color="#FF6B6B", marker="+", linewidths=2.0, zorder=6)
ax.annotate(
    f"Barycentre\nx = 0  (origine)",
    xy=(x_bary, 0),
    xytext=(0.08, 0.018),
    color="#FF6B6B",
    fontsize=9,
    fontfamily="monospace",
    arrowprops=dict(arrowstyle="->", color="#FF6B6B", lw=0.8),
)

# Cote μ : distance Soleil → barycentre
ax.annotate(
    "",
    xy=(0, -0.03),
    xytext=(x_sun, -0.03),
    arrowprops=dict(arrowstyle="<->", color="#FFD700", lw=1.0),
)
ax.text(
    (x_sun + 0) / 2,
    -0.038,
    f"μ = {MU:.2e}",
    color="#FFD700",
    ha="center",
    fontsize=8,
    fontfamily="monospace",
)

# Cote 1−μ : distance barycentre → Terre
ax.annotate(
    "",
    xy=(x_earth, -0.03),
    xytext=(0, -0.03),
    arrowprops=dict(arrowstyle="<->", color="#3A9BD5", lw=1.0),
)
ax.text(
    x_earth / 2,
    -0.038,
    f"1-μ ≈ {1-MU:.6f}",
    color="#3A9BD5",
    ha="center",
    fontsize=8,
    fontfamily="monospace",
)

# Cote totale : 1 UA
ax.annotate(
    "",
    xy=(x_earth, 0.035),
    xytext=(x_sun, 0.035),
    arrowprops=dict(arrowstyle="<->", color="#E0E0E0", lw=0.8),
)
ax.text(
    (x_sun + x_earth) / 2,
    0.042,
    "1 UA (unité adim.)",
    color="#E0E0E0",
    ha="center",
    fontsize=8,
    fontfamily="monospace",
)

ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.07, 0.07)
ax.set_xlabel("x  [adim., 1 = 1 UA]", color="#E0E0E0", fontfamily="monospace")
ax.set_yticks([])
ax.tick_params(colors="#E0E0E0")
ax.set_title(
    f"Repère tournant CR3BP — Soleil, Terre, Barycentre\n"
    f"Le barycentre est à {d_sun_bary:.0f} 000 km du centre du Soleil "
    f"(≈ {d_sun_bary / 6.957e5:.5f} R☉)",
    color="#E0E0E0",
    fontfamily="monospace",
    fontsize=10,
)
for spine in ax.spines.values():
    spine.set_edgecolor("#2A2A3E")

plt.tight_layout()
plt.show()
