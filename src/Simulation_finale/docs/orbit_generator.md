# Génération d'orbites périodiques

Fichier source : `src/simulation/orbit_generator.py`

Ce module fournit des outils pour construire des conditions initiales d'orbites périodiques autour des points de Lagrange, en particulier L2 : Lyapunov, halo et quasi-halo.

## Méthode utilisée

- Correcteur différentiel simple : estimer la vitesse initiale → propager jusqu'au croisement y=0 → corriger vy/vz → répéter.
- Intégrateur : `scipy.integrate.solve_ivp` (méthode `DOP853`) avec tolérances strictes.

## Fonction principale

- `generate_l2_periodic_orbit(crtbp_model, amplitude_y, amplitude_z, ...)` :
  - Entrée : instance `CRTBP3Body`, amplitudes désirées en `y`/`z` (m).
  - Sortie : `(state_initial, period)` en RLP (m, m/s).

## Validation

- `validate_periodic_orbit(crtbp_model, state_initial, period)` : propage sur une période et vérifie la fermeture (erreur < 1% par défaut).

## Conseils pratiques

- Les générateurs d'orbites sont sensibles aux tolérances numériques : augmenter `rtol/atol` si l'on veut plus de robustesse au prix de précision.
- Utiliser des estimations d'amplitude raisonnables (ex: 100e6 m pour y, 50e6 m pour z) pour converger rapidement.
- Les fonctions sont décorées `@njit` (numba) pour accélérer la boucle principale — attention à la compatibilité si vous déboguez (retirer `njit` temporairement).

Voir les tests et exemples dans `tools/` pour des scripts d'illustration.
