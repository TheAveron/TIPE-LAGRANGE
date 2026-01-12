# Modèle haute-fidélité (MHF) avec éphémérides

Fichier source : `src/simulation/MHF_ephem_dynamics.py`

Ce module fournit `HighFidelityDynamics`, un modèle dynamique complet destiné aux simulations opérationnelles du spacecraft (JWST) autour de L2.

## Inclut

1. Gravité N-corps via SPICE (Soleil, planètes, Lune, optionnellement astéroïdes massifs)
2. Pression de radiation solaire (SRP) — modèle canon-ball par défaut
3. Corrections relativistes post-Newtoniennes (squelettes, optionnel)
4. Effets J2 et traînée (implémentés mais négligeables à L2)

## Configuration

Initialisation via `DynamicsConfig` (voir `dynamics_conf.md`). Si aucun `EphemerisManager` n'est fourni, le module en crée un et charge les kernels par défaut.

Exemple :

```
from simulation.dynamics_conf import DynamicsConfig, DynamicsModel
from simulation.MHF_ephem_dynamics import HighFidelityDynamics

cfg = DynamicsConfig(model=DynamicsModel.EPHEMERIS_SRP, include_srp=True, include_moon=True)
hf = HighFidelityDynamics(cfg)
```

## Méthodes importantes

- `equations_of_motion(t, state)` : retourne `[vx, vy, vz, ax, ay, az]` en écliptique J2000.
- `compute_acceleration(t, state)` : somme des contributions : `a_gravity + a_srp + a_relativity + a_j2 + a_drag`.

## Détails d'implémentation notables

- Gravité : la méthode `_compute_gravitational_acceleration` interroge `EphemerisManager` et accumule les contributions μ×(r_i − r)/|r_i − r|³.
- SRP : modèle canon-ball avec `P_srp_1au` adapté et factor `(1 + ρ)` où ρ est le coefficient de réflectivité.
- Relativité : stub implémenté (effet très faible pour L2). Peut être étendu si nécessaire.
- Optimisation : cache des positions planétaires par pas temporel pour accélérer les évaluations successives.

## Limitations et recommandations

- Ce modèle est plus coûteux en calcul ; activer le cache et limiter la fréquence d'appel à SPICE si possible.
- Pour analyses rapides, utiliser le CRTBP (moins précis mais beaucoup moins coûteux).

Voir les commentaires détaillés dans le code source pour formules et choix numériques.
