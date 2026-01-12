# Configuration des modèles dynamiques

Fichier source : `src/simulation/dynamics_conf.py`

Ce module centralise la configuration et l'interface commune des modèles dynamiques.

## Structures principales

- `DynamicsModel` (Enum) : identifie le modèle à utiliser (`CRTBP`, `EPHEMERIS`, `EPHEMERIS_SRP`, `FULL`).
- `DynamicsConfig` (dataclass) : regroupe les options communes : inclusion de la Lune, SRP, effets relativistes, ratio A/m et coefficient de réflectivité.

## `BaseDynamics`

Classe abstraite définissant l'interface attendue pour tout modèle dynamique :

- `equations_of_motion(t, state)` → dérivée d'état (utilisable directement par `solve_ivp`).
- `compute_acceleration(t, state)` → accélération seule (utile pour diagnostics et contrôles).

## Bonnes pratiques

- Construire un `DynamicsConfig` et le passer au constructeur du modèle choisi.
- Utiliser `DynamicsModel.FULL` pour activer automatiquement SRP et effets relativistes.

Exemple :

```
from simulation.dynamics_conf import DynamicsConfig, DynamicsModel
from simulation.MHF_ephem_dynamics import HighFidelityDynamics

cfg = DynamicsConfig(model=DynamicsModel.FULL, include_moon=True, include_srp=True)
dyn = HighFidelityDynamics(cfg)
```

## Validation

La `__post_init__` de `DynamicsConfig` force des relations logiques (par ex. `FULL` active `include_srp` et `include_relativity`) et vérifie la cohérence du ratio A/m si SRP est activé.
