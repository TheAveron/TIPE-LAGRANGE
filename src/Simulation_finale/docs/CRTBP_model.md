# CRTBP (Circular Restricted Three-Body Problem)

Ce document décrit l'implémentation du CRTBP dans `src/simulation/CRTBP_model_dynamics.py` ainsi que son utilisation pour l'analyse et la génération d'orbites périodiques.

## Objectif

Le module implémente le modèle restreint circulaire à trois corps et fournit :

- la classe `CRTBP3Body` (héritant de `BaseDynamics`)
- le calcul des accélérations dans le référentiel tournant
- des utilitaires : potentiel effectif, constante de Jacobi

Ce modèle est principalement destiné à l'analyse qualitative (Lagrange points, régions interdites, vecteurs propres, orbites périodiques). Pour des simulations opérationnelles, préférez le modèle haute-fidélité (`MHF_ephem_dynamics`).

## Hypothèses du modèle

1. Les deux primaires (ex : Soleil et Terre) sont en orbite circulaire.
2. Les primaires sont ponctuels (pas de J2, pas d'oblateness).
3. Le troisième corps (vaisseau spatial) a une masse négligeable.
4. Pas de perturbations externes (SRP, corps tiers) sauf si explicitement ajoutés.

## Utilisation principale

Initialisation :

```python
from simulation.dynamics_conf import DynamicsConfig, DynamicsModel
from simulation.CRTBP_model_dynamics import CRTBP3Body

config = DynamicsConfig(model=DynamicsModel.CRTBP)
crtbp = CRTBP3Body(config, normalized=False, primary='sun-earth')
```

Points importants :

- `normalized=True` active les unités canoniques (L=1, ω=1) utiles pour études théoriques.
- `primary` peut être `"sun-earth"` ou `"earth-moon"`.

## Méthodes et propriétés

- `equations_of_motion(t, state)` : retourne la dérivée d'état `[vx, vy, vz, ax, ay, az]`.
- `compute_acceleration(t, state)` : calcule l'accélération en composants.
- `jacobi_constant(state)` : calcule la constante de Jacobi (invariant du CRTBP pur).
- `effective_potential(x,y,z)` : potentiel effectif U*, utile pour tracer les courbes de Hill.

## Limites et recommandations

- Le CRTBP néglige l'excentricité : prévoir des écarts importants pour des comparaisons temporelles longues.
- Pour la génération d'orbites périodiques (halo, Lyapunov), combiner `CRTBP3Body` avec `orbit_generator.py`.

## Références et formules

- Potentiel effectif (normalisé) : U* = (1−μ)/r1 + μ/r2 + ½(x² + y²)
- Équations du mouvement dans le référentiel tournant : r̈ − 2Ω × ṙ − Ω × (Ω × r) = −∇U
- Constante de Jacobi :  C = 2U* - v²

Voir les commentaires dans le fichier source pour la dérivation complète.
