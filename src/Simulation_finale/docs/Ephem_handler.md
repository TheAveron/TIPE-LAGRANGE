# Gestion des éphémérides (SPICE)

Fichier source : `src/simulation/Ephem_handler.py`

Ce module contient `EphemerisManager`, un utilitaire pour charger des kernels SPICE et obtenir des états position/vitesse des corps célestes.

## Kernels requis

- `naif0012.tls` (LSK - leap seconds)
- `pck00010.tpc` (PCK - constantes planétaires)
- `de440.bsp` (SPK - éphémérides planétaires) — le dépôt contient `data/spice/de440.bsp`.

Les fichiers doivent être placés dans `data/spice/` ou fournis via l'argument `kernel_dir` du gestionnaire.

## Fonctionnalités

- Chargement et déchargement des kernels (ordre critique : LSK → PCK → SPK).
- Conversion temps J2000.0 → ET (compatibilité SPICE).
- Méthode `get_body_state(body, time_et, observer, reference_frame)` retourne `(position [m], vitesse [m/s])`.
- Cache des positions pour améliorer les performances sur requêtes rapprochées dans le temps.

## API et comportements

- `EphemerisManager(kernel_dir=None)` : initialise et charge les kernels.
- `get_body_state('EARTH', et)` : renvoie l'état dans le référentiel demandé. Attention : SPICE renvoie des valeurs en km et km/s → `EphemerisManager` convertit en m et m/s.
- `get_earth_moon_barycenter(et)` : utilité pour corriger `include_moon` dans les transformations.
- `clear_cache()` / `unload_kernels()` pour la maintenance.

## Erreurs courantes

- `FileNotFoundError` si les kernels ne sont pas trouvés dans le dossier attendu. Le message indique l'URL de téléchargement NAIF.
- Exceptions SPICE lors du chargement : vérifier la compatibilité des kernels.

## Tests rapides

Le module contient `test_ephemeris_manager()` (nécessite les kernels) qui effectue des vérifications basiques (position Terre à J2000 ≈ 1 AU, distance Terre-Lune ≈ 384400 km, barycentre Terre-Lune cohérent).
