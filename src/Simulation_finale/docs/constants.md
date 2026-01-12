# Constantes

Ce document décrit les constantes centralisées dans `src/simulation/constants.py` utilisées par l'ensemble des modules de simulation. L'objectif est de fournir un point unique de vérité pour les constantes physiques, paramètres de la mission JWST et tolérances numériques.

**Sources principales**:

- [NASA Fact Sheet - Sun](https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html)
- [NASA Fact Sheet - Earth](https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html)
- [NASA Fact Sheet - Moon](https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html)
- [NASA - Astronomical Constants](https://nssdc.gsfc.nasa.gov/planetary/const.html)
- [Wikipedia - Gravitational parameter](https://en.wikipedia.org/wiki/Gravitational_parameter)

- [The nist reference on constants, units, and uncertainty](https://physics.nist.gov/cuu/Constants/index.html)

**Organisation**:

- **`Constants`** : constantes physiques fondamentales (G, c, masses, distances, μ, unités de normalisation).
- **`JWSTParameters`** : paramètres spécifiques à la mission (masse, surface du pare-soleil, contraintes d'attitude, station-keeping, amplitudes d'orbite autour de L2).
- **`NumericalConstants`** : tolérances et unités de normalisation utilisées par les intégrateurs et correcteurs.

**Bonnes pratiques**:

- Importer uniquement les constantes nécessaires : `from simulation.constants import Constants`.
- Ne pas modifier ces valeurs sans référence bibliographique claire.

**Exemples d'utilisation**:

- Normalisation CRTBP :

	- Longueur caractéristique : `L_star = Constants.AU`
	- Temps caractéristique : `T_star = 1.0 / Constants.OMEGA_EARTH`
	- Vitesse caractéristique : `V_star = L_star / T_star`

- Paramètres d'intégration :

	- `atol = NumericalConstants.INTEGRATION_ATOL_HIGH`
	- `rtol = NumericalConstants.INTEGRATION_RTOL_HIGH`

**Validation**:

Le module inclut des valeurs cohérentes (par exemple `MU_SUN`, `MU_EARTH`) et des fonctions utilitaires dans les tests (`src/tests/tests_constants.py`) qui vérifient des relations physiques (distance Terre–Soleil ≈ 1 AU, distance Terre–Lune ≈ 384400 km, etc.).

Voir également: [coordinates.md](coordinates.md) pour l'utilisation de ces constantes lors des transformations de référentiels.
