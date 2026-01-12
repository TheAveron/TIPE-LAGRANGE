# Coordonnées

## Classe ReferenceFrame

Cette classe définit les différents systèmes de coordonnées utilisés dans les simulations orbitales.

## Classe CoordinateTransformer

Cette classe gère les transformations entre différents systèmes de coordonnées utilisés dans les simulations orbitales. Elle inclut des méthodes pour convertir les positions et vitesses entre les cadres écliptiques J2000, le cadre rotatif lié au système Terre-Lune (RLP).

### Variables des instances

- `include_moon`: Booléen indiquant si la Lune doit être prise en compte dans les transformations.
- `ephem_manager`: Instance de `EphemerisManager` pour accéder aux éphémérides planétaires.
- `mu_ratio`: Paramètre μ du système considéré (Terre-Lune ou Soleil-Terre).
- `R`: Distance entre les deux corps principaux du système. (Défini par distance Terre-Soleil).
- `omega`: Vitesse angulaire du système. (Définie par la période orbitale de la Terre autour du Soleil).

Si `include_moon` est `True`, on ajoute une correction pour la position de la Terre.

### Méthodes principales

#### `rlp_to_ecliptic` et `ecliptic_to_rlp`: Convertissent les coordonnées entre le cadre RLP et le cadre écliptique J2000

Cette méthode applique les transitions adéquates, c'est à dire:

- Transformation de coordonnées rotatives/non-rotatives.
- Transformation des vitesses en tenant compte de la rotation du cadre.
- Ajout/soustraction de la position de la Terre (et de la Lune si `include_moon` est `True`).

#### `rlp_to_crtbp` et `crt_bp_to_rlp`: Convertissent les coordonnées entre le cadre RLP et le cadre CRTBP (Circular Restricted Three-Body Problem)

Ces méthodes normalisent les positions et vitesses en fonction des paramètres du système (distance entre les corps principaux et vitesse angulaire).

Les transformations appliquées sont les suivantes:

- Normalisation/dénormalisation des positions par la distance `R`.
- Normalisation/dénormalisation des vitesses par `R * omega`.

#### `eci_to_ecliptic` et `ecliptic_to_eci`: Convertissent les coordonnées entre le cadre ECI (Earth-Centered Inertial) et le cadre écliptique J2000

Ces méthodes efectues des translations basées sur la position de la Terre (et de la Lune si `include_moon` est `True`) pour passer entre les cadres centrés sur la Terre et centrés sur le Soleil.

#### `_compute_earth_circular_orbit`: Calcule la position et la vitesse de la Terre en supposant une orbite circulaire autour du Soleil

Pour ce faire on applique les formules suivantes:

- Projection de la position de la Terre
- Calcul de la vitesse orbitale circulaire à l'aide de la constante gravitationnelle et de la distance au Soleil.

#### `compute_l2_distance_from_earth`: Calcule la distance approximative du point de Lagrange L2 par rapport à la Terre

Pour se faire, utilise la formule approchée basée sur le paramètre μ du système Soleil-Terre:
$r_{L2} \approx R \left( \frac{\mu}{3} \right)^{1/3}$
où $R$ est la distance entre la Terre et le Soleil, et $ \mu $ est le rapport de la masse de la Terre à la somme des masses du Soleil et de la Terre. Cette distance est utile pour positionner des satellites comme le JWST près du point L2.

Cette aproximation est obtenue en résolvant l'équation de position des points de Lagrange dans le cadre du problème à trois corps restreint circulaire, en supposant que la distance entre les deux corps principaux (Soleil et Terre) est grande par rapport à la distance entre la Terre et le point L2.

Voici la démonstration:

Considérons le système Soleil-Terre avec les masses respectives $M_{S}$ et $M_{E}$, et la distance entre elles $R$. Le point de Lagrange L2 se trouve sur la ligne reliant les deux corps, au-delà de la Terre.

La position de L2, notée $r_{L2}$, peut être approximée en résolvant l'équation de la force gravitationnelle et de la force centrifuge dans le cadre du problème à trois corps restreint circulaire. En supposant que $r_{L2}$ est proche de la Terre, on peut écrire:
$$\frac{G M_{S}}{(R + r_{L2})^2} + \frac{G M_{E}}{r_{L2}^2} = \omega^2 (R + r_{L2})$$
où $\omega$ est la vitesse angulaire du système.

En utilisant l'approximation que $r_{L2} \ll R$, on peut simplifier cette équation et obtenir:
$$r_{L2} \approx R \left( \frac{\mu}{3} \right)^{1/3}$$
où $\mu = \frac{M_{E}}{M_{S} + M_{E}}$ est le rapport de la masse de la Terre à la somme des masses du Soleil et de la Terre.

#### `get_transformation_info`: Fournit des informations sur les transformations effectuées entre deux cadres de référence donnés

Les infos données incluent:

- Les cadres de référence source et cible.
- Le changement d'origine.
- Si la transformation inclut une rotation, et si oui l'angle de rotation.
- Les changements d'unités de longueurs.
