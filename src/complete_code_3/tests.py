"""
Simulateur de trajectoire pour le satellite James Webb autour du point de Lagrange L2
Système Soleil-Terre dans le référentiel héliocentrique inertiel
"""

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# CONSTANTES PHYSIQUES (Valeurs réelles)
# ============================================================================


class Constants:
    """Constantes physiques du système Soleil-Terre"""

    # Constante gravitationnelle [m^3 kg^-1 s^-2]
    G = 6.67430e-11

    # Masses [kg]
    M_SUN = 1.98892e30  # Masse du Soleil
    M_EARTH = 5.97219e24  # Masse de la Terre

    # Distance Terre-Soleil (1 UA) [m]
    AU = 1.495978707e11

    # Rayon de l'orbite terrestre (circulaire) [m]
    R_EARTH_ORBIT = AU

    # Vitesse angulaire du système Soleil-Terre [rad/s]
    # ω = 2π / T où T = 365.25 jours
    T_YEAR = 365.25 * 24 * 3600  # Période en secondes
    OMEGA = 2 * np.pi / T_YEAR

    # Vitesse orbitale de la Terre [m/s]
    V_EARTH = OMEGA * R_EARTH_ORBIT

    # Paramètre gravitationnel du Soleil μ_S = G*M_S [m^3/s^2]
    MU_SUN = G * M_SUN

    # Paramètre gravitationnel de la Terre μ_T = G*M_T [m^3/s^2]
    MU_EARTH = G * M_EARTH


# ============================================================================
# CLASSE VECTOR3D
# ============================================================================


class Vector3D:
    """Vecteur 3D avec opérations vectorielles"""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Crée un Vector3D depuis un array numpy"""
        return cls(arr[0], arr[1], arr[2])

    def to_array(self) -> np.ndarray:
        """Convertit en array numpy"""
        return np.array([self.x, self.y, self.z])

    def __add__(self, other):
        """Addition de vecteurs"""
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """Soustraction de vecteurs"""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float):
        """Multiplication par un scalaire"""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float):
        """Multiplication par un scalaire (ordre inverse)"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        """Division par un scalaire"""
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self):
        """Négation"""
        return Vector3D(-self.x, -self.y, -self.z)

    def dot(self, other) -> float:
        """Produit scalaire"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """Produit vectoriel"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        """Norme euclidienne"""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        """Vecteur unitaire dans la même direction"""
        n = self.norm()
        if n == 0:
            return Vector3D(0, 0, 0)
        return self / n

    def __repr__(self):
        return f"Vector3D({self.x:.3e}, {self.y:.3e}, {self.z:.3e})"


# ============================================================================
# CLASSE STATE
# ============================================================================


@dataclass
class State:
    """État d'un objet : position et vitesse"""

    position: Vector3D
    velocity: Vector3D

    def to_array(self) -> np.ndarray:
        """Convertit en array numpy [x, y, z, vx, vy, vz]"""
        return np.concatenate([self.position.to_array(), self.velocity.to_array()])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Crée un State depuis un array numpy"""
        return cls(
            position=Vector3D.from_array(arr[0:3]),
            velocity=Vector3D.from_array(arr[3:6]),
        )

    def copy(self):
        """Copie profonde de l'état"""
        return State(
            position=Vector3D(self.position.x, self.position.y, self.position.z),
            velocity=Vector3D(self.velocity.x, self.velocity.y, self.velocity.z),
        )


# ============================================================================
# CLASSE CELESTIAL BODY
# ============================================================================


class CelestialBody:
    """Corps céleste avec propriétés physiques"""

    def __init__(
        self,
        name: str,
        mass: float,
        initial_position: Vector3D,
        initial_velocity: Vector3D,
    ):
        self.name = name
        self.mass = mass
        self.mu = Constants.G * mass  # Paramètre gravitationnel
        self.initial_state = State(initial_position, initial_velocity)

    def get_state_at_time(self, t: float) -> State:
        """
        Retourne l'état du corps à l'instant t
        Pour l'instant, orbite circulaire simplifiée
        """
        # Soleil : fixe à l'origine
        if self.name == "Sun":
            return self.initial_state

        # Terre : orbite circulaire dans le plan (x,y)
        if self.name == "Earth":
            angle = Constants.OMEGA * t
            r = Constants.R_EARTH_ORBIT

            position = Vector3D(r * np.cos(angle), r * np.sin(angle), 0.0)

            velocity = Vector3D(
                -Constants.V_EARTH * np.sin(angle),
                Constants.V_EARTH * np.cos(angle),
                0.0,
            )

            return State(position, velocity)

        return self.initial_state


# ============================================================================
# FONCTIONS DE CALCUL DES FORCES
# ============================================================================


def gravitational_force(
    pos_satellite: Vector3D, pos_body: Vector3D, mass_satellite: float, mu_body: float
) -> Vector3D:
    """
    Calcule la force gravitationnelle exercée par un corps sur le satellite

    F = -G * M * m / r^2 * r̂ = -μ * m / r^2 * r̂

    Args:
        pos_satellite: Position du satellite
        pos_body: Position du corps attracteur
        mass_satellite: Masse du satellite [kg]
        mu_body: Paramètre gravitationnel du corps (G*M) [m^3/s^2]

    Returns:
        Force gravitationnelle [N]
    """
    r_vec = pos_satellite - pos_body
    r = r_vec.norm()

    if r == 0:
        return Vector3D(0, 0, 0)

    # F = -μ * m / r^3 * r_vec (le r^3 vient de r^2 * r)
    force_magnitude = -mu_body * mass_satellite / (r**3)
    force = r_vec * force_magnitude

    return force


def total_acceleration(
    pos_satellite: Vector3D, bodies: List[CelestialBody], t: float
) -> Vector3D:
    """
    Calcule l'accélération totale du satellite due à tous les corps

    Args:
        pos_satellite: Position du satellite
        bodies: Liste des corps célestes
        t: Temps actuel [s]

    Returns:
        Accélération totale [m/s^2]
    """
    acceleration = Vector3D(0, 0, 0)
    mass_satellite = 1.0  # On calcule F/m directement

    for body in bodies:
        body_state = body.get_state_at_time(t)
        force = gravitational_force(
            pos_satellite, body_state.position, mass_satellite, body.mu
        )
        acceleration = acceleration + force  # F/m car mass=1

    return acceleration


# ============================================================================
# INTÉGRATEUR RK4
# ============================================================================


def derivative(state: State, t: float, bodies: List[CelestialBody]) -> State:
    """
    Calcule la dérivée de l'état : dX/dt = (v, a)

    Args:
        state: État actuel (position, vitesse)
        t: Temps actuel [s]
        bodies: Liste des corps célestes

    Returns:
        Dérivée de l'état (vitesse, accélération)
    """
    acceleration = total_acceleration(state.position, bodies, t)
    return State(position=state.velocity, velocity=acceleration)


def rk4_step(state: State, t: float, dt: float, bodies: List[CelestialBody]) -> State:
    """
    Effectue un pas d'intégration Runge-Kutta d'ordre 4

    Args:
        state: État actuel
        t: Temps actuel [s]
        dt: Pas de temps [s]
        bodies: Liste des corps célestes

    Returns:
        Nouvel état après le pas de temps
    """
    # k1 = f(t, y)
    k1 = derivative(state, t, bodies)

    # k2 = f(t + dt/2, y + dt/2 * k1)
    state2 = State(
        position=state.position + k1.position * (dt / 2),
        velocity=state.velocity + k1.velocity * (dt / 2),
    )
    k2 = derivative(state2, t + dt / 2, bodies)

    # k3 = f(t + dt/2, y + dt/2 * k2)
    state3 = State(
        position=state.position + k2.position * (dt / 2),
        velocity=state.velocity + k2.velocity * (dt / 2),
    )
    k3 = derivative(state3, t + dt / 2, bodies)

    # k4 = f(t + dt, y + dt * k3)
    state4 = State(
        position=state.position + k3.position * dt,
        velocity=state.velocity + k3.velocity * dt,
    )
    k4 = derivative(state4, t + dt, bodies)

    # y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    new_position = state.position + (
        k1.position + k2.position * 2 + k3.position * 2 + k4.position
    ) * (dt / 6)
    new_velocity = state.velocity + (
        k1.velocity + k2.velocity * 2 + k3.velocity * 2 + k4.velocity
    ) * (dt / 6)

    return State(position=new_position, velocity=new_velocity)


# ============================================================================
# CALCUL DES POINTS DE LAGRANGE
# ============================================================================


def calculate_lagrange_points() -> dict:
    """
    Calcule les positions des 5 points de Lagrange dans le référentiel tournant
    Pour le système Soleil-Terre avec orbite circulaire

    Returns:
        Dictionnaire avec les positions des points L1 à L5
    """
    # Paramètre de masse μ = M_T / (M_S + M_T)
    mu = Constants.M_EARTH / (Constants.M_SUN + Constants.M_EARTH)
    r = Constants.R_EARTH_ORBIT

    # L1, L2, L3 : approximations au premier ordre
    # Position exacte nécessite résolution numérique d'équation quintique
    # Approximation : L2 est à r * (1 + (mu/3)^(1/3)) du Soleil

    # L1 : entre Soleil et Terre
    r_L1 = r * (1 - (mu / 3) ** (1 / 3))

    # L2 : au-delà de la Terre (où se trouve JWST)
    r_L2 = r * (1 + (mu / 3) ** (1 / 3))

    # L3 : opposé à la Terre
    r_L3 = -r * (1 + 5 * mu / 12)

    # À t=0, la Terre est sur l'axe +x
    lagrange_points = {
        "L1": Vector3D(r_L1, 0, 0),
        "L2": Vector3D(r_L2, 0, 0),
        "L3": Vector3D(r_L3, 0, 0),
        "L4": Vector3D(r * np.cos(np.pi / 3), r * np.sin(np.pi / 3), 0),
        "L5": Vector3D(r * np.cos(-np.pi / 3), r * np.sin(-np.pi / 3), 0),
    }

    return lagrange_points


# ============================================================================
# CLASSE SIMULATOR
# ============================================================================


class Simulator:
    """Simulateur principal pour la propagation de trajectoires"""

    def __init__(self):
        """Initialise le simulateur avec le système Soleil-Terre"""
        # Création des corps célestes
        self.sun = CelestialBody(
            name="Sun",
            mass=Constants.M_SUN,
            initial_position=Vector3D(0, 0, 0),
            initial_velocity=Vector3D(0, 0, 0),
        )

        self.earth = CelestialBody(
            name="Earth",
            mass=Constants.M_EARTH,
            initial_position=Vector3D(Constants.R_EARTH_ORBIT, 0, 0),
            initial_velocity=Vector3D(0, Constants.V_EARTH, 0),
        )

        self.bodies = [self.sun, self.earth]

        # Calcul des points de Lagrange
        self.lagrange_points = calculate_lagrange_points()

        # Générateur d'orbites de halo
        self.halo_generator = HaloOrbitGenerator()

        # Contraintes d'attitude
        self.attitude_constraints = AttitudeConstraints()

        # Historique de la simulation
        self.history = []

    def propagate(
        self, initial_state: State, t_start: float, t_end: float, dt: float
    ) -> List[Tuple[float, State]]:
        """
        Propage la trajectoire d'un satellite

        Args:
            initial_state: État initial du satellite
            t_start: Temps de début [s]
            t_end: Temps de fin [s]
            dt: Pas de temps [s]

        Returns:
            Liste de tuples (temps, état)
        """
        history = []
        state = initial_state.copy()
        t = t_start

        # Enregistrement de l'état initial
        history.append((t, state.copy()))

        # Boucle d'intégration
        while t < t_end:
            state = rk4_step(state, t, dt, self.bodies)
            t += dt
            history.append((t, state.copy()))

        self.history = history
        return history

    def get_energy(self, state: State, t: float) -> float:
        """
        Calcule l'énergie mécanique totale du satellite
        E = E_cinétique + E_potentielle

        Args:
            state: État du satellite
            t: Temps [s]

        Returns:
            Énergie totale [J/kg] (énergie spécifique)
        """
        # Énergie cinétique spécifique
        v = state.velocity.norm()
        E_kin = 0.5 * v**2

        # Énergie potentielle spécifique
        E_pot = 0.0
        for body in self.bodies:
            body_state = body.get_state_at_time(t)
            r = (state.position - body_state.position).norm()
            if r > 0:
                E_pot -= body.mu / r

        return E_kin + E_pot

    def analyze_orbit_characteristics(
        self, history: List[Tuple[float, State]], reference_frame: str = "rotating"
    ) -> dict:
        """
        Analyse les caractéristiques d'une orbite simulée

        Args:
            history: Historique de simulation
            reference_frame: "inertial" ou "rotating"

        Returns:
            Dictionnaire avec les caractéristiques orbitales
        """
        times = np.array([t for t, _ in history])

        if reference_frame == "rotating":
            # Conversion vers le référentiel tournant
            positions = []
            for t, state in history:
                angle = -Constants.OMEGA * t
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
                pos_array = state.position.to_array()
                pos_rot = R @ pos_array
                positions.append(pos_rot)
        else:
            positions = [state.position.to_array() for _, state in history]

        positions = np.array(positions)

        # Calcul des amplitudes
        x_amp = np.max(np.abs(positions[:, 0]))
        y_amp = np.max(np.abs(positions[:, 1]))
        z_amp = np.max(np.abs(positions[:, 2]))

        # Distance à L2
        L2_pos = self.lagrange_points["L2"].to_array()
        if reference_frame == "rotating":
            distances_L2 = np.linalg.norm(positions - L2_pos, axis=1)
        else:
            # L2 tourne dans le référentiel inertiel
            distances_L2 = []
            for t, state in history:
                angle = Constants.OMEGA * t
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
                L2_rot = R @ L2_pos
                dist = np.linalg.norm(state.position.to_array() - L2_rot)
                distances_L2.append(dist)
            distances_L2 = np.array(distances_L2)

        # Croisements du plan XZ (y = 0)
        y_positions = positions[:, 1]
        crossings = []
        for i in range(1, len(y_positions)):
            if y_positions[i - 1] * y_positions[i] < 0:  # Changement de signe
                crossings.append(times[i])

        # Période approximative (temps entre 2 croisements successifs)
        if len(crossings) >= 2:
            period_estimate = np.mean(np.diff(crossings)) * 2  # Demi-période
        else:
            period_estimate = None

        return {
            "amplitude_x": x_amp,
            "amplitude_y": y_amp,
            "amplitude_z": z_amp,
            "min_distance_L2": np.min(distances_L2),
            "max_distance_L2": np.max(distances_L2),
            "mean_distance_L2": np.mean(distances_L2),
            "xz_plane_crossings": crossings,
            "estimated_period": period_estimate,
            "duration": times[-1] - times[0],
        }

    def check_jwst_compatibility(self, characteristics: dict) -> dict:
        """
        Vérifie si une orbite est compatible avec les contraintes JWST

        Args:
            characteristics: Caractéristiques orbitales

        Returns:
            Dictionnaire avec les résultats de vérification
        """
        # Contraintes approximatives basées sur les documents NASA
        # Amplitudes typiques pour JWST: Y~771,000 km, Z~418,000 km

        checks = {}

        # Vérification des amplitudes (en km)
        y_amp_km = characteristics["amplitude_y"] / 1e3
        z_amp_km = characteristics["amplitude_z"] / 1e3

        # Tolérance large pour une orbite quasi-halo
        checks["y_amplitude_ok"] = 400e3 < y_amp_km < 1200e3
        checks["z_amplitude_ok"] = 200e3 < z_amp_km < 700e3

        # Distance à L2
        mean_dist_km = characteristics["mean_distance_L2"] / 1e3
        checks["l2_distance_ok"] = mean_dist_km < 1000e3  # Moins de 1M km

        # Période (environ 6 mois = 180 jours pour une orbite de halo)
        if characteristics["estimated_period"]:
            period_days = characteristics["estimated_period"] / (24 * 3600)
            checks["period_reasonable"] = 100 < period_days < 250
        else:
            checks["period_reasonable"] = None

        # Résumé
        checks["all_ok"] = all(v for v in checks.values() if v is not None)

        return checks


# ============================================================================
# FONCTIONS DE VISUALISATION 3D
# ============================================================================


class Visualizer:
    """Classe pour la visualisation 3D des trajectoires"""

    def __init__(self, simulator: Simulator):
        self.simulator = simulator

    def plot_trajectory_3d(
        self,
        history: List[Tuple[float, State]],
        title: str = "Trajectoire du satellite",
        show_lagrange: bool = True,
        show_orbits: bool = True,
        reference_frame: str = "inertial",
    ):
        """
        Affiche la trajectoire 3D du satellite

        Args:
            history: Historique de la simulation
            title: Titre du graphique
            show_lagrange: Afficher les points de Lagrange
            show_orbits: Afficher l'orbite terrestre
            reference_frame: "inertial" ou "rotating"
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Extraction des positions
        times = np.array([t for t, _ in history])
        positions = np.array(
            [[s.position.x, s.position.y, s.position.z] for _, s in history]
        )

        # Conversion en km pour lisibilité
        positions_km = positions / 1e3

        # Si référentiel tournant, rotation inverse
        if reference_frame == "rotating":
            positions_km = self._to_rotating_frame(positions_km, times)

        # Trajectoire du satellite
        ax.plot(
            positions_km[:, 0],
            positions_km[:, 1],
            positions_km[:, 2],
            "b-",
            linewidth=2,
            label="Satellite",
            alpha=0.7,
        )

        # Position initiale et finale
        ax.scatter(
            positions_km[0, 0],
            positions_km[0, 1],
            positions_km[0, 2],
            c="green",
            s=100,
            marker="o",
            label="Début",
            zorder=5,
        )
        ax.scatter(
            positions_km[-1, 0],
            positions_km[-1, 1],
            positions_km[-1, 2],
            c="red",
            s=100,
            marker="X",
            label="Fin",
            zorder=5,
        )

        # Soleil (à l'origine)
        ax.scatter(
            0,
            0,
            0,
            c="yellow",
            s=300,
            marker="o",
            edgecolors="orange",
            linewidths=2,
            label="Soleil",
            zorder=10,
        )

        # Orbite de la Terre
        if show_orbits:
            if reference_frame == "inertial":
                theta = np.linspace(0, 2 * np.pi, 100)
                r_earth_km = Constants.R_EARTH_ORBIT / 1e3
                orbit_x = r_earth_km * np.cos(theta)
                orbit_y = r_earth_km * np.sin(theta)
                orbit_z = np.zeros_like(theta)
                ax.plot(
                    orbit_x,
                    orbit_y,
                    orbit_z,
                    "g--",
                    linewidth=1,
                    alpha=0.3,
                    label="Orbite Terre",
                )

        # Position de la Terre au début et à la fin
        earth_start = self.simulator.earth.get_state_at_time(times[0])
        earth_end = self.simulator.earth.get_state_at_time(times[-1])

        earth_pos_start = (
            np.array(
                [earth_start.position.x, earth_start.position.y, earth_start.position.z]
            )
            / 1e3
        )
        earth_pos_end = (
            np.array([earth_end.position.x, earth_end.position.y, earth_end.position.z])
            / 1e3
        )

        if reference_frame == "rotating":
            earth_pos_start = self._to_rotating_frame(
                earth_pos_start.reshape(1, 3), np.array([times[0]])
            )[0]
            earth_pos_end = self._to_rotating_frame(
                earth_pos_end.reshape(1, 3), np.array([times[-1]])
            )[0]

        ax.scatter(
            earth_pos_start[0],
            earth_pos_start[1],
            earth_pos_start[2],
            c="blue",
            s=200,
            marker="o",
            label="Terre (début)",
            zorder=8,
        )
        ax.scatter(
            earth_pos_end[0],
            earth_pos_end[1],
            earth_pos_end[2],
            c="cyan",
            s=200,
            marker="o",
            label="Terre (fin)",
            alpha=0.6,
            zorder=8,
        )

        # Points de Lagrange
        if show_lagrange and reference_frame == "rotating":
            for name, point in self.simulator.lagrange_points.items():
                pos_km = np.array([point.x, point.y, point.z]) / 1e3
                color = "red" if name == "L2" else "orange"
                size = 150 if name == "L2" else 80
                ax.scatter(
                    pos_km[0],
                    pos_km[1],
                    pos_km[2],
                    c=color,
                    s=size,
                    marker="*",
                    label=name,
                    zorder=9,
                    edgecolors="black",
                    linewidths=1,
                )

        # Configuration des axes
        ax.set_xlabel("X [km]", fontsize=12)
        ax.set_ylabel("Y [km]", fontsize=12)
        ax.set_zlabel("Z [km]", fontsize=12)
        ax.set_title(
            f"{title}\nRéférentiel: {reference_frame}", fontsize=14, fontweight="bold"
        )

        # Légende
        ax.legend(loc="upper left", fontsize=10)

        # Grille
        ax.grid(True, alpha=0.3)

        # Aspect ratio égal
        self._set_axes_equal(ax)

        plt.tight_layout()
        return fig, ax

    def plot_dual_view(
        self,
        history: List[Tuple[float, State]],
        title: str = "Comparaison référentiels",
    ):
        """
        Affiche deux vues côte à côte : inertiel et tournant
        """
        fig = plt.figure(figsize=(18, 8))

        # Vue inertielle
        ax1 = fig.add_subplot(121, projection="3d")
        self._plot_single_view(
            ax1, history, "inertial", "Référentiel héliocentrique inertiel"
        )

        # Vue tournante
        ax2 = fig.add_subplot(122, projection="3d")
        self._plot_single_view(
            ax2, history, "rotating", "Référentiel tournant Soleil-Terre"
        )

        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        return fig, (ax1, ax2)

    def _plot_single_view(self, ax, history, reference_frame, subtitle):
        """Helper pour tracer une vue unique"""
        times = np.array([t for t, _ in history])
        positions = np.array(
            [[s.position.x, s.position.y, s.position.z] for _, s in history]
        )
        positions_km = positions / 1e3

        if reference_frame == "rotating":
            positions_km = self._to_rotating_frame(positions_km, times)

        # Trajectoire
        ax.plot(
            positions_km[:, 0],
            positions_km[:, 1],
            positions_km[:, 2],
            "b-",
            linewidth=2,
            alpha=0.7,
        )

        # Début/Fin
        ax.scatter(
            positions_km[0, 0],
            positions_km[0, 1],
            positions_km[0, 2],
            c="green",
            s=100,
            marker="o",
            zorder=5,
        )
        ax.scatter(
            positions_km[-1, 0],
            positions_km[-1, 1],
            positions_km[-1, 2],
            c="red",
            s=100,
            marker="X",
            zorder=5,
        )

        # Soleil
        ax.scatter(
            0,
            0,
            0,
            c="yellow",
            s=300,
            marker="o",
            edgecolors="orange",
            linewidths=2,
            zorder=10,
        )

        # Terre
        earth_pos = self.simulator.earth.get_state_at_time(times[0])
        earth_km = (
            np.array([earth_pos.position.x, earth_pos.position.y, earth_pos.position.z])
            / 1e3
        )
        if reference_frame == "rotating":
            earth_km = self._to_rotating_frame(
                earth_km.reshape(1, 3), np.array([times[0]])
            )[0]
        ax.scatter(
            earth_km[0], earth_km[1], earth_km[2], c="blue", s=200, marker="o", zorder=8
        )

        # Points de Lagrange (seulement en tournant)
        if reference_frame == "rotating":
            for name, point in self.simulator.lagrange_points.items():
                pos_km = np.array([point.x, point.y, point.z]) / 1e3
                color = "red" if name == "L2" else "orange"
                size = 150 if name == "L2" else 80
                ax.scatter(
                    pos_km[0],
                    pos_km[1],
                    pos_km[2],
                    c=color,
                    s=size,
                    marker="*",
                    zorder=9,
                    edgecolors="black",
                    linewidths=1,
                )

        ax.set_xlabel("X [km]", fontsize=10)
        ax.set_ylabel("Y [km]", fontsize=10)
        ax.set_zlabel("Z [km]", fontsize=10)
        ax.set_title(subtitle, fontsize=12)
        ax.grid(True, alpha=0.3)
        self._set_axes_equal(ax)

    def _to_rotating_frame(
        self, positions: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """
        Convertit des positions du référentiel inertiel au tournant

        Args:
            positions: Array (N, 3) de positions en référentiel inertiel
            times: Array (N,) de temps correspondants

        Returns:
            Positions dans le référentiel tournant
        """
        rotated = np.zeros_like(positions)

        for i, t in enumerate(times):
            angle = -Constants.OMEGA * t  # Rotation inverse
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Matrice de rotation autour de z
            R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

            rotated[i] = R @ positions[i]

        return rotated

    def _set_axes_equal(self, ax):
        """Fixe les limites des axes pour avoir un aspect ratio égal"""
        limits = np.array(
            [
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ]
        )

        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    def plot_energy_conservation(self, history: List[Tuple[float, State]]):
        """
        Affiche l'évolution de l'énergie pour vérifier la conservation
        """
        times = np.array([t for t, _ in history]) / (24 * 3600)  # En jours
        energies = [self.simulator.get_energy(state, t) for t, state in history]

        # Normalisation par rapport à l'énergie initiale
        E0 = energies[0]
        relative_error = [(E - E0) / abs(E0) * 100 for E in energies]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Énergie absolue
        ax1.plot(times, energies, "b-", linewidth=2)
        ax1.set_xlabel("Temps [jours]", fontsize=12)
        ax1.set_ylabel("Énergie spécifique [J/kg]", fontsize=12)
        ax1.set_title(
            "Évolution de l'énergie mécanique", fontsize=14, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3)

        # Erreur relative
        ax2.plot(times, relative_error, "r-", linewidth=2)
        ax2.set_xlabel("Temps [jours]", fontsize=12)
        ax2.set_ylabel("Erreur relative [%]", fontsize=12)
        ax2.set_title("Conservation de l'énergie", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Statistiques
        max_error = max(abs(min(relative_error)), abs(max(relative_error)))
        ax2.text(
            0.02,
            0.98,
            f"Erreur max: {max_error:.2e}%",
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return fig, (ax1, ax2)


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Création du simulateur
    sim = Simulator()
    vis = Visualizer(sim)

    # Test : Simulation satellite près de L2
    print("=== Simulation satellite près de L2 ===")

    # État initial : orbite de halo autour de L2
    # Petite perturbation en position et vitesse
    initial_pos = sim.lagrange_points["L2"] + Vector3D(1e8, 5e7, 3e7)  # km
    initial_vel = Vector3D(-50, Constants.V_EARTH + 10, 5)  # m/s
    initial_state = State(initial_pos, initial_vel)

    # Simulation sur 180 jours
    t_start = 0
    t_end = 180 * 24 * 3600  # 180 jours
    dt = 3600  # 1 heure

    print(f"Durée: {t_end / (24*3600):.0f} jours, Pas: {dt / 3600:.1f} h")
    print("Propagation en cours...")

    history = sim.propagate(initial_state, t_start, t_end, dt)
    print(f"✓ {len(history)} points calculés")

    # Visualisations
    print("\nGénération des graphiques...")

    # Vue 3D dans le référentiel tournant
    fig1, ax1 = vis.plot_trajectory_3d(
        history,
        title="Trajectoire autour de L2 (référentiel tournant)",
        reference_frame="rotating",
    )

    # Vue 3D dans le référentiel inertiel
    fig2, ax2 = vis.plot_trajectory_3d(
        history,
        title="Trajectoire autour de L2 (référentiel inertiel)",
        reference_frame="inertial",
        show_lagrange=False,
    )

    # Comparaison double vue
    fig3, (ax3, ax4) = vis.plot_dual_view(history, title="Comparaison des référentiels")

    # Conservation de l'énergie
    fig4, (ax5, ax6) = vis.plot_energy_conservation(history)

    plt.show()

    print("\n✓ Visualisations générées avec succès !")
