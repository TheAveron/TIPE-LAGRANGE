import numpy as np


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
