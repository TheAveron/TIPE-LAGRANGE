from dataclasses import dataclass


@dataclass
class ReferenceFrame:
    """
    Classe de données décrivant un référentiel.

    Attributs:
        name: Nom du référentiel
        origin: Description de l'origine
        is_inertial: True si référentiel inertiel (non-rotatif)
        length_unit: Unité de longueur (m pour SI, AU pour normalisé, etc.)
        time_unit: Unité de temps (s pour SI, TU pour normalisé)
    """

    name: str
    origin: str
    is_inertial: bool
    length_unit: str = "m"
    time_unit: str = "s"

    def __str__(self) -> str:
        return f"{self.name} (origine: {self.origin}, inertiel: {self.is_inertial})"


FRAME_ECLIPTIC_J2000 = ReferenceFrame(
    name="Ecliptic J2000.0",
    origin="Barycentre du système solaire",
    is_inertial=True,
    length_unit="m",
    time_unit="s",
)

FRAME_RLP = ReferenceFrame(
    name="Rotating Libration Point (RLP)",
    origin="Barycentre Soleil-Terre(-Lune)",
    is_inertial=False,
    length_unit="m",
    time_unit="s",
)

FRAME_CRTBP = ReferenceFrame(
    name="CRTBP Normalized",
    origin="Barycentre des deux primaires",
    is_inertial=False,
    length_unit="LU",
    time_unit="TU",
)

FRAME_ECI = ReferenceFrame(
    name="Earth-Centered Inertial J2000.0",
    origin="Centre de la Terre",
    is_inertial=True,
    length_unit="m",
    time_unit="s",
)
