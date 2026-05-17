import numpy as np
import numpy.typing as npt

StateVector = npt.NDArray[np.float64]
PositionVector = npt.NDArray[np.float64]
VelocityVector = npt.NDArray[np.float64]
RotationMatrix = npt.NDArray[np.float64]


def create_state_vector() -> StateVector:
    return np.zeros(6, dtype=np.float64)


def create_postion_vector() -> PositionVector:
    return np.zeros(3, dtype=np.float64)


def create_velocity_vector() -> VelocityVector:
    return np.zeros(3, dtype=np.float64)
