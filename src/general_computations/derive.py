from acceleration import (
    compute_acceleration_vect,
    compute_grav_accel_vec,
    compute_acceleration_norm,
)
from constants import *
from therocal_poisitions import compute_theorical_lagrangian_points


def main(type):
    x_L1, x_L2, x_L3, (x_L4, y_L4), (x_L5, y_L5) = compute_theorical_lagrangian_points()

    X, Y, a_norm = compute_acceleration_norm(1000, 2e11)
    a_total_vec = compute_acceleration_vect(X, Y)
