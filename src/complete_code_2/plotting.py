# plotting.py
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from constants import x_earth, x_sun
from l2 import compute_L2

x_L2 = compute_L2()


# plotting.py additions (supplement)
def plot_trajectory_3d(traj, show_primaries=True):
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=2, label="Trajectoire")

    ax.scatter(x[0], y[0], z[0], s=40, marker="x", label="Début")  # type: ignore
    ax.scatter([x_L2], [0], [0], s=80, marker="*", label="L2")  # type: ignore
    if show_primaries:
        ax.scatter([x_earth], [0], [0], s=60, marker="o", label="Earth")  # type: ignore
        ax.scatter([x_sun], [0], [0], s=40, marker="o", label="Sun")  # type: ignore
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")  # type: ignore

    margin_neg = x_earth - 1e6
    margin_pos = x_L2 + 1e6

    ax.set_xlim(margin_neg, margin_pos)
    ax.set_ylim(-1e5, 1e5)

    ax.set_zlim(-1e5, 1e5)  # type: ignore

    ax.legend()
    plt.show()


def plot_projections(traj):
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]

    plt.figure()
    plt.plot(x, y, linewidth=0.8)
    if x_L2 is not None:
        plt.scatter([x_L2], [0], s=30, marker="*")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("Projection XY")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(x, z, linewidth=0.8)
    if x_L2 is not None:
        plt.scatter([x_L2], [0], s=30, marker="*")
    plt.xlabel("x (km)")
    plt.ylabel("z (km)")
    plt.title("Projection XZ")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(y, z, linewidth=0.8)
    plt.xlabel("y (km)")
    plt.ylabel("z (km)")
    plt.title("Projection YZ")
    plt.grid(True)
    plt.show()
