# plotting.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# plotting.py additions (supplement)
def plot_trajectory_3d(times, traj, x_L2=None, show_primaries=True):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=0.8, label="Trajectoire")
    if x_L2 is not None:
        ax.scatter([x_L2], [0], [0], s=80, marker="*", label="L2")  # type: ignore
    if show_primaries:
        from constants import x_earth, x_sun

        ax.scatter([x_earth], [0], [0], s=60, marker="o", label="Earth")  # type: ignore
        ax.scatter([x_sun], [0], [0], s=40, marker="o", label="Sun")  # type: ignore
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")  # type: ignore
    ax.legend()
    plt.show()


def plot_projections(times, traj, x_L2=None):
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


def plot_dv_log(log):
    if len(log) == 0:
        print("Aucun cycle Δv enregistré.")
        return

    dvs = [abs(entry["dv"]) for entry in log]
    cum = np.cumsum(dvs)

    plt.figure()
    plt.step(range(1, len(dvs) + 1), cum, where="mid")
    plt.xlabel("Cycle")
    plt.ylabel("Δv cumulé (km/s)")
    plt.title("Budget de Δv au fil des corrections")
    plt.grid(True)
    plt.show()
