import matplotlib.pyplot as plt
from numpy import log10
from constants import x_earth, x_sun, x_moon
from therocal_poisitions import compute_theorical_lagrangian_points
from acceleration import compute_acceleration_norm, compute_acceleration_vect
from moonpos import plot_moon_phase_effects


### Plot ###
def main(type):
    x_L1, x_L2, x_L3, (x_L4, y_L4), (x_L5, y_L5) = compute_theorical_lagrangian_points()

    match type:
        case "normal":
            X, Y, a_norm = compute_acceleration_norm(500, 2e11)
            a_total_vec = compute_acceleration_vect(X, Y)

            graphical_representation(
                X, Y, a_norm, x_L1, x_L2, x_L3, x_L4, y_L4, x_L5, y_L5, a_total_vec
            )

        case "moon_phases":
            plot_moon_phase_effects(500, 2e9)


def graphical_representation(
    X, Y, a_norm, x_L1, x_L2, x_L3, x_L4, y_L4, x_L5, y_L5, a_total_vec
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    plt.contourf(X, Y, log10(a_norm), levels=1000, cmap="coolwarm")
    plt.colorbar(label="log10(|acceleration|) [m/s²]")

    ## Placing Sun and Earth
    plt.scatter([x_earth, x_moon], [0, 0], color=["blue", "green"], s=80)
    # plt.text(x_sun, 0, "Sun", color="yellow", ha="right")
    plt.text(x_earth, 0, "Earth", color="blue", ha="right")
    plt.text(x_moon, 0, "Moon", color="green", ha="right")

    ## Placing Lagragian points
    plt.scatter([x_L1, x_L2], [0, 0], color="grey", s=40, label="Lagrange Points")

    plt.text(x_L1, 0, "L1", color="grey", ha="center", va="bottom")
    plt.text(x_L2, 0, "L2", color="grey", ha="center", va="bottom")

    plt.title("Net Acceleration Field (Sun-Earth-Moon, Rotating Barycentric Frame)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(False)
    plt.tight_layout()
    ax.set_aspect("equal", adjustable="box")

    step = 1

    plt.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        a_total_vec[0, ::step, ::step],
        a_total_vec[1, ::step, ::step],
        color="white",
        scale=5e-3,
        width=0.002,
    )
    plt.show()


if __name__ == "__main__":
    main("normal")
