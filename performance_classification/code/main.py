# %% Imports

import matplotlib.pyplot as plt
import numpy as np


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
# %% Set matplotlib to dark mode
plt.style.use("dark_background")


# %% Functions
def calc_mean_arithmetic(x, y) -> float:
    """
    Calculate the arithmetic mean of x and y (typically, precision and recall).
    """
    # return np.mean([x, y])
    return (x + y) / 2


def calc_f1_score(x, y) -> float:
    """
    Calculate the harmonic mean of x and y (typically, precision and recall.
    """
    # return 2 * statistics.harmonic_mean([x, y])
    return 2 * (x * y) / (x + y)


def plot_means(precision, recall, ax=None, title: str = "Harmonic and Arithmetic Means"):
    if not ax:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    # Add a title
    ax.set_title(title)
    arithmetic = calc_mean_arithmetic(precision, recall)
    f1 = calc_f1_score(precision, recall)
    harmonic = f1 / 2

    # Add horizontal grid lines to the plot
    ax.grid(True, which="major", axis="y", linestyle="-", alpha=0.4)

    # Set [p, r] as the x-axis
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["precision", "recall"])

    # Set the y-axis to 5
    ax.set_ylim(0, 5)
    # Set the x-axis to 2
    ax.set_xlim(0.5, 2.5)

    # Plot precisions[0] in first column
    ax.plot(1, precision, "o", label="Precision", markersize=10, color=COLORS[0])
    # Plot precisions[1] in second column
    ax.plot(2, recall, "o", label="Recall", markersize=10, color=COLORS[1])

    # Plot the arithmetic mean
    # Plot a line from the precision marker to the recall marker
    ax.plot(
        [1, 2],
        [precision, recall],
        "--",
        linewidth=2.5,
        alpha=0.7,
        zorder=0,
        color=COLORS[2],
    )
    # Plot a dot in the middle of the line
    ax.plot(1.5, arithmetic, "*", label="Arithmetic", markersize=10, color=COLORS[2])
    # Plot a vertical line from 1.5 to arithmetic
    ax.plot([1.5, 1.5], [0, arithmetic], "--", linewidth=2.5, alpha=0.5, zorder=0, color=COLORS[2])

    # Plot the harmonic mean
    # Declare the x, y coordinates for each lines' start and end points
    l1_x = [1, 2]
    l1_y = [0, recall]
    l2_x = [2, 1]
    l2_y = [0, precision]
    # Derive the slope (m) and intercept (b) of the lines
    l1_m = (l1_y[1] - l1_y[0]) / (l1_x[1] - l1_x[0])
    l1_b = l1_y[0] - l1_m * l1_x[0]
    l2_m = (l2_y[1] - l2_y[0]) / (l2_x[1] - l2_x[0])
    l2_b = l2_y[0] - l2_m * l2_x[0]
    # Derive the intersection point of the lines
    int_x = (l2_b - l1_b) / (l1_m - l2_m)
    int_y = f1 / 2  # int_y = l1_m * int_x + l1_b
    # Plot a line from the bottom of the precision bar to the top of the recall bar
    ax.plot(l1_x, l1_y, "-", linewidth=2.5, alpha=0.5, zorder=0, color=COLORS[3])
    # Plot a line from the bottom of the precision bar to the top of the recall bar
    ax.plot(l2_x, l2_y, "-", linewidth=2.5, alpha=0.5, zorder=0, color=COLORS[3])
    # Plot a dot at the intersection of the two lines
    ax.plot(int_x, int_y, "*", label="Harmonic", markersize=10, color=COLORS[3])
    ax.plot(int_x, int_y * 2, "*", label="F1", markersize=10, color=COLORS[6])
    # Plot a vertical line from the x-axis to the intersection point
    ax.plot(  # Intersection point
        [int_x, int_x],
        [0, int_y],
        "--",
        linewidth=2.5,
        alpha=0.5,
        zorder=0,
        color=COLORS[3],
    )
    # Plot a vertical line from the x-axis to the F1-score
    ax.plot(  # F1-score
        [int_x, int_x],
        [int_y, int_y * 2],
        "--",
        linewidth=2.5,
        alpha=0.5,
        zorder=0,
    )
    # Add annotations to the arithmetic mean, harmonic mean, and F1-score
    ax.annotate(
        f"{arithmetic:.2f}",
        xy=(1.5, arithmetic),
        xytext=(1.5 + 0.1, arithmetic + 0.1),
        ha="center",
        size=14,
        color=COLORS[2],
    )
    ax.annotate(
        f"{harmonic:.2f}",
        xy=(int_x, int_y),
        xytext=(int_x + 0.15, int_y - 0.075),
        ha="center",
        size=14,
        color=COLORS[3],
    )
    if f1 != arithmetic:
        ax.annotate(
            f"{f1:.2f}",
            xy=(int_x, int_y * 2),
            xytext=(int_x + 0.15, int_y * 2),
            ha="center",
            size=14,
            color=COLORS[6],
        )
    # Add a legend
    ax.legend()



# %% Initialize variables
precisions = [3, 1, 2]
recalls = [2, 4.5, 2]

# %% Calculate means
scores_mean = np.array(
    [calc_mean_arithmetic(x, y) for x, y in zip(precisions, recalls)]
)
# [2.4, 1.636, 2]
scores_f1 = np.array([calc_f1_score(x, y) for x, y in zip(precisions, recalls)])

# %%
print(
    f"""
Precision : {precisions}
Recall    : {recalls}
Arithmetic: {scores_mean}
Harmonic  : {scores_f1}
Difference: {scores_mean - scores_f1}
"""
)

# %%
for precision, recall in zip(precisions, recalls):
    plot_means(precision, recall)

# %%
