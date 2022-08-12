# %% Imports

import matplotlib.pyplot as plt
import numpy as np

# %% Set matplotlib to dark mode
plt.style.use('dark_background')

# %% Functions
def calc_mean_arithmetic(x, y) -> float:
    """
    Calculate the mean of the arithmetic mean of x and y.
    """
    # return np.mean([x, y])
    return (x + y) / 2


def calc_mean_harmonic(x, y) -> float:
    """
    Calculate the mean of the harmonic mean of x and y.
    """
    # return statistics.harmonic_mean([x, y])
    return 2 * (x * y) / (x + y)


# %% Initialize variables
precisions = [3, 1, 2]
recalls = [2, 4.5, 2]

# %% Calculate means
scores_mean = np.array(
    [calc_mean_arithmetic(x, y) for x, y in zip(precisions, recalls)]
)
# [2.4, 1.636, 2]
scores_f1 = np.array([calc_mean_harmonic(x, y) for x, y in zip(precisions, recalls)])

# %%
print(
    f"""
Precision: {precisions}
Recall: {recalls}
Arithmetic: {scores_mean}
Harmonic  : {scores_f1}
Difference: {scores_mean - scores_f1}
"""
)

# %% Plot each result as a bar chart
fig, ax = plt.subplots()
ax.bar(precisions, scores_mean, label="Arithmetic")
ax.bar(precisions, scores_f1, label="Harmonic")
ax.legend()
ax.set_xlabel("Precision")
ax.set_ylabel("F1")
ax.set_title("Precision vs. F1")
plt.show()

# %%

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

# Add horizontal grid lines to the plot
ax.grid(True, which="major", axis="y", linestyle="-", color="white", alpha=0.4)

# Set [p, r] as the x-axis
ax.set_xticks([1, 2])
ax.set_xticklabels(["precision", "recall"])

# Set the y-axis to 5
ax.set_ylim(0, 3.5)
# Set the x-axis to 2
ax.set_xlim(0.5, 2.5)

# Plot precisions[0] in first column
ax.plot(1, precisions[0], "bo", label="Precision", markersize=20)
# Plot precisions[1] in second column
ax.plot(2, recalls[0], "ro", label="Recall", markersize=20)

# Plot the arithmetic mean
# Plot a line from (1, precisions[0]) to (2, recalls[0])
ax.plot([1, 2], [precisions[0], recalls[0]], "--", linewidth=2.5, alpha=0.7, zorder=0, color="green")
# Plot a dot in the middle of the line
ax.plot(1.5, scores_mean[0], "o", label="Arithmetic", markersize=10, color="green")
# Plot a vertical line from 1.5 to scores_mean[0]
ax.plot([1.5, 1.5], [0, scores_mean[0]], "--", linewidth=2.5, alpha=0.5, color="green")

# Plot the harmonic mean
# Declare the x, y coordinates for each lines' start and end points
l1_x = [1, 2]
l1_y = [0, recalls[0]]
l2_x = [2, 1]
l2_y = [0, precisions[0]]
# Derive the slope and intercept of the lines
l1_m = (l1_y[1] - l1_y[0]) / (l1_x[1] - l1_x[0])
l1_b = l1_y[0] - l1_m * l1_x[0]
l2_m = (l2_y[1] - l2_y[0]) / (l2_x[1] - l2_x[0])
l2_b = l2_y[0] - l2_m * l2_x[0]
# Derive the intersection point of the lines
int_x = (l2_b - l1_b) / (l1_m - l2_m)
int_y = scores_f1[0] / 2  # int_y = l1_m * int_x + l1_b
# Plot a line from the bottom of the precision bar to the top of the recall bar
ax.plot(l1_x, l1_y, "-", linewidth=2.5, alpha=0.5, zorder=0, color="white")
# Plot a line from the bottom of the precision bar to the top of the recall bar
ax.plot(l2_x, l2_y, "-", linewidth=2.5, alpha=0.5, zorder=0, color="white")
# Plot a dot at the intersection of the two lines
ax.plot(int_x, int_y, "o", label="Harmonic", markersize=10, color="white")
# Plot a vertical line from the intersection point to the x-axis
ax.plot([int_x, int_x], [0, int_y], "--", linewidth=2.5, alpha=0.5, zorder=0, color="white")

# Add a legend
ax.legend()

# %%
# %%
plt.show()
# %%
