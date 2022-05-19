# %% Import libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# %% Create 500 cleanly separated data points for each class
num_samples_per_class = 500

class_a_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)
class_b_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)

# %% Create a stacked 2D tensor of the data points
# The first 500 samples are from class A, the next 500 samples are from class B
inputs = np.vstack((class_a_samples, class_b_samples)).astype(np.float32)

# %% Generate labels for the data points
# The first 500 labels are 0 (class A), and the next 500 are 1 (class B)
labels = np.vstack(
    (
        np.zeros((num_samples_per_class, 1), dtype=np.float32),
        np.ones((num_samples_per_class, 1), dtype=np.float32),
    )
)

# %% Plot the two classes
# Class A is represented by green dots, and class B is represented by blue dots,
plt.clf()
class_a = inputs[:num_samples_per_class]
class_b = inputs[num_samples_per_class:]
plt.scatter(
    class_a[:, 0],
    class_a[:, 1],
    c="green",
    alpha=0.25,
    s=100,
    label="Class A",
    edgecolors="none",
)
plt.scatter(
    class_b[:, 0],
    class_b[:, 1],
    c="blue",
    alpha=0.25,
    s=100,
    label="Class B",
    edgecolors="none",
)
plt.savefig("../images/linear_classifier_data.png", transparent=True)
plt.legend()
plt.show()

# %% Initialize the weights and biases
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


# %% Create the forward pass function: prediction = matmul(inputs, W) + b
def model(inputs: np.ndarray) -> np.ndarray:
    return tf.matmul(inputs, W) + b


# %% Create the mean squared error loss function: loss = mean(square(prediction - labels))
def square_loss(targets, predictions) -> float:
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


# %% Create the training step function
def training_step(inputs, targets, learning_rate: float = 0.1) -> float:
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(learning_rate * grad_loss_wrt_W)
    b.assign_sub(learning_rate * grad_loss_wrt_b)
    return loss


# %% Create the batch training loop
num_epochs = 50
loss_all = []
predictions_all = []
predictions_first_10 = []

for step in range(num_epochs):
    step_loss = training_step(inputs, labels)
    loss_all.append(step_loss)
    if step < 10:
        predictions_first_10.append(model(inputs))
    if step % 5 == 0:
        print(f"Step {step}: Loss = {step_loss}")
        # Save every 5th prediction
        predictions_all.append(model(inputs))

# Retrieve the model's final predictions
predictions = model(inputs)

# %% Plot the loss over time
plt.clf()
plt.plot(loss_all[1:])  # Skip the first value, since it's just the initialization
plt.xlabel("Epoch")
plt.ylabel("Loss")


# %% Plot the line separating the two classes based on the model's final predictions
# The line is represented by the equation y = -W[0]/W[1] * x - (b/W[1])
x = np.linspace(-2, 6, 100)
y = -W[0] / W[1] * x - (0.5 - b) / W[1]
plt.plot(x, y, c="red")
plt.scatter(
    inputs[:, 0],
    inputs[:, 1],
    c=["green" if pred else "blue" for pred in predictions[:, 0] < 0.5],
)

# %% Plot the line for predictions in predictions_all, each prediction is a subplot
plt.figure(0)
for i, prediction in enumerate(predictions_first_10):
    ax = plt.subplot(2, 5, i + 1)
    plt.scatter(
        class_a[:, 0],
        class_a[:, 1],
        c=["green" if pred else "blue" for pred in prediction[:num_samples_per_class, 0] < 0.5],
        alpha=0.25,
        s=100,
        label="Class A",
        edgecolors="none",
    )
    plt.scatter(
        class_b[:, 0],
        class_b[:, 1],
        c=["green" if pred else "blue" for pred in prediction[num_samples_per_class:, 0] < 0.5],
        alpha=0.25,
        s=100,
        label="Class B",
        edgecolors="none",
    )
    # ax.scatter(
    #     inputs[:, 0],
    #     inputs[:, 1],
    #     c=["green" if pred else "blue" for pred in prediction[:, 0] < 0.5],
    # )
    # ax.plot(x, y, c="red")
plt.legend()
plt.show()

# %% Scatter plot for the model's first prediction where the dots are green if the prediction is accurate, red if the prediction is incorrect
plt.figure(1)
plt.scatter(
    inputs[:, 0],
    inputs[:, 1],
    c=["green" if labels[idx] == pred else "red" for idx, pred in enumerate(predictions_first_10[0] > 0.5)],
)
plt.show()

# %% Train the model for another 50 epochs
# loss_all = []
for step in range(num_epochs):
    step_loss = training_step(inputs, labels)
    loss_all.append(step_loss)
    if step % 5 == 0:
        print(f"Step {step}: Loss = {step_loss}")

# %% Plot the loss over time
plt.clf()
plt.plot(loss_all[num_epochs:])  # Skip the first value, since it's just the initialization
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# %% Retrieve the model's predictions
predictions = model(inputs)

# %% Plot the line separating the two classes
y = -W[0] / W[1] * x - (0.5 - b) / W[1]
plt.plot(x, y, c="red")
plt.scatter(
    inputs[:, 0],
    inputs[:, 1],
    c=["green" if pred else "blue" for pred in predictions[:, 0] < 0.5],
)

# %% Scatter plot for the model's last prediction where the dots are green if the prediction is accurate, red if the prediction is incorrect
plt.figure(1)

plt.scatter(
    inputs[:, 0],
    inputs[:, 1],
    c=["green" if labels[idx] == pred else "red" for idx, pred in enumerate(predictions > 0.5)],
)
plt.show()

# %%
