# %% Import libraries
from io import BytesIO

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
    alpha=0.50,
    s=100,
    label="Class A",
    edgecolors="none",
)
plt.scatter(
    class_b[:, 0],
    class_b[:, 1],
    c="blue",
    alpha=0.50,
    s=100,
    label="Class B",
    edgecolors="none",
)
plt.legend()
plt.savefig("../img/linear_classifier_data.png", transparent=False)
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
def square_loss(predictions, targets) -> float:
    # Calculate the loss per sample, results in tensor of shape (len(targets), 1)
    per_sample_loss = tf.square(targets - predictions)
    # Average the per-sample loss and return a single scalar loss value
    return tf.reduce_mean(per_sample_loss)


# %% Create the training step function
def training_step(inputs, targets, learning_rate: float = 0.1) -> float:
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(predictions, targets)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(learning_rate * grad_loss_wrt_W)
    b.assign_sub(learning_rate * grad_loss_wrt_b)
    return loss


# %% Create the batch training loop
num_epochs = 50
loss_all = []
predictions_all = []
parameters_all = []

for step in range(num_epochs):
    step_loss = training_step(inputs, labels)
    loss_all.append(step_loss)
    if step % 5 == 0:
        print(f"Step {step}: Loss = {step_loss:.4f}")
    # Save every prediction
    predictions_all.append(model(inputs))
    parameters_all.append((W.numpy(), b.numpy()))

# %% Retrieve the model's final predictions
predictions = predictions_all[-1]

# %% Plot the loss over time
plt.clf()
plt.plot(loss_all[1:41])  # Skip the first value, since it's just the initialization
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("../img/loss_exclude_initial_and_tail.png", transparent=False)


# %% Function to plot the two classes
def plot_prediction(
    prediction,
    inputs,
    parameters: tuple[np.ndarray, float] = None,
    legend: bool = False,
):
    plt.scatter(
        inputs[:num_samples_per_class, 0],
        inputs[:num_samples_per_class, 1],
        c=[
            "green" if pred else "blue"
            for pred in prediction[:num_samples_per_class, 0] < 0.5
        ],
        label="Class A",
    )
    plt.scatter(
        inputs[num_samples_per_class:, 0],
        inputs[num_samples_per_class:, 1],
        c=[
            "green" if pred else "blue"
            for pred in prediction[num_samples_per_class:, 0] < 0.5
        ],
        label="Class B",
    )

    if parameters:
        W, b = parameters
        x = np.linspace(-1, 5, 100)
        y = -W[0] / W[1] * x + (0.5 - b) / W[1]
        plt.plot(x, y, c="red")
        plt.xlim(-3, 6)
        plt.ylim(-3, 6)
    if legend:
        plt.legend(["Class A", "Class B"])


# %% Plot the line separating the two classes based on the model's final predictions
# The line is represented by the equation y = -W[0]/W[1] * x - (b/W[1])
plot_prediction(predictions, inputs, parameters=(W, b), legend=True)

# %% Plot the model's first ten predictions
plt.figure(0)
for i, prediction in enumerate(predictions_all[:10]):

    ax = plt.subplot(2, 5, i + 1)
    plt.scatter(
        class_a[:, 0],
        class_a[:, 1],
        c=[
            "green" if pred else "blue"
            for pred in prediction[:num_samples_per_class, 0] < 0.5
        ],
        alpha=0.25,
        s=100,
        label="Class A",
        edgecolors="none",
    )
    plt.scatter(
        class_b[:, 0],
        class_b[:, 1],
        c=[
            "green" if pred else "blue"
            for pred in prediction[num_samples_per_class:, 0] < 0.5
        ],
        alpha=0.25,
        s=100,
        label="Class B",
        edgecolors="none",
    )

plt.show()


# %% Scatter plot for the model's predictions where the dots are green if the prediction is accurate, red if the prediction is incorrect
def plot_prediction_acc(
    prediction: np.ndarray,
    inputs: np.ndarray,
    buffer=None,
    parameters: tuple[np.ndarray, float] = None,
    savename: str = "",
    title: str = "",
):
    plt.scatter(
        inputs[:, 0],
        inputs[:, 1],
        c=[
            "green" if labels[idx] == pred else "red"
            for idx, pred in enumerate(prediction > 0.5)
        ],
        alpha=0.20,
        s=100,
    )
    # Draw the red line separating the two classes
    if parameters:
        W, b = parameters
        x = np.linspace(-1, 5, 100)
        y = -W[0] / W[1] * x + (0.5 - b) / W[1]
        plt.plot(x, y, c="red")
        plt.xlim(-3, 6)
        plt.ylim(-3, 6)
    # Add a title to the plot
    if title:
        plt.title(title)
    # Save the plot to a file
    if savename:
        plt.savefig(f"../img/{savename}.png", transparent=False)
    # Save the plot to a buffer
    if buffer:
        plt.savefig(buffer, format="png", transparent=False)
    else:
        plt.show()
    plt.close()


# %% Create function for making gif
def make_gif(predictions: np.ndarray, inputs: np.ndarray, savename: str):
    fig, ax = plt.subplots()

    with imageio.get_writer(f"../img/{savename}.gif", mode="I") as writer:
        for prediction_idx, prediction in enumerate(predictions):
            parameters = parameters_all[prediction_idx]
            buffer = BytesIO()
            plot_prediction_acc(
                prediction=prediction,
                inputs=inputs,
                buffer=buffer,
                parameters=parameters,
                title=f"Prediction {prediction_idx}",
            )
            buffer.seek(0)
            img = plt.imread(buffer, format="png")
            writer.append_data(img)

    plt.show()


def make_gif_with_duration(
    predictions: np.ndarray, inputs: np.ndarray, savename: str, duration: float
):
    images = []
    for prediction_idx, prediction in enumerate(predictions):
        params = parameters_all[prediction_idx]
        buffer = BytesIO()
        plot_prediction_acc(
            prediction=prediction,
            inputs=inputs,
            buffer=buffer,
            parameters=params,
            title=f"Prediction {prediction_idx}",
        )
        buffer.seek(0)
        images.append(plt.imread(buffer, format="png"))
    imageio.mimsave(f"../img/{savename}.gif", images, duration=duration)


# %% Make gif
make_gif(predictions_all, inputs, "prediction_accuracy")

# %% Train the model for another 50 epochs
for step in range(num_epochs):
    step_loss = training_step(inputs, labels)
    loss_all.append(step_loss)
    if step % 5 == 0:
        print(f"Step {step}: Loss = {step_loss:.4f}")
    # Save every prediction
    predictions_all.append(model(inputs))
    parameters_all.append((W.numpy(), b.numpy()))

# %% Plot the loss over time
plt.clf()
plt.plot(loss_all[1:])  # Skip the first value, since it's just the initialization
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# %% Retrieve the model's predictions
predictions = model(inputs)

# %% Plot the line separating the two classes
plot_prediction(predictions, inputs, parameters=(W, b))

# %% Scatter plot for the model's last prediction where the dots are green if the prediction is accurate, red if the prediction is incorrect
plot_prediction_acc(predictions, inputs)

# %% Plot all predictions as a gif
make_gif(predictions_all, inputs, "prediction_accuracy_100")

# %% Make gif of first 20 predictions with 1s duration
make_gif_with_duration(predictions_all[:20], inputs, "prediction_accuracy_slowed", 0.5)

# %% Plot the model's classes with params
plot_prediction(predictions, inputs, parameters=(W, b), legend=True)
