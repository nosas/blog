import math
from typing import Callable

import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.001
OPTIMIZER = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)


# Naive Dense layer class
class NaiveDense:
    def __init__(self, input_size: int, output_size: int, activation: Callable):
        self.activation = activation

        # Create a weight matrix of shape (input_size, output_size) intialized with random values
        weights_shape = (input_size, output_size)
        weights_initial = tf.Variable(
            tf.random.uniform(shape=weights_shape, minval=0, maxval=1e-1)
        )
        self.weights = tf.Variable(initial_value=weights_initial)

        # Create a zero-filled vector of biases with the same shape as the output_size
        bias_shape = (output_size,)
        bias_initial = tf.zeros(shape=bias_shape)
        self.bias = tf.Variable(initial_value=bias_initial)

    def __call__(self, inputs):
        # Apply the forward pass and compute the output of the layer
        return self.activation(tf.matmul(inputs, self.weights) + self.bias)

    @property
    # Convenient method for retrieving the layer's weights and biases
    def params(self):
        return [self.weights, self.bias]


# Naive Sequential model class
class NaiveSequential:
    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

    @property
    # Convenient method for retrieving the model's weights and biases
    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params
        return params


# Iterate over the MNIST dataset in mini-batches using the BatchGenerator class
class BatchGenerator:
    def __init__(self, images: np.ndarray, labels: np.ndarray, batch_size: int = 128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(images.shape[0] / batch_size)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        # Get a batch from the dataset based on the batch size and index
        if self.index < self.num_batches * self.batch_size:
            images = self.images[self.index : self.index + self.batch_size]
            labels = self.labels[self.index : self.index + self.batch_size]
            self.index += self.batch_size
            # Return the images and labels corresponding to the batch indices
            return images, labels
        raise StopIteration

    def reset(self):
        self.index = 0


# Define the optimizer and apply it to the model's weights
def update_weights(gradients, params, use_optimizer: bool = False):
    if not use_optimizer:
        for param, gradient in zip(params, gradients):
            param.assign_sub(LEARNING_RATE * gradient)
    else:
        OPTIMIZER.apply_gradients(zip(gradients, params))


# Define the training step function that computes the loss and updates the model's weights
def one_training_step(
    model: NaiveSequential,
    batch_images: np.ndarray,
    batch_labels: np.ndarray,
    use_optimizer: bool,
) -> float:
    with tf.GradientTape() as tape:
        predictions = model(batch_images)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            batch_labels, predictions
        )
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.params)
    update_weights(gradients, model.params, use_optimizer)
    return average_loss


# Fit function containing a training loop that applies one training step to a model for num_epochs epochs
def fit(
    model: NaiveSequential,
    images: np.ndarray,
    labels: np.ndarray,
    num_epochs: int,
    use_optimizer: bool = False,
) -> None:
    for epoch in range(num_epochs):
        for batch in BatchGenerator(images, labels):
            batch_images, batch_labels = batch
            loss = one_training_step(model, batch_images, batch_labels, use_optimizer)
        print(f"Epoch {epoch + 1} loss: {loss}")


# Preproccess the image data to be normalized between [0, 1] and reshaped to be a 2D tensor
def preprocess_images(images: np.ndarray) -> np.ndarray:
    images = images.reshape((images.shape[0], 28 * 28))
    images = images.astype("float32") / 255.0
    return images


# fmt: off
# Load training and test data from mnist dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# Preprocess the training and test data
train_images = preprocess_images(images=train_images)
test_images = preprocess_images(images=test_images)

# fmt: off
# Create a two-layer Sequential model using the NaiveDense layer class
model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
    ])
# Verify the size of the model's weights and biases
assert len(model.params) == 2 * 2

# Create a BatchGenerator object to iterate over the training data in mini-batches
batches = BatchGenerator(train_images, train_labels, batch_size=128)

# Fit the model to the training data
fit(model=model, images=train_images, labels=train_labels, num_epochs=5, use_optimizer=True)

# Evaluate the model on the test data
predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"Accuracy: {np.mean(matches)}")


def image_to_tensor(path_to_img: str) -> np.ndarray:
    img = tf.io.read_file(filename=path_to_img)
    tensor = tf.io.decode_png(img, channels=1)
    tensor = tensor.numpy().astype("float32") / 255.0
    tensor[tensor == 1] = 0
    tensor = tensor.reshape((28 * 28))
    return tensor


path_to_img = "code/images/4.png"
tensor = image_to_tensor(path_to_img)
pred = model([tensor])
pred_np = pred.numpy()
pred_label = pred_np.argmax()
print(pred_label)
