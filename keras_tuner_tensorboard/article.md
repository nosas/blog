<title>Hyperparameter optimization: KerasTuner & TensorBoard</title>

# Hyperparameter optimization

Finding the optimal model architecture and training configuration is a tedious and time-consuming task.
The manual process of repeatedly tuning a model's hyperparameters and training configuration often leads to sub-optimal model performance.

*Hyperparameters* are values that are used to control the model's learning process during training.
Their values determine the model's performance - specifically, the model's ability to correctly map the input data to the desired labels or targets.
The more optimal the hyperparameters, the better the model's performance.

In deep learning models, the most common hyperparameters are the number of hidden layers, the number of neurons in each layer, and the activation function used in each layer.

<details>
<summary>Common hyperparameters</summary>

- Train-validation-test split ratio
- Optimizer algorithm (e.g., gradient descent, stochastic gradient descent, or Adam optimizer)
- Optimizer's learning-rate
- Convolutional layer's kernel or filter size
- Activation function in a neural network layer (e.g. Sigmoid, ReLU, Tanh)
- Number of hidden layers
- Number of activation units in each layer
- Dropout rate
- Pooling size
- Batch size
- Number of iterations (epochs) during training
- Number of clusters in a clustering task

</details>

We can use [KerasTuner](https://keras.io/keras_tuner/) to automate the process of hyperparameter optimization.
[TensorBoard](https://www.tensorflow.org/tensorboard/) visualizer can be used alongside KerasTuner to visualize the optimization progress.

This article will cover the basics of hyperparameter optimization in deep learning projects using KerasTuner and TensorBoard.
The examples will be based on my own [ToonVision](../toonvision/classification) computer vision project.


<details>
    <summary>Table of Contents</summary>

- [Hyperparameter optimization](#hyperparameter-optimization)
    - [Project description](#project-description)
        - [Model architecture](#model-architecture)
        - [Hyperparameters](#hyperparameters)
    - [KerasTuner](#kerastuner)
        - [Create a model-building function and search space](#create-a-model-building-function-and-search-space)
            - [Search space considerations](#search-space-considerations)
        - [Define a tuner instance](#define-a-tuner-instance)
        - [Launch the tuning process](#launch-the-tuning-process)

</details>

---
## Project description

The ToonVision project is a multiclass classification model for classifying [Cogs](https://toontownrewritten.fandom.com/wiki/Cogs) in ToonTown Online.
There are four unique Cog types - also called [corporate ladders](https://toontownrewritten.fandom.com/wiki/Corporate_ladder) or suits.
Our goal is to train a model that can classify Cogs into the four unique suits, as seen in the image below.

<figure class="center">
    <img src="img/unique_cogs.png" style="width:100%;"/>
    <figcaption>Unique Cog types: Bossbot, Lawbot, Cashbot, Sellbot</figcaption>
</figure>

### Model architecture

We'll create a model from scratch and use my [ToonVision dataset](../toonvision/classification/#the-toonvision-dataset) to train and evaluate the model.

The model will be a convolutional neural network (CNN).
It will have two "blocks", each of which contains a single convolutional layer, two max pooling layers, and a dropout layer.
The final layer will be a fully-connected layer (Dense) with four output nodes, one for each of the four Cog types.

```python
def make_multiclass_model(name: str = "", dropout: float = 0.0) -> keras.Model:
    inputs = keras.Input(shape=(600, 200, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = keras.layers.RandomFlip("horizontal")(x)

    # Block 1: Conv2d -> MaxPool2D -> MaxPool2D -> Dropout
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)
    # Block 2: Conv2D -> MaxPool2D -> MaxPool2D -> Dropout
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(units=4, activation="softmax")(x)
    model = keras.Model(name=name, inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
```

### Hyperparameters

The model's hyperparameters were chosen by intuition and experimentation.
However, I believe that we can find better hyperparameters by tuning the model's hyperparameters using KerasTuner.

We'll focus on tuning the following hyperparameters with KerasTuner:

- `filters`: The number of convolutional filters in each convolutional layer.
- `kernel_size`: The size of the convolutional kernel.
- `pool_size`: The size of the max pooling layers.
- `dropout_rate`: The probability of dropping a neuron.

```python
x = layers.Conv2D(filters, kernel_size, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size)(x)
x = layers.MaxPooling2D(pool_size)(x)
x = layers.Dropout(rate)(x)
```

Additional hyperparameter tunings could include the number of layers (convolutional/pooling/dropout), optimizer algorithm, and learning rate, but I will not cover these here.

Before we start tuning the hyperparameters, let's discuss what KerasTuner does and how it helps ML engineers.

---
## KerasTuner

KerasTuner is a general-purpose hyperparameter tuning library.
The library is well integrated with Keras, allowing for hyperparameter tuning with minimal code changes.
It truly is a powerful, yet simple, library.

We can begin tuning with three easy steps:

1. Create a function that returns a Keras model with the desired hyperparameter search space
2. Define a KerasTuner optimization instance (tuner) of `Hyperband`, `BayesianOptimization`, or `RandomSearch`
3. Launch the tuning process

Pretty simple, right?
Let's take a look at how we can implement the above steps.

### Create a model-building function and search space

Defining a search space is as simple as replacing the layers' hyperparameter values with KerasTuner's search space methods: `hp.Int`, `hp.Float`, `hp.Choice`, etc.
More details about the KerasTuner search space methods can be found [here](https://keras.io/api/keras_tuner/hyperparameters/).

For instance, the follow code block defines a search space for the number of convolutional filters in some convolutional layer.
When launched, the tuner searches for the most optimal filter count by varying the number of filters in the layer from 4 to 16 and training the model.

```python
model = keras.Sequential(
    [
        keras.layers.Conv2D(
            filters=hp.Int("conv_1_filters", min_value=4, max_value=16, step=4),
            kernel_size=5,
            activation="relu",
            padding="same",
        ),
    ])
```

What was once a tedious, manual task is now simple and powerful process for ML engineers.

The following code block is our model-building function with defined search spaces.
Recall that we're searching for the most optimal filter count, kernel size, pooling sizes, and dropout rate.

Note the use of `hp.Int`, `hp.Float`, and `hp.Choice` methods in each layer.
Each of these methods defines a search space for the corresponding hyperparameter.
Integers and floats are used for discrete search spaces (minimum and maximum values with steps), while choices are used for categorical search spaces.

```python
def model_builder(hp):
    model = keras.Sequential(
        [
            # Input and augmentation layers
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.RandomFlip("horizontal"),

            # Block 1: Conv2D -> MaxPool2D -> MaxPool2D -> Dropout
            keras.layers.Conv2D(
                filters=hp.Int("conv_1_filters", min_value=4, max_value=16, step=4),
                kernel_size=hp.Choice("conv_1_kernel_size", values=[3, 5]),
                activation="relu",
                padding="same",
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_1_size", min_value=2, max_value=4, step=1),
            ),
            # Min value == 1 will void the second pooling layer
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_2_size", min_value=1, max_value=4, step=1),
            ),
            keras.layers.Dropout(
                rate=hp.Float("dropout_1_rate", min_value=0.0, max_value=0.9, step=0.1),
            ),
            ...  # Repeat for Block 2 (omitted for brevity)

            # Output layer
            keras.layers.Flatten(),
            keras.layers.Dense(units=4, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
```

#### Search space considerations

Selecting the correct methods and values for the search space is critical to the success of the tuning process.
We do not want such a large search space that the tuner takes too much time and resources.
However, we also do not want such a small search space that the tuner does not find any optimal hyperparameters.

Rather, we must consider meaningful values for each hyperparameter.
This is where intuition, experimentation, and domain expertise comes in to help us define the search space.

For my model, I knew that the number of convolutional filters should remain low (4 to 16).
This choice was made in part because I wanted to avoid overfitting to the validation data during training.
However, I also knew from experience that the more filters I have, the lower my model's generalization performance.

Furthermore, I selected two MaxPooling2D layers for each block because I knew the main differentiation between classes is the Cog's suit color.
My intuition says that more pooling is better, but I'm putting it to the test by defining a search space that also evaluates only a single MaxPooling2D layer.
This is how domain expertise - knowing your data's characteristics - helps us define meaningful search space.

### Define a tuner instance

KerasTuner contains multiple tuners: `RandomSearch`, `BayesianOptimization`, and `Hyperband`.
Each has their own unique tuning algorithm, but all of them share the same search space defined above.
Here are the three tuners along with their respective algorithms:

- `kerastuner.tuners.randomsearch.RandomSearch`: An inefficient, random search algorithm.
- `kerastuner.tuners.bayesian.BayesianOptimization`: A Bayesian optimization algorithm that follows a probabilistic search approach by taking previous results into account.
- `kerastuner.tuners.hyperband.Hyperband`: An optimized variant of the `RandomSearch` algorithm in terms of time and resource usage.

More details above each tuner can be found in [this article](https://neptune.ai/blog/hyperband-and-bohb-understanding-state-of-the-art-hyperparameter-optimization-algorithms).
Additionally, refer to the [KerasTuner documentation](https://keras.io/api/keras_tuner/tuners/) for API details.

My preferred tuning method is to first perform a `RandomSearch` with a large number of trials (100).
Each trial samples a random set of hyperparameter values from the search space.
The goal is to find the best hyperparameter values that minimizes (or maximizes) the objective - in our case, the goal is minimizing the validation loss.

```python
tuner = RandomSearch(
    hypermodel=model_builder,
    objective="val_loss",
    max_trials=100,
    executions_per_trial=1,  # Increase to reduce variance of the results
    directory="models",
    project_name="tuned_multiclass_randomsearch",
    seed=42,
)
```

`RandomSearch` is the least efficient algorithm, but it provides useful insight into the general whereabouts of optimal hyperparameter values.
These insights can be used to further constrain and reduce the search space for more effective tuning.

Following the random search, I'll review the highest performing parameters in TensorBoard, tighten my search space, and then launch a more efficient `Hyperband` or `BayesianOptimization` search.
Let's launch a `RandomSearch` and review the results.

### Launch the tuning process
