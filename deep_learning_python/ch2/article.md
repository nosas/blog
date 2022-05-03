<title>Deep Learning with Python: Chapter 2 - Mathematical building blocks of neural networks</title>


# Deep Learning with Python

This article is part 2/13 (?) of a series of articles named *Deep Learning with Python*.

In this series, I will read through the second edition of *Deep Learning with Python* by François Chollet.
Articles in this series will sequentially review key concepts, examples, and interesting facts from each chapter of the book.

---
# Chapter 2: The mathematical building blocks of neural networks

This chapter covers...

* A first example of a neural network
* Tensors and tensor operations
* How neural networks learn via backpropagation and gradient descent

Understanding deep learning requires familiarity with many simple mathematical concepts: *tensors*, *tensor operations*, *differentiation*, *gradient descent*, and so on.
This chapter will build on the concepts above without getting overly technical.
The use of precise, unambiguous executable code, instead of mathematical notation, will allow most programmers to easily grasp these concepts.

---
## First look at neural networks

Concrete example of a neural network (NN) with the use of the Python library `Keras` to learn how to classify handwritten digits.

### The problem

The problem we're trying to solve here is to classify grayscale images of handwritten digits (28x28 pixels) into their 10 categories (digits 0 through 9).
This problem is commonly referred to as the "Hello World" of deep learning - it's what you do to verify your algorithms are working as expected.

> **NOTE**: Classification problem keywords
>
> In ML classification problems, a **category** is called a **class**.
> Data points - such as individual train or test images - are called **samples**.
> The class associated with a specific sample is called a **label**.

We'll be using the MNIST dataset: a set of 60,000 training images, plus 10,000 test images, assembled by the National Institute of Standards and Technology (the NIST in MNIST) in the 1980s.

The MNIST dataset is preloaded in `Keras`, in the form of four `NumPy` arrays

```python
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```
Let's take a peek at the shape of the data.
We should see 60,000 training images and labels, 10,000 test images and labels.

```python
>>> train_images.shape
(60000, 28, 28)
>>> len(train_labels)
60000
>>> train_labels
array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
>>>
>>> test_images.shape
(10000, 28, 28)
>>> len(test_labels)
10000
>>> test_labels
array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)
```

Let's look at a sample image using the `matplotlib` library:

```python
import matplotlib.pyplot as plt
digit = train_images[4]
plt.imgshow(digit, cmap=plit.cm.binary)
plt.show()
```

<font style="color:red">TODO: Insert MNIST sample digits</font>

Lastly, let's look at what label corresponds to the previous image:

```python
>>> train_labels[4]
9
```

### Defining the network architecture

The core building block of a neural network is the *layer*.
A layer can be considered as a data filter: data goes in, and comes out more purified - more useful.
Specifically, layers extract *representations* out of the input data.

In deep learning models, simple layers are chains together to form a *data distillation* network.
Deep learning models could be visualized as a sieve for data processing - successive layers refining input data more and more.

The following example is a two-layer neural network.
We aren't expected to know exactly what the example means - we'll learn throughout the next two chapters.

The model consists of a sequence of two `Dense` layers, which are densely connected (also called *fully connected*).
The second layer is a 10-way *softmax classification* layer, which means it will return an array of 10 probability scores (summing to 1).
Each score will be the probability that the current digit image belongs to on our of 10 digit classes.

```python
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

### Preparing the model for training

Before we begin training, we must compile three more things, in addition to the training and testing data, as part of the *compilation* step:
We brushed over the jobs of the loss score and optimizer in the previous chapter.
The specifics of their jobs will be made clear throughout the next two chapters.

1. *An optimizer*: How the model will update itself - its weights - based on the training data it sees, so as a to improve its performance
1. *A loss function*: How the model will measure its performance on the training data and how it will be able to steer itself in the more correct direction
1. *Metrics to monitor during training and testing*: For now, we'll only care about accuracy - the fraction of images that were correctly classified

```python
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```

### Preparing the data

Before training, we'll preprocess the data to ensure consistent data shapes and scales.
We'll reshape the data into the shape the model expects and scale it so that all values are in the [0, 1] interval instead of [0, 255] interval.

The training image data will be transformed from a `uint8 ` array of shape `(60000, 28, 28)` with values between [0, 255] to a `float32` array of shape `(60000, 28*28)` with values between [0, 1].
The same will be done to the testing image data.

```python
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255
```

### "Fitting" (Training) the model

With the data properly pre-processed, we are finally read to train the model!
In Keras, training the model is done via a call to the model's `fit()` method - we *fit* the model to its training data.

```python
>>> model.fit(train_images, train_labels, epochs=5, batch_size=128)
Epoch 1/5
60000/60000 [==========================] - 5s - loss: 0.2524 - acc: 0.9273
Epoch 2/5
51328/60000 [====================>.....] - ETA: 1s - loss: 0.1035 - acc: 0.9692
```

The model swiftly reaches a decent accuracy of 96% after roughly 2 epochs of fitting to the training data.

### Making predictions with the trained model

Now that the model is trained, we can use it to make class predictions on the *new*, unseen data - such as the testing images.

```python
>>> test_digit = test_images[0]
>>> prediction = model.predict(test_digit)
>>> prediction
array([1.0726176e-10, 1.6918376e-10, 6.1314843e-08, 8.4106023e-06,
       2.9967067e-11, 3.0331331e-09, 8.3651971e-14, 9.9999106e-01,
       2.6657624e-08, 3.8127661e-07], dtype=float32)
```

Each index *i* in `predictions[0]` corresponds to the probability that `prediction` belongs to class *i*.
In this example, the highest probability is index 7, meaning the model believes that `test_digit` is the number 7.

We can verify if the model's prediction is correct by comparing the prediction against the test_labels data.

```python
>>> predictions.argmax()  # Return the index of the highest probability
7
>>> predictions[7]
0.99999106
>>> test_labels[0]
7
>>> predictions.argmax() == test_labels[0]
True
```

### Evaluating the model on new data

We can evaluate the model's accuracy against data it has never seen before using the model's `evaluate()` method.
This method will allow us to compute the average accuracy against an entire test set.

```python
>>> test_loss, test_acc = model.evaluate(test_images, test_labels)
>>> print(f"test_acc: {test_acc}")
test_acc: 0.9785
```

This concludes our first example.
We just saw how easy it is to build and train a neural network classification model in less than 15 lines of Python code.

Let's learn more about data representations and how the neural network interprets and refines input data using tensors.

---
## Data representations: Tensors

*Tensors* are fundamental data structures used in machine learning.
At its core, a tensor is a container for data - usually numeric data.
Matrices (2D arrays) are considered to be rank-2 tensors.

Therefore, tensors are generalizations of matrices to an arbitrary number of *dimensions*.
Note that in the context of tensors, a dimension is often called an *axis*.

Let's take a look at definitions and examples of rank-0 to rank-3 and higher tensors.

### Scalars (rank-0 tensors)

A tensor that contains only one number is called a *scalar* - or scalar tensor, rank-0 tensor, or 0D tensor.
Using NumPy's `ndim` attribute, you'll notice a scalar tensor has 0 *axes* (`ndim == 0`).
The number of axes of a tensor is also called its *rank*.

```python
>>> import numpy as np
>>> x = np.array(22)
>>> x
array(12)
>>> x.ndim
0
```

### Vectors (rank-1 tensors)

An array of numbers is called a *vector* - or rank-1 tensor, 1D tensor, tensor of rank 1.
A rank-1 tensor has exactly one axis.

```python
>>> x = np.array([15, 2, 3, 11, 93])
>>> x
array([15, 2, 3, 11, 93])
>>> x.ndim
1
```

The vector above has five entries and so is called a *5-dimensional vector*.
It's important to not confuse a 5D *vector* with a 5D *tensor*.
A 5D vector has a single axis and has five dimensions along its axis.
A 5D tensor - or *tensor of rank 5* -  on the other hand, has five axes and any number of dimensions along each axes.

### Matrices (rank-2 tensors)

An array of vectors is a *matrix* - or rank-2 tensor, 2D tensor, tensor of rank 2.
A matrix has two axes often referred to as *rows* and *columns*.

```python
>>> x = np.array([[4, 8, 15, 16, 23, 42],
                  [24, 2, 61, 51, 8, 3],
                  [44, 3, 52, 62, 9, 9]])
>>> x.ndim
2
```

The entries from the first axis are called the *rows*, and the entries from the second axis are called the *columns*.
`[4, 8, 15, 16, 23, 42]` is the first row of `x`, and `[4, 24, 44]` is the first column.

### Rank-3 and higher-rank tensors

If you insert matrices (rank-2 tensors) into an array, you obtain a rank-3 tensor.
Rank-3 tensors can be visualized as a cube of numbers.

```python
>>> x = np.array([[[4, 18, 15, 6, 23, 22],
                   [5, 32, 61, 1, 28, 23],
                   [6, 33, 52, 2, 29, 29]],
                  [[4, 18, 15, 6, 23, 42],
                   [5, 32, 61, 1, 28, 23],
                   [6, 33, 52, 2, 29, 29]]])
>>> x.ndim
3
```

Inserting rank-3 tensors in an array will create a rank-4 tensor, and so on.
In deep learning, we'll generally only work with rank-0 to rank-4 tensors.
Although, rank-5 tensors may be used if processing video data.

### Key attributes

- *Number of axes (rank)*: For instance, a rank-3 tensor has three axes, and a matrix has two axes. This is also called the tensor's `ndim` in Python libraries such as NumPy or TensorFlow.
- *Shape*: This is a tuple of integers that describes how many dimensions the tensor has along each axis.
For instance, a matrix with shape `(3, 5)` has three rows and five columns.
A vector with a single element could have the shape `(5,)`, whereas a scalar has an empty shape, `()`.
Lastly, a rank-3 tensor, such as the example above, has shape `(2, 3, 5)`.
- *Data type*: Usually called the` dtype` in Python libraries, this is the type of the data contained in the tensor.
For instance, a tensor's type could be `float16`, `float32`, `uint8`, and so on.
It's also possible to come across `string` tensors in TensorFlow.

---
## Real-world examples of data tensors

- *Vector data*: Rank-2 tensors of shape `(samples, features)`, where each sample is a vector of numerical attributes ("features")
- *Timeseries data or sequence data*: Rank-3 tensors of shape `(samples, timesteps, features)`, where each sample is a sequence (of length `timesteps`) of feature vectors
- *Images*: Rank-4 tensors of shape `(samples, height, width, channels)`, where each sample is a 2D grid of pixels, and each pixel is represented by a vector of values ("channels").
- *Video*: Rank-5 tensors of shape `(samples, frames, height, width, channels)`, where each sample is a sequence (of length `frames`) of images

### Vector

This is one of the most common use cases of tensors.
Each data point in a dataset is encoded as a vector.
A batch of data will be encoded as a rank-2 tensor - that is, an array of vectors - where the first axis is the `samples axis` and the second axis is he `features axis`.

Let's look at an example:

- A dataset of cars, where we consider each car's make, model, manufactured year, and odometer reading.
Each car can be characterized as a vector of 4 values.
An entire dataset of 100,000 cars can be stored in a rank-2 tensor of shape `(100000, 4)`.

### Timeseries data or sequence data

Whenever time matters in your data - or the notion of sequential order - it makes sense to store it in a rank-3 tensor with an explicit time axis.
Each sample can be encoded as a sequence of vectors (a rank-2 tensor), and thus a batch of data will be encoded as a rank-3 tensor.

<font style="color:red">TODO: Insert rank-3 timeseries data tensor</font>

By convention, the time axis is always the second axis.
Let's take a look at an example:

- A dataset of a MotoGP rider's lap around Laguna Seca.
Every percentage of lap completed, we store the motorcycle's speed, lean angle, throttle input, brake input, and steering input.
Ideally, it would be as close to realtime as possible instead of every single percentage, but let's keep it simple.
Thus, every lap is encoded as a 5D vector of shape `(101, 5)`, where 101 is 0 percent to 100 percent, inclusive.
An entire race (assuming 30 laps) is encoded as a rank-3 tensor of shape `(30, 101, 5)`.

- A dataset of stock prices.
Every minute, we store the current price of the stock, the highest price in the past minute, and the lowest price in the past minute.
Thus, every minute is encoded as a 3D vector, an entire day of trading is encoded as a matrix of shape `(390, 3)` (there are 390 minutes in a trading day), and 365 days' worth of data can be stored in a rank-3 tensor of shape `(365, 390, 3)`.
Here, each sample would be one day's worth of data.


### Image data

Images usually have three dimensions: height, width, and color channels.
Grayscale images (black-and-white images, like our MNIST images) have only a single color channel.
Colored images typically have three color channels: RGB (red, green, blue).

A batch of 500 grayscale images of size 256x256 could thus be stored in a rank-4 tensor of shape `(500, 256, 256, 1)`, whereas a batch of 500 *colored* images could be stored in a tensor a shape `(500, 256, 256, 3)`.

<font style="color:red">TODO: Insert rank-4 image data tensor</font>

### Video data

Video data is one of the few types of real-world data for which rank-5 tensors are used.
A video can be simplified as a sequence of frames, each frame being a color image.

Each frame can be stores in a rank-3 tensor `(height, width, color_channel)`.
A sequence of frames can be stored in a rank-4 tensor `(frames, height, width, color_channel)`.
Therefore, a batch of videos can be stored in a rank-5 tensor of shape `(samples, frames, height, width, color_channel)`.

For instance, a 20-second, 1920x1080 video clip sampled at 10 frames per second would have 200 frames.
A batch of 5 such video clips would be stored in a tensor of shape `(5, 200, 1920, 1080, 3)`.
That's a total of 6,220,800,000 values!

---
## Tensor operations

Similar to how to computer programs can be reduced to a small set of binary operations (AND, OR, XOR, and so on), all transformations learned by deep neural networks can be reduced to a handful of *tensor operations*.

In our initial example, we built our model by sequentially stacking `Dense` layers.
In Keras, a `Dense` layer with 512 nodes and activation function `relu` looks like this:

```python
keras.layers.Dense(512, activation="relu")
```

### Basic operations

- *Addition*: `t1 + t2`
- *Subtraction*: `t1 - t2`
- *Element-wise multiplication*: `t1 * t2`
- *Element-wise division*: `t1 / t2`
- *Exponentiation*: `t1 ** t2`
- *Modulo*: `t1 % t2`
- *Floor division*: `t1 // t2`
- *Element-wise maximum*: `tf.maximum(t1, t2)`
- *Element-wise minimum*: `tf.minimum(t1, t2)`
- *Element-wise greater than*: `tf.greater(t1, t2)`
- *Element-wise less than*: `tf.less(t1, t2)`
- *Element-wise greater than or equal to*: `tf.greater_equal(t1, t2)`
- *Element-wise less than or equal to*: `tf.less_equal(t1, t2)`
- *Element-wise equality*: `tf.equal(t1, t2)`
- *Element-wise not equal*: `tf.not_equal(t1, t2)`
- *Element-wise logical AND*: `tf.logical_and

### Element-wise operations

### Broadcasting

### Tensor product

### Tensor reshaping

### Geometric interpretations

---
## How neural networks learn via backpropagation and gradient descent

### Backpropagation

Backpropagation is the process of finding the derivative of the loss function with respect to the weights and biases of a neural network.

### Gradient descent

Gradient descent is a common technique for optimizing neural networks.
It is a process of iteratively moving the weights and biases of a neural network towards the minimum of the loss function.

### Stochastic gradient descent

Stochastic gradient descent (SGD) is a variant of gradient descent that is used to train neural networks.
It is a stochastic approach to gradient descent, where the learning rate is not constant, but rather is a function of the iteration number.

### Backpropagation algorithm

How can we get the gradient of the loss with respect to the weights?
Using the *Backpropagation algorithm*.

---
## Recap: Looking back at our first example

We should now have a general understanding of what's going on behind the scenes in a neural network.
What was previously a mysterious black box has turned into a clearer picture seen below: the **model**, composed of sequential **layers**, maps the input data to predictions.
The loss function then compares the predictions to the target values, producing a **loss value**: a measure of how well the model's predictions match what was expected.
The **optimizer** uses this loss value to update the model's **weights**.

### Input

Now we understand that the input images are stored in NumPy tensors.
Prior to training the model, the input images - training and testing images - were pre-processed: training tensors were converted to type `float32` and reshaped to shape `(60000, 28*28)` from `(60000, 28, 28)`, and testing tensors were similarly reformatted and reshaped `(10000, 28*28)` from `(10000, 28, 28)`.

```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

### Layers

Recall that our two-layer neural network model was created like so:

```python
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

We now understand that this model consists of a chain of two `Dense` layers.
Each layer performs simple tensor operations to the input data, further refining the data to more useful data representations.

These layers are incorporate the usage of layer *weight* tensors.
Weight tensors, which are attributes of the layers, are where the *knowledge* of the model persists.

### Loss function and optimizer

This was the model-compilation step:

```python
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```

We understand that `sparse_categorical_crossentropy` is the loss function that's used as to calculate the loss score.
The loss score is used as a feedback signal for learning the weight tensors.
During the training phase, the training loop will attempt to minimize the loss score.

The reduction of the loss happens via mini-batch stochastic (random) gradient descent.
The exact rules and specifications of loss reduction are defined by the `rmsprop` optimizer passed as the model's first argument.

### Training loop

Finally, this was the model's training loop:

```python
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

Fitting the model to the training data is simple: the model will iterate on the training data in mini-batch of 128 samples, 5 times over.
Each iteration over the entire training dataset is called an *epoch*.
Given that there are 60000 training images, there are a total of 60000/128 (~469, or 500) mini-batches.

For each mini-batch, the model will compute the gradient of the loss with regard to the weights.
Using the *Backpropagation* algorithm (which derives from the chain rule in calculus), the optimizer moves the weights in the direction that will reduce the value of the loss for this batch.

And that's it!
It sounds complicated when all the keywords are used, but we firmly understand that it's simply matrix multiplication, addition, subtraction, and derivatives.

---
## Summary

- *Tensors* form the foundation of modern machine learning systems. They come in various flavors of `rank`, `shape`, and `dtype`.

- We can manipulate numerical tensors via *tensor operations*: addition, tensor product, or element-wise multiplication.
In general, everything in deep learning is comparable to a geometric transformation.

- Deep learning models consist of sequences of simple tensor operations, parameterized by *weights*, which are tensors themselves.
The weights of a model are where the model's "knowledge" is stored.

- *Learning* means finding a set of values for the model's weights such that the *loss score* is minimized for a given batch of training data samples.

- Learning happens by drawing random batches of data samples and their targets, and computing the gradient of the model parameters with respect to the batch's loss score.
The model's parameters are then moved - the magnitude of which is determined by the learning rate - in the opposite direction from gradient.
This is called *mini-batch stochastic gradient descent*.

- The entire learning process is made possible by the fact that all tensor operations in neural networks are differentiable, making it possible to apply the chain rule of derivation.
The chain rule of derivation allows us to find the gradient function mapping the current parameters and current batch of data to a gradient value.
This is called *backpropagation*.

- Two key concepts we'll see frequently in future chapters are *loss* and *optimizers*.
    - The *loss* is the quantity we'll attempt to minimize during training, so it should represent a measure of success for the task we're trying to solve.
    - The *optimizer* specifies the exact way in which the gradient of the loss will be used to update parameters.