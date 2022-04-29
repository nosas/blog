<title>Deep Learning with Python: Chapter 2 - Mathematical building blocks of neural networks</title>


# Deep Learning with Python

This article is part 2/13 (?) of a series of articles named *Deep Learning with Python*.

In this series, I will read through the second edition of *Deep Learning with Python* by François Chollet.
Articles in this series will sequentially review key concepts, examples, and interesting facts from each chapter of the book.

---
# Chapter 2: The mathematical building blocks of neural networks?

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

<font style="color:red">TODO: Insert MNIST sample digits</font>

### Defining the network architecture

### Preparing the model for training

1. *An optimizer*
1. *A loss function*
1. *Metrics to monitor during training and testing*

---
## Data representations: Tensors

*Tensors* are fundamental data structures used in machine learning.
At its core, a tensor is a container for data - usually numeric data.
Matrices (2D arrays) are considered to be rank-2 tensors.

Therefore, tensors are generalizations of matrices to an arbitrary number of *dimensions*.
Note that in the context of tensors, a dimension is often called an *axis*.

Let's take a look at definitions and examples of rank-0 to rank-3 and higher tensors.

### Scalars (rank-0 tensors)

A tensor that contains only one number is called a *scalar* - or scalar tensor, rank-0 tensor, or 0D tensor).
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

### Matrices (rank-2 tensors)

### Rank-3 and higher-rank tensors

### Key attributes

- *Number of axes (rank)*: For instance, a rank-3 tensor has three axes, and a matrix has two axes. This is also called the tensor's `ndim` in Python libraries such as NumPy or TensorFlow.
- *Shape*: This is a tuple of integers that describes how many dimensions the tensor has along each axis.
For instance, a matrix with shape `(3, 5)` has three rows and five columns.
A vector with a single element could have the shape `(5,)`, whereas a scalar has an empty shape, `()`.
Lastly, a rank-3 tensor, such as the example above, has shape `(3, 3, 5)`.
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

---
## How neural networks learn via backpropagation and gradient descent