<title>Deep Learning with Python: Chapter 3 - Introduction to Keras and TensorFlow</title>


# Deep Learning with Python  <!-- omit in toc -->

This article is part 3/13 (?) of a series of articles named *Deep Learning with Python*.

In this series, I will read through the second edition of *Deep Learning with Python* by François Chollet.
Articles in this series will sequentially review key concepts, examples, and interesting facts from each chapter of the book.

<details>
    <summary>Table of Contents</summary>

- [Chapter 3: Introduction to Keras and TensorFlow](#chapter-3-introduction-to-keras-and-tensorflow)
    - [What's TensorFlow?](#whats-tensorflow)
        - [TensorFlow vs. NumPy](#tensorflow-vs-numpy)
        - [TensorFlow ecosystem](#tensorflow-ecosystem)
    - [What's Keras?](#whats-keras)
        - [Keras and TensorFlow: A brief history](#keras-and-tensorflow-a-brief-history)
    - [Setting up a deep learning workspace](#setting-up-a-deep-learning-workspace)
        - [Physical machine with NVIDIA GPU](#physical-machine-with-nvidia-gpu)
        - [Cloud GPU instances](#cloud-gpu-instances)
        - [Google Colab](#google-colab)
    - [First steps with TensorFlow](#first-steps-with-tensorflow)
        - [Constant tensors and variables](#constant-tensors-and-variables)
        - [A second look at the Gradient Tape API](#a-second-look-at-the-gradient-tape-api)
        - [Computing second-order gradients](#computing-second-order-gradients)
    - [Linear classifier example in pure TensorFlow](#linear-classifier-example-in-pure-tensorflow)
        - [What is linear classification?](#what-is-linear-classification)
        - [Generating synthetic data](#generating-synthetic-data)
</details>

---
# Chapter 3: Introduction to Keras and TensorFlow

This chapter covers...

* A closer look at TensorFlow, Keras, and their relationship
* Setting up a deep learning workspace
* Review of how deep learning concepts learned in previous chapters translate to Keras and TensorFlow

This chapter gives us everything required to get started with deep learning.
By the end of this chapter, we'll be ready to move on to practical, real-world applications of deep learning.

---
## What's TensorFlow?

TensorFlow is a free and open-source machine learning framework for Python.
It was primarily developed by Google.
Similar to NumPy, it is a general-purpose and efficient numerical library used by engineers to manipulate mathematical expressions using numerical tensors.

### TensorFlow vs. NumPy

TensorFlow far surpasses NumPy in the following ways:

- Automatically computes the gradient of any differentiable expressions (as seen in Ch2 with `GradientTape`)
- Runs not only on CPUs, but also on GPUs and TPUs (highly-parallel hardware accelerators)
- Computations defined in TensorFlow can be easily distributed across multiple devices (CPUs, GPUs, and TPUs) and machines
- TensorFlow programs can be exported and easily deployed to other runtimes, such as C++, JavaScript (for browsers), or TensorFlow lite (for mobile devices or embedded systems)

### TensorFlow ecosystem

TensorFlow is much more than a single Python library.
Rather, it's a platform home to a vast ecosystem of components.

Components of the ecosystem include:

- TF-Agents for reinforcement learning
- TF-Hub (repository) for pre-trained deep neural networks
- TensorFlow Serving for production deployment
- TFX for distributed training and ML workflow managements

Together, these components cover a wide-range of use cases: from cutting-edge research to large-scale production, or just a simple image classification application to see if a dog or a cat is in a picture.

Scientists from Oak Ridge National Lab have used TensorFlow to train an extreme weather forecasting model on the 27,000 GPUs within the IBM Summit supercomputer.
Google, on the other hand, has used TensorFlow to develop deep learning applications such as the chess-playing and Go-playing agent AlphaZero.

---
## What's Keras?

Keras is a high-level deep learning API built on top of TensorFlow.
It provides a convenient and flexible API for building and training deep learning models.

<font style="color:red">TODO: Insert image of Keras, TF, hardware hierarchical diagram</font>

Keras is known for providing a clean, simple, and efficient API that prioritizes the developer experience.
It's an API for human beings, not machines, and follows best practices for reducing cognitive load.
The API provides consistent and simple workflows, minimizes the number of actions required for common use cases, and outputs clear and actionable feedback upon user error.

Much like Python, Keras' large and diverse user base enables a well-documented and wide range of workflows for utilizing the API.
Keras does not force one to follow a single "true" way of building and training models.
Rather, it allows the user to build and train models corresponding to their own preference, and to explore the possibilities of each approach.

### Keras and TensorFlow: A brief history

Keras was designed originally in March 2015 to be used with Theano, a tensor-manipulation library developed by Montreal Institute for Learning Algorithms (MILA).
Theano pioneered the idea of using static computation graphs for automatic differentiation and compiling code to both CPU and GPU support.

Following the release of TensorFlow 1.0 in November 2015, Keras was refactored to support multiple backend architectures: starting with Theano and Tensorflow in late 2015, and adding support for CNTK and MXNet in 2017.

Keras became well known as the user-friendly way to develop TensorFlow applications.
By late 2017, a majority of TensorFlow users were using Keras.
In 2018, the TensorFlow leadership picked Keras and TensorFlow's official high-level API.
As a result, as of September 2019, the Keras API is the official API for TensorFlow 2.0.

Enough of the history, let's learn how to set up a deep learning workspace.

---
## Setting up a deep learning workspace

There are a handful of ways to set up a deep learning workspace:

- Buy and install a physical machine with an NVIDIA GPU
- Use GPU instances on AWS, Google Cloud, or cheaper alternatives such as Jarvis Labs
- Use the free GPU runtime from Google Colab, a hosted Jupyter notebook service that executes code on GPUs and even TPUs

Each approach has its own advantages and disadvantages in terms of flexibility, cost, and ease of use.
I'll briefly discuss the advantages and disadvantages of each approach below, but I will not discuss setup at all.

### Physical machine with NVIDIA GPU

Buying a machine with a GPU is not the easiest way to get started with DL, as it requires manual setup and it's also the most expensive upfront.
The upfront cost is amplified by the current (as of May 2022) chip shortage and GPU scalpers.
This method involves installing proper drivers, sorting out version conflicts, and then configuring the libraries to use the GPU instead of the CPU.

Most users already have NVIDIA GPUs installed in their computers.
Given the large user base of TensorFlow, there are many tutorials for setting up NVIDIA GPUs for deep learning, so this is not a bad option for tech-savvy people.

The alternative to buying a GPU is the use of embedded devices built specifically for efficient and highly-parallelized math operations, such as [NVIDIA's Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) or [Google's Coral](https://coral.ai/products/).

### Cloud GPU instances

Using GPU instances is cheaper in the short-term because you pay as you go (per hour basis), but it's not sustainable in the long run if you're a heavy user of deep learning.
The benefit of using GPU instances is that it requires minimal setup as most instances have Python, TensorFlow, and Keras pre-installed - it's mostly plug-and-play and easy to use.
Moreover, the GPU instance can easily be upgraded, torn down, cloned, and re-installed.
Lastly, students and enterprise employees often get discounts - or free usage - for AWS and Google Cloud.

I personally use Jarvis Labs and AWS for my deep learning needs because I have discounts for both services.
There aren't many differences between cloud instance providers other than the availability of high-powered GPUs.

### Google Colab

The last approach - using free GPUs from Google Colab - is the simplest way to get started with deep learning.
It's recommended for those who are not familiar with the hardware and software, and for those who are new to deep learning.
Francois himself recommends executing code examples found in the book using Google Colab as it requires the least amount of setup.
The drawback of Colab is that the free GPU is time-limited and shared by users - meaning that the execution may be slower.

---
## First steps with TensorFlow

Training a neural network revolves around low-level tensor manipulations and high-level deep learning concepts.
TensorFlow takes care of the tensor manipulation through the use of:

- *Tensors*, including special tensors that store the network's state (*variables*)
- *Tensor operations* such as addition, `relu`, `matmul`, etc.
    - The previous article details [tensor operations](https://fars.io/deep_learning_python/ch2/#tensor-operations)
- *Backpropagation*, a way to compute gradients of mathematical operations (using TensorFlow's `GradientTape`)
    - The previous article discusses [backpropagation](https://fars.io/deep_learning_python/ch2/#backpropagation) and [TensorFlow's GradientTape](https://fars.io/deep_learning_python/ch2/#tensorflows-gradient-tape)

Let's take a deeper dive into how all of the concepts above translate to TensorFlow.

### Constant tensors and variables

To do anything in TensorFlow, we need to create tensors.
Let's look at code examples for creating tensors with all ones, zeros, or random values:

```python
import tensorflow as tf

# Equivalent to np.ones((2, 2))
t_ones = tf.ones(shape=(2, 2))
# Equivalent to np.zeros((2, 1))
t_zeros = tf.zeros(shape=(2, 1))
# Equivalent to np.random.normal(size=(3, 1), loc=0., scale=1)
t_random_normal = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
# Equivalent to np.random.uniform(size=(1, 3), low=0., high=1.)
t_random_uniform = tf.random.uniform(shape=(1, 3), minval=0., maxvval=1.)
```

What do the outputs of each tensor look like?

```python
>>> print(t_ones)
tf.Tensor(
    [[1. 1.]
     [1. 1.]], shape=(2, 2), dtype=float32)

>>> print(zeros)
tf.Tensor(
    [[0.]
     [0.]], shape=(2, 1), dtype=float32)

>>> print(t_random_normal)
tf.Tensor(
    [[-0.8276905]
     [ 0.2264915]
     [ 0.1399505]], shape=(3, 1), dtype=float32)

>>> print(t_random_uniform)
tf.Tensor(
    [[0.141 0.824 0.912]], shape=(1, 3), dtype=float32)
```

A significant difference between NumPy arrays and TensorFlow tensors is that tensors are not assignable: they're *constant*.
For instance, in NumPy, we can assign a value to a tensor, as seen in the code block below.
Whereas, in TensorFlow, we are greeted with an error: `TypeError: 'Tensor' object does not support item assignment`.

```python
>>> import numpy as np
>>> t_ones = np.ones((2, 2))
>>> t_ones
array([[1., 1.],
       [1., 1.]])
>>> t_ones[0, 0] = 0
>>> t_ones
array([[0., 1.],
       [1., 1.]])

>>> import tensorflow as tf
>>> t_ones = tf.ones((2, 2))
>>> t_ones
tf.Tensor(
    [[1. 1.]
     [1. 1.]], shape=(2, 2), dtype=float32)
>>> t_ones[0, 0] = 0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'Tensor' object does not support item assignment
```

To train a model, however, it's important to be able to change the values of the tensors - update the weights of the model.
This is where the TensorFlow's *variable* (`tf.Variable`) comes in to play:

```python
>>> v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
>>> print(v)
array([[-0.644994 ],
       [ 1.47064  ],
       [-0.6413262]], dtype=float32)>
```

The state of a variable - the entirety of or a subset of coefficients - can be modified via its `assign` method:

```python
>>> v.assign(tf.ones((3, 1)))
array([[1.],
       [1.],
       [1.]], dtype=float32)>
>>> v[0,0].assign(9)
array([[9.],
       [1.],
       [1.]], dtype=float32)>
```

Similarly, the `assign_add()` and `assign_sub()` are tensor-efficient equivalents of `+=` and `-=`, respectively.

```python
>>> v.assign_add(tf.ones((3, 1)))
array([[10.],
       [ 2.],
       [ 2.]], dtype=float32)>
>>> v.assign_sub(tf.ones((3, 1)))
array([[9.],
       [1.],
       [1.]], dtype=float32)>
```

### A second look at the Gradient Tape API

TensorFlow's ability to retrieve gradients of any expression with respect to any of its inputs is what makes the TensorFlow library so powerful.
All we have to do is open a `GradientTape` context, manipulate the input tensors, and retrieve the gradients with respect to the inputs.

```python
import tensorflow as tf

input = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    output = input * input

# <tf.Tensor: shape=(), dtype=float32, numpy=6.0>
gradient = tape.gradient(output, input)
```

The gradient tape is most commonly used to retrieve the gradients of the model's loss with respect to its weights: `gradient = tape.gradient(loss, weights)`.
We discussed and demonstrated this functionality in the [previous article](https://fars.io/deep_learning_python/ch2/#tensorflows-gradient-tape).

So far, we've only looked at the simplest case of `GradientTapes` - where the input tensors in `tape.gradient()` were TensorFlow variables.
It's actually possible for the input tensors to be any arbitrary tensor, not just variables, by calling `tape.watch(arbitrary_tensor)` within the `GradientTape` context.

```python
arbitrary_tensor = tf.constant(value=2.)
with tf.GradientTape() as tape:
    tape.watch(arbitrary_tensor)
    output = arbitrary_tensor * arbitrary_tensor

# tf.Tensor(4.0, shape=(), dtype=float32)
gradient = tape.gradient(output, arbitrary_tensor)
```

By default, only *trainable variables* are tracked because computing the gradient of a loss with regard to a trainable variable is the most common use case.
Furthermore, it would be too expensive to preemptively store the information required to compute the gradient of anything with respect to anything.
In an effort avoid wasting resources, only the trainable variables are tracked unless otherwise explicitly specified.

### Computing second-order gradients

The gradient tape is a powerful utility capable of computing *second-order gradients* - or, the gradient of a gradient.

For instance, the gradient of the position of an object with respect to time is the speed of the object.
The second-order gradient of the object speed is its acceleration.


```python
time = tf.Variable(initial_value=0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position, time)

# <tf.Tensor: shape=(), dtype=float32, numpy=9.8>
acceleration = outer_tape.gradient(speed, time)
```

---
## Linear classifier example in pure TensorFlow

We now know about tensors, variables, tensor operations, and gradient computation.
That's enough to build any machine learning model based on gradient descent.
Let's put our knowledge to the test and build an end-to-end linear classification model purely in TensorFlow.

We're going to implement a linear classifier that predicts whether a given input belongs to class A or class B.
But first, we need to understand what linear classification is.

### What is linear classification?

In linear classification problems, the model is trying to find a linear combination of the input features that best predicts the target variable.
Simply put, the model is trying to classify input data into 2+ categories (classes) by drawing a line through the the data.
The line is best fit to separate the data into two classes.

<font style="color:red">TODO: Insert image of a linear classification plot with a line separating the classes</font>

This is the basic idea behind linear classification.
Now let's generate some data and train a linear classifier.
All of the code related to this linear classifier can be found on my [GitHub](https://github.com/nosas/blog/blob/main/deep_learning_python/ch3/code/linear_classifier.py) as an interactive python file.
I recommend using VSCode to utilize the interactive python code blocks - similar to Jupyter Notebooks.

### Generating synthetic data

We need some nicely linear data to train our linear classifier.
To keep it simple, we'll create two classes of points in a 2D plane and call them class A and class B.
To keep it more simple, we won't explain all the math behind the data generation - just understand that both classes should be clearly separated and roughly distributed like a cloud.
We'll just use the following formula to generate the data:

```python
num_samples_per_class = 500
class_a_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0], [0, 1]],
    size=num_samples_per_class)
class_b_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0], [0, 1]],
    size=num_samples_per_class)
```

The figure below shows the linearly-separable data from classes A and B.
See the following code block to see how we plot the data.

<figure class="center">
    <img src="img/linear_classifier_data.png" style="width:100%;background:white;"/>
    <figcaption>Two classes of synthetic and random points in the 2D plane</figcaption>
</figure>

Both samples are arrays of shape `(500, 2)` - meaning there are 500 rows of 2-dimensional data (x and y coordinate points).
Let's stack both class samples into a single array with shape `(1000, 2)`.
Stacking the samples into single array will allow for easier processing later on, such as plotting the data.

```python
import matplotlib.pyplot as plt
import numpy as np

# The first 500 samples are from class A, the next 500 samples are from class B
inputs = np.vstack((class_a_samples, class_b_samples)).astype(np.float32)
# The first 500 labels are 0 (class A), and the next 500 are 1 (class B)
labels = np.vstack(
    (
        np.zeros((num_samples_per_class, 1), dtype=np.float32),
        np.ones((num_samples_per_class, 1), dtype=np.float32),
    )
)
class_a = inputs[:num_samples_per_class]
class_b = inputs[num_samples_per_class:]

# %% Plot the two classes
# Class A is represented by green dots, and class B is represented by blue dots,
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
```
