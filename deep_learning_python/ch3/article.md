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
This is where the TensorFlow's *variables* - `tf.Variable` - come in to play:

```python
>>> v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
>>> print(v)
array([[-0.644994 ],
       [ 1.47064  ],
       [-0.6413262]], dtype=float32)>
```

The state of a variable - the entirety or a subset of coefficients - can be modified via its `assign` method:

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
