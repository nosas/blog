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

TensorFlow far surpasses NumPy is the following ways:

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

