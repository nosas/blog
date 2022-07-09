<title>ToonVision: Multiclass Classification</title>

# ToonVision - Multiclass Classification

This article is the second in a series on **ToonVision**.
The [first article](https://fars.io/toonvision/classification/) covered the basics of classification and binary classification of Toons and Cogs.
This article covers multiclass classification of Cogs: Cog suits (4 unique suits) and Cog entities (32 unique Cogs).

After reading this article, we'll have a better understanding of how to

- deal with a model overfitting to a small, imbalanced dataset
- utilize image augmentation and dropout to improve the model's generalization capability
- compare different models, optimizers, and hyperparameters
- interpret and visualize what the model is learning

The following article will cover image segmentation of ToonTown's streets, roads, Cogs, and Cog buildings.
Afterwards, we'll implement real-time object detection and, if possible, image segmentation.
For now, let's focus on multiclass classification.

## ToonVision

ToonVision is my computer vision project for teaching a machine how to see in [ToonTown Online](https://en.wikipedia.org/wiki/Toontown_Online) - an MMORPG created by Disney in 2003.
The ultimate goal is to teach a machine (nicknamed **OmniToon**) how to play ToonTown and create a self-sustaining ecosystem where the bots progress through the game together.

---
## Classification

As discussed in the [previous article](https://fars.io/toonvision/classification/#classification), image classification is the process of assigning a label to an input image.
For instance, given a dog-vs-cat classification model and an image of a Pomeranian, the model will predict that the image is a dog.

There are a few variants of the image classification problem: binary, multiclass, multi-label, and so on.
We'll focus on [multiclass classification](https://fars.io/toonvision/classification/#multiclass-classification) in this article.

### Multiclass classification

Multiclass classification is a problem in which the model predicts which class an input image belongs.
For instance, the model could predict that an animal belongs to the class of dogs, cats, rabbits, horses, or any other animal.

We're building a model to predict which Cog suit a Cog belongs to - a 4-class classification problem.
We'll push the model further by also predicting which Cog entity a Cog belongs to - a 32-class classification problem.
Both multiclass classification problems require us to improve the ToonVision dataset.
The current dataset is imbalanced and does not contain samples of all the Cog suits and entities.
Let's look at how we can improve the dataset in the next section.

---
## The ToonVision dataset

### Dataset considerations

### Dataset balance

### Creating the dataset objects

#### Splitting the images into train, validate, and test

---
## Compiling the model

### Loss function

### Optimizer

### Metrics

### Callbacks

### Defining the model

---
## Training the baseline model

### Baseline loss and accuracy plots

### Baseline wrong predictions

---
## Training the optimized model

### Preventing overfitting

### Wrong predictions

### Baseline comparison: Training

### Baseline comparison: Evaluation

---
## Model interpretation and visualization

### Intermediate convnet outputs (intermediate activations)

### Convnet filters

### Class activation heatmaps
