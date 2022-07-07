<title>ToonVision: Multiclass Classification</title>

# ToonVision - Multiclass Classification

This article is the second in a series on **ToonVision**.
The [first article](https://fars.io/toonvision/classification/) covers the basics of classification - specifically binary classification of Toons and Cogs.

ToonVision is my computer vision project for teaching a machine how to see in [ToonTown Online](https://en.wikipedia.org/wiki/Toontown_Online) - an MMORPG created by Disney in 2003.
The ultimate goal is to teach a machine (nicknamed **OmniToon**) how to play ToonTown and create a self-sustaining ecosystem where the bots progress through the game together.

This article covers multiclass classification of Cogs: Cog suits (4 unique suits) and Cog entities (32 unique Cogs).

After reading this article, we'll have a better understanding of how to

- deal with a model overfitting to a small, imbalanced dataset
- utilize image augmentation and dropout to improve the model's generalization capability
- compare different models, optimizers, and hyperparameters
- interpret and visualize what the model is learning

The following article will cover image segmentation of ToonTown's streets, roads, Cogs, and Cog buildings.
Afterwards, we'll implement real-time object detection and, if possible, image segmentation.
For now, let's focus on multiclass classification.

---
## Classification

### Multiclass classification

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
