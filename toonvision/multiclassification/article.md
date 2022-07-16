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

<details>
    <summary>Table of Contents</summary>

- [ToonVision - Multiclass Classification](#toonvision---multiclass-classification)
    - [ToonVision](#toonvision)
    - [Classification](#classification)
        - [Multiclass classification](#multiclass-classification)
    - [The ToonVision dataset](#the-toonvision-dataset)
        - [Dataset considerations](#dataset-considerations)
        - [Why does the street matter?](#why-does-the-street-matter)
        - [Dataset balance](#dataset-balance)
        - [Creating the dataset objects](#creating-the-dataset-objects)
            - [Splitting the images into train, validate, and test](#splitting-the-images-into-train-validate-and-test)
    - [Compiling the model](#compiling-the-model)
        - [Loss function](#loss-function)
        - [Optimizer](#optimizer)
        - [Metrics](#metrics)
        - [Callbacks](#callbacks)
        - [Defining the model](#defining-the-model)
    - [Training the baseline model](#training-the-baseline-model)
        - [Baseline loss and accuracy plots](#baseline-loss-and-accuracy-plots)
        - [Baseline wrong predictions](#baseline-wrong-predictions)
    - [Training the optimized model](#training-the-optimized-model)
        - [Preventing overfitting](#preventing-overfitting)
        - [Wrong predictions](#wrong-predictions)
        - [Baseline comparison: Training](#baseline-comparison-training)
        - [Baseline comparison: Evaluation](#baseline-comparison-evaluation)
    - [Model interpretation and visualization](#model-interpretation-and-visualization)
        - [Intermediate convnet outputs (intermediate activations)](#intermediate-convnet-outputs-intermediate-activations)
        - [Convnet filters](#convnet-filters)
        - [Class activation heatmaps](#class-activation-heatmaps)
</details>

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

We'll tweak the existing dataset considerations to focus on balancing the dataset's Cog entity samples.
The current Cog dataset contains two glaring problems:

1. The samples do not account for *where* - on *which street* - the Cog is located.
1. There are no samples of the two highest-ranked Cogs

To address problem #1, the ToonVision dataset now requires a balanced number of samples from each of the 6 uniquely designed streets: The Brrrgh, Daisy's Garden, Donald's Dreamland, Donald's Dock, Minnie's Melodyland, and ToonTown Central.
Given that there are 6 streets, the sample images per Cog will increase from 20 to 30; we need 5 images of a Cog from each street.

We'll leverage Cog invasions to ensure we meet the new dataset requirement because some Cogs are not present in certain streets *unless* there's an ongoing invasion.
Cog invasions will solve both problems moving forward.

### Why does the street matter?

Each street in ToonTown has a unique design - colors, houses, floors, obstacles - resulting in unique backgrounds for our extracted objects.
In this case, our extracted objects are Cogs.
If we have a diverse dataset of Cogs with different backgrounds, we can teach our model to better recognize Cog features rather than background features.
Therefore, it's important to take screenshots of Cogs from each street so our model can generalize Cog features across all streets.

### Dataset balance

The Cog vs Toon dataset is imbalanced with a majority of the dataset belonging to the Cog class.
However, the Cog dataset is mostly balanced.
Our goal is ~30 samples per Cog entity with at least 5 samples per street.
We have achieved the 30 samples per Cog entity requirement, but we're not meeting the 5 samples per street requirement.
Refer to the image below to see the current dataset balance and the balance per street.

<font style="color:red">TODO: Add image of dataset balance per street</font>

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
