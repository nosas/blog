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
            - [Why does The Brrrgh have the most Bossbot samples?](#why-does-the-brrrgh-have-the-most-bossbot-samples)
            - [Why are there so many Lawbot and Sellbot samples in Daisy's Garden?](#why-are-there-so-many-lawbot-and-sellbot-samples-in-daisys-garden)
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
        - [Keras Tuner](#keras-tuner)
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

<figure class="center" style="width:95%">
    <img src="img/dataset_streets.png" style="width:100%;"/>
    <figcaption>Dataset balance per street</figcaption>
</figure>

There image above shows two notable imbalances.
In the Suits per street graph, we can see:

1. Daisy's Garden (DG) has the most Lawbot and Sellbot samples.
2. The Brrrgh (BR) has the most Bossbot samples.

#### Why does The Brrrgh have the most Bossbot samples?

The street I frequented to acquire samples in The Brrrgh was [Walrus Way](https://toontown.fandom.com/wiki/Walrus_Way), where the Cog presence is split [90%, 10%] between Bossbots and Lawbots, respectively.
It's no surprise that the majority of the samples from the Brrrgh street are Bossbot samples.

#### Why are there so many Lawbot and Sellbot samples in Daisy's Garden?

The Lawbot imbalance is because the street I visited was majority Lawbot and I took too many screenshots of Bottom Feeders.
The Sellbot imbalance, on the other hand, is due to the Sellbot HQ being located near Daisy's Garden.
As a result, the majority of Sellbot Cogs reside in the [streets of DG](https://toontown.fandom.com/wiki/Daisy_Gardens#Streets).

I would not have noticed either of these imbalances without charting the samples per street.
Moving forward, we'll be more conscious of street imbalance and take screenshots of Cogs from other streets.


### Creating the dataset objects

We'll create simple tuples of images and labels instead of `Keras.dataset` objects.

```python
def create_suit_datasets(
    split_ratio: list[float, float, float] = None
) -> tuple[tuple[np.array[float], np.array[float]]]:
    """Create multiclass Cog suit datasets for training, validation, and testing

    Args:
        split_ratio (list[float, float, float], optional): Train/val/test split. Defaults to None.

    Returns:
        tuple[tuple, tuple, tuple]: Train, validate, and test datasets. Each tuple contains
                                    a numpy array of images and a numpy array of labels.
    """
    if split_ratio:
        split_data(split_ratio=split_ratio)

    data_dirs = [TRAIN_DIR, VALIDATE_DIR, TEST_DIR]
    suits = get_suits_from_dir(directories=data_dirs)
    result = {}
    for dir_name in data_dirs:
        filepaths, labels = suits[dir_name]
        labels = np.asarray(suit_to_integer(labels), dtype=np.float32)
        images = np.asarray(
            [get_img_array_from_filepath(fp) for fp in filepaths], dtype=np.float32
        )
        result[dir_name] = (images, labels)

    return (result[TRAIN_DIR], result[VALIDATE_DIR], result[TEST_DIR])
```

The datasets are now ready to be used by the model.
We can retrieve the datasets, and separate the images and labels, as follows:

```python
# Split unsorted images into train, validate, and test sets
ds_train, ds_validate, ds_test = create_suit_datasets(split_ratio=[0.6, 0.2, 0.2])
# Create the dataset
train_images, train_labels = ds_train
val_images, val_labels = ds_validate
test_images, test_labels = ds_test
```

#### Splitting the images into train, validate, and test

Previously, we split the entire dataset into 60%/20%/20% train/validate/test sets.
This resulted in unbalanced suit samples in each set.
For instance, the Bossbot suit samples would contained a disproportionate number of sample from Flunkies.

Now, in order to maintain balanced datasets for each Cog suit, we split each individual Cog entity using the 60/20/20 split.
This ensures a representative sample of each Cog entity.

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

### Keras Tuner

`KerasTuner` is a tool for fine-tuning a model's hyperparameters.
Hyperparameters include the model's layers, layer sizes, and optimizer.
We can leverage this tool to find the best hyperparameters for our model instead of manually tuning the model and comparing the results.

### Preventing overfitting

### Wrong predictions

### Baseline comparison: Training

### Baseline comparison: Evaluation

---
## Model interpretation and visualization

### Intermediate convnet outputs (intermediate activations)

### Convnet filters

### Class activation heatmaps
