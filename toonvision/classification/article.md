# ToonVision - Classification

This article is first in a series on **ToonVision**.

ToonVision is my personal computer vision project to teach a machine how to see in (ToonTown)[https://en.wikipedia.org/wiki/Toontown_Online] - an MMORPG created by Disney in 2002.
The ultimate goal is to teach a machine how to play ToonTown (nicknamed **OmniToon**) and create a self-sustaining ecosystem within the game where the bots progress through the game together.

In later articles, we'll dive into segmentation and object detection.
For now, let's focus on real-time classification.

This article covers ...

- Binary classification: Toon vs Cog
- Multiclass classification: Cog suits (4 unique suits) and Cog names (32 unique names)


<details>
    <summary>Table of Contents</summary>

- [ToonVision - Classification](#toonvision---classification)
    - [Classification](#classification)
        - [Binary Classification](#binary-classification)
        - [Multiclass Classification](#multiclass-classification)
    - [ToonTown](#toontown)
        - [Toon](#toon)
        - [Cog](#cog)
        - [Why is it important for Toons to classify Cogs?](#why-is-it-important-for-toons-to-classify-cogs)
    - [Creating the dataset](#creating-the-dataset)
        - [Dataset considerations](#dataset-considerations)
        - [Filename and data folder structure](#filename-and-data-folder-structure)

</details>

---
## Classification

### Binary Classification

### Multiclass Classification

We could beef up the multiclass classification to a multilabel multiclass classification: Cog state/level/hp/name/suit.
But this adds unneeded complexity.
Let's keep it simple.
In the future, I will surely add the classification of the Cog's state: battle, patrolling, spawning, de-spawning, etc.

---
## ToonTown

### Toon

There are X number of animals.

### Cog

There are 4 Cog suits: Lawbot, Cashbot, Sellbot, and Bossbot.
Each suit has 8 Cogs, totaling 32 unique Cogs.

### Why is it important for Toons to classify Cogs?

---
## Creating the dataset

### Dataset considerations

- Images are split into training, validation, and test sets: 40% training, 20% validation, and 40% test.
- Images of Toons and Cogs must be...
    - Taken from the street, not playground
    - Taken of the entity's front, back, and side
- In the Cog dataset, there must be an equal part of each Cog suit
    - There must must be an equal part of each unique Cog (32 unique Cogs)
        - There is a minimum requirement of 20 images per unique Cog (32*20 = 640 images total)
    - Must not include the Cog's nametag in the image
- There must be an equal part of Toons and Cogs in each set
    - There must be an equal part Cog suit in each set
- In the Toon dataset, balance of animal types is welcome but not necessary

### Filename and data folder structure

Cog filename structure: `cog_<suit>_<name>_<index>.png`.

Toon filename structure: `toon_<animal>_<index>.png`.

Data folder structure:
```
data/
    - [train/val/test]/
        - cog/
        - toon/
```

There's no need for a unique folder for each Cog suit because we can filter on the filename.