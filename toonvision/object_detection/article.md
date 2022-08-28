<title>ToonVision: Object detection</title>

# ToonVision - Object Detection

This article is the third in a series on **ToonVision**.
The [first article](https://fars.io/toonvision/classification/) covered the basics of classification and binary classification of Toons and Cogs.
More specifically, the previous article covered how to...

- convert a binary dataset into a multiclass dataset
- use classification performance measures such as precision, recall, and F1-score
- automatically optimize hyperparameter values with Keras-tuner
- interpret and visualize what the model is learning with confusion matrices and class activation maps

This article explains object detection of Cogs and Toons - also called entity detection.
After reading this article, you should have a better understanding of how to...

- differentiate between object detection algorithms (YOLO, R-CNN, SSD)
- create an object detection model
- extract detected objects from images and videos
- build data pipelines to semi-autonomously grow a dataset (semi-supervised learning)

The next article will cover image segmentation of ToonTown's streets, roads, Toons, Cogs, and Cog buildings.
For now, let's focus on object detection.

- [ToonVision - Object Detection](#toonvision---object-detection)
    - [ToonVision](#toonvision)
    - [Object detection](#object-detection)
        - [Object detection algorithms](#object-detection-algorithms)
        - [SSD](#ssd)
        - [R-CNN](#r-cnn)
        - [YOLO](#yolo)
    - [Creating an object detection model](#creating-an-object-detection-model)

## ToonVision

ToonVision is my computer vision project for teaching a machine how to see in [ToonTown Online](https://en.wikipedia.org/wiki/Toontown_Online) - an MMORPG created by Disney in 2003.
The project's objective is to teach a machine (nicknamed **OmniToon**) how to play ToonTown and create a self-sustaining ecosystem where the bots progress through the game together.
In the process, I will explain the intricacies of building computer vision models, configuring static and real-time (stream) data pipelines, and visualizing results and progress.

---
## Object detection

Object detection is a computer vision task of detecting instances of objects from pre-defined classes in images and videos.
Specifically, object detection algorithms detect and draw a bounding box around each object.

<font style="color:red">TODO: Include row of ToonVision images: classification, classification + localization, object detection, instance segmentation</font>

Common use-cases include face detection in cameras and pedestrian detection in autonomous vehicles.
In ToonVision's, object detection is applied to locating all entities - Cogs and Toons - in both images and real-time video.


### Object detection algorithms


### SSD

### R-CNN

### YOLO

---
## Creating an object detection model
