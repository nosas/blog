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

- differentiate between object detection models (YOLO, R-CNN, SSD)
- create an object detection model
- extract detected objects from images and videos
- build data pipelines to semi-autonomously grow a dataset (semi-supervised learning)

The next article will cover image segmentation of ToonTown's streets, roads, Toons, Cogs, and Cog buildings.
For now, let's focus on object detection.

<details>
    <summary>Table of Contents</summary>

- [ToonVision - Object Detection](#toonvision---object-detection)
    - [ToonVision](#toonvision)
    - [Object detection](#object-detection)
        - [Object detection models](#object-detection-models)
            - [Two-shot](#two-shot)
            - [Single-shot](#single-shot)
        - [R-CNN](#r-cnn)
        - [SSD](#ssd)
        - [YOLO](#yolo)
    - [Creating an object detection model](#creating-an-object-detection-model)
    - [References](#references)
</details>

## ToonVision

ToonVision is my computer vision project for teaching a machine how to see in [ToonTown Online](https://en.wikipedia.org/wiki/Toontown_Online) - an MMORPG created by Disney in 2003.
The project's objective is to teach a machine (nicknamed **OmniToon**) how to play ToonTown and create a self-sustaining ecosystem where the bots progress through the game together.
In the process, I will explain the intricacies of building computer vision models, configuring static and real-time (stream) data pipelines, and visualizing results and progress.

---
## Object detection

Object detection is a computer vision task of detecting instances of objects from pre-defined classes in images and videos.
Specifically, object detection models detect, label, and draw a bounding box around each object.

<font style="color:red">TODO: Include row of ToonVision images: classification, classification + localization, object detection, instance segmentation</font>

Common use-cases include face detection in cameras and pedestrian detection in autonomous vehicles.
In ToonVision's, object detection is applied to locating all entities - Cogs and Toons - in both images and real-time video.


### Object detection models

The most popular object detection models can be split into two main groups: single-shot and two-shot.
Each group has accuracy and speed tradeoffs.
Single-shot models excel in tasks requiring high detection speed, such as in real-time videos.
Two-shot models are slower but more accurate; therefore, they're primarily used in tasks involving image data or low-FPS videos.

#### Two-shot

Two-shot detection models have two stages: region proposal and then classification of those regions and refinement of the location prediction.
The two steps require significant computational resources, resulting in slow training and inference.

Despite the slowness, two-shot models have far superior accuracy when compared to single-shot models.
R-CNN<sup>[1]</sup> is a commonly used two-shot detection model.
Faster R-CNN<sup>[2]</sup>, R-CNN's improved variant, is the more popular choice for two-shot models.


#### Single-shot

Single-shot models are designed for real-time object detection.
They have quicker inference speeds and use less resources during training than two-shot models.
These two properties allow for quick training, prototyping, and experimenting without consuming considerable computation resources.

In a single forward pass, these models predict bounding boxes and class labels directly from the input's feature maps.
They skip the region proposal stage and yield final localization and content prediction at once.
SSD<sup>[3]</sup> and YOLO<sup>[4]</sup> are popular single-shot object detection models capable of running at 5-160 frames per second!

### R-CNN

Two-shot algorithm
2014, many variations, slow but accurate

### SSD

Single-shot algorithm
2015, SOTA at the time, but no longer. YOLO is king

### YOLO

Single-shot algorithm
2015, many different versions v1 -> v7
2022, YOLOv7 newest version, 5-160 FPS<sup>[5]</sup>
Faster than SSD, although less accurate.

---
## Creating an object detection model

---
## References

1. Rich feature hierarchies for accurate object detection and semantic segmentation, [https://arxiv.org/abs/1311.2524](https://arxiv.org/abs/1311.2524)

2. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, [https://arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497)

3. SSD: Single Shot MultiBox Detector, [https://arxiv.org/abs/1512.02325](https://arxiv.org/abs/1512.02325)

4. You Only Look Once: Unified, Real-Time Object Detection, [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)

5. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors, [https://arxiv.org/abs/2207.02696](https://arxiv.org/abs/2207.02696)