<title>ToonVision: Object detection</title>

# ToonVision - Object Detection

This article is the third in a series on **ToonVision**.
The [first article](https://fars.io/toonvision/classification/) covered the basics of classification and binary classification of Toons and Cogs.
More specifically, the previous article covered how to...

- convert a binary dataset into a multiclass dataset
- use classification performance measures such as precision, recall, and F1-score
- automatically optimize hyperparameter values with Keras-tuner
- interpret and visualize what the model is learning with confusion matrices and class activation maps

This article covers the object detection of Cogs and Toons.
After reading this article, we'll have a better understanding of...

- differences between object detection algorithms (YOLO vs R-CNN)
- extracting detected objects from images and videos
- automatically build data pipelines to grow a dataset

The next article will cover image segmentation of ToonTown's streets, roads, Toons, Cogs, and Cog buildings.
For now, let's focus on object detection.

- [ToonVision - Object Detection](#toonvision---object-detection)
