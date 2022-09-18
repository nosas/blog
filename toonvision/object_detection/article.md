<title>ToonVision: Object Detection</title>

# ToonVision - Object Detection

This article is the third in a series on **ToonVision**.
The [first article](https://fars.io/toonvision/classification/) covered the basics of classification and binary classification of Toons and Cogs.
More specifically, the previous article covered how to...

- convert a binary dataset into a multiclass dataset
- use classification performance measures such as precision, recall, and F1-score
- automatically optimize hyperparameter values with Keras-tuner
- interpret and visualize what the model is learning with confusion matrices and class activation maps

This article explains object detection of Cogs - also called entity detection.
After reading this article, you should have a better understanding of how to...

- differentiate between object detection models (YOLO, R-CNN, SSD)
- create an object detection model
- extract objects from images and videos
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
        - [Issues encountered](#issues-encountered)
            - [Input image size](#input-image-size)
            - [Accuracy or speed](#accuracy-or-speed)
            - [Unable to detect Toons](#unable-to-detect-toons)
            - [Dataset inconsistencies](#dataset-inconsistencies)
    - [Prediction results](#prediction-results)
        - [Run inference](#run-inference)
        - [Retrieve predicted bounding boxes](#retrieve-predicted-bounding-boxes)
        - [Visualize boxes and labels on the input image](#visualize-boxes-and-labels-on-the-input-image)
        - [Sample predictions](#sample-predictions)
    - [Save predicted annotations in PASCAL VOC format](#save-predicted-annotations-in-pascal-voc-format)
        - [Review, correct, and verify the predicted annotations](#review-correct-and-verify-the-predicted-annotations)
        - [Extract the annotated objects](#extract-the-annotated-objects)
    - [Build data pipelines to semi-autonomously grow a dataset](#build-data-pipelines-to-semi-autonomously-grow-a-dataset)
        - [Future pipeline enhancements](#future-pipeline-enhancements)
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

<font style="color:red">TODO: Insert image of ToonTown region proposal -> classification and regression networks</font>

Despite the slowness, two-shot models have far superior accuracy when compared to single-shot models.
R-CNN<sup>[1]</sup> is a commonly used two-shot detection model.
Faster R-CNN<sup>[2]</sup>, R-CNN's improved variant, is the more popular choice for two-shot models.

#### Single-shot

Single-shot models are designed for real-time object detection.
They have quicker inference speeds and use less resources during training than two-shot models.
These two properties allow for quick training, prototyping, and experimenting without consuming considerable computation resources.

<font style="color:red">TODO: Insert image of anchor boxes and feature maps</font>

In a single forward pass, these models predict bounding boxes and class labels directly from the input's feature maps.
They skip the region proposal stage and yield final localization and content prediction at once.
SSD<sup>[3]</sup> and YOLO<sup>[4]</sup> are popular single-shot object detection models capable of running at 5-160 frames per second!

### R-CNN

Regions with Convolutional Neural Networks (R-CNN) is a two-shot detection algorithm created in 2013.
R-CNN combines **rectangular region proposals** with **convolutional neural network features** to detect objects.
The first stage, region proposal, identifies a subset of regions in an image that might contain an object.
The second stage classifies the object in each region using a CNN classifier.
Each proposed object requires a forward pass of the classification network; as a result, the algorithm has slow inference speed.

<font style="color:red">TODO: Insert image showing bounding boxes, different region proposals, and result</font>

R-CNNs can be boiled down to the following three processes:

1. Find regions in the image that might contain an object (region proposals)
2. Extract features from the region proposals
3. Classify the objects using the extracted features

There are many variants of R-CNN: Fast R-CNN<sup>[5]</sup>, Faster R-CNN<sup>[2]</sup>, Mask R-CNN<sup>[6]</sup>.
Each variant improves performance, but the algorithm is still slow when compared to single-shot.
Mask R-CNN is unique because it's used for image segmentation while all others are for object detection.

R-The CNN algorithm has incredible accuracy in low-FPS or still-image tasks.
However, both SSD and YOLO significantly outperform R-CNN in real-time object detection.

### SSD

Developed in 2015, the Single Shot MultiBox Detector (SSD) is a method for detecting object in images using a single deep neural network.
Boxes of different aspect ratios and scales overlay each feature map.
At prediction time, the network generates scores for the presence of each object in each box.
The network combines predictions from multiple feature maps with different resolutions to handle objects of various sizes.

<font style="color:red">TODO: Insert image showing bounding boxes, different ratio boxes, and result</font>

Like all other single-shot models, SSD eliminates proposal generation and feature resampling.
All computation is done in a single network, making SSD easy to train and integrate into systems requiring a detection component.

Although SSD was state-of-the-art when it came out in 2015, there's a new king in town: YOLO.

### YOLO

The You Only Look Once (YOLO) model is a new approach to unified, real-time object detection.
Created in 2015 by Joseph Redmon and gang, YOLO reframes object detection as a regression problem rather than leveraging regional proposals and a CNN classifier.
Like all single-shot algorithms, YOLO's single neural network predicts bounding boxes and class probabilities from images in one pass.

<font style="color:red">TODO: Insert image showing bounding boxes, different ratio boxes, and result</font>

<!-- The YOLO framework has three main components:

- Backbone
- Neck
- Head

The **Backbone** mainly extracts essential features of an image and feeds them to the Head through Neck.
The **Neck** collects feature maps extracted by the Backbone and creates feature pyramids.
Finally, the **Head** consists of output layers that have final detections. -->

YOLO is insanely fast.
The base model processes images in real-time at 45 FPS.
A smaller version, Fast YOLO, processes an astounding 155 FPS!
Take a peek at YOLOv3's performance from the author's own 3-minute [YouTube video](https://www.youtube.com/watch?v=MPU2HistivI).

Where YOLO excels in speed, it struggles in accuracy.
The algorithm is prone to making localization errors (sizes and location of bounding boxes).
When comparing to state-of-the-art detection systems, however, YOLO is far less likely to predict false detections where nothing exists.

There have been **seven** iterations on the algorithm since its inception in 2015.
Each variation resulted in higher accuracy and inference speed.
The newest version, YOLOv7<sup>[7]</sup>, was released in July 2022 and is capable of 160FPS.

---

## Creating an object detection model

I'll leverage [TensorFlow's model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) to fine-tune and train a SSD model.
This article will explain the general procedures for building the model.
A rough step-by-step page can be found in my knowledge base.
In the future, I'd like to write an article about how feature extractors are created and used for smaller projects like this.

### Issues encountered

I encountered many issues that are not often discussed in articles and tutorials.
In an effort to provide a realistic overview of my process, I've explained the following issues:

1. Input image size being too large
2. Accuracy or speed for model selection
3. Model does not detect Toons
4. Bounding box inconsistencies in the dataset

#### Input image size

My main concern about this project revolved around the dataset's image sizes being too large at 3440x1440.
I trained a larger model (Faster R-CNN ResNet152) on the dataset and each training step took over 3 seconds.
It took over 90 minutes to train 1000 steps!
Even worse, the model likely would not converge until the following day; so, I stopped training.

Scaling the image sizes down by half (1720x720) resulted in faster training and inference speed.
Further scaling the images down to 1/4 the original size (860x360) led to even faster training.
The model converged in less than an hour after 50,000 training steps.
Issue #1 resolved!

#### Accuracy or speed

I had originally planned to have two model: one for real-time detection in videos (speed) and another for detection in images (accuracy).
Faster R-CNN was to be used in image detection because I wanted accuracy for the data pipeline.
SSD or YOLO for the real-time video detection.
Given the large size and lengthy training process of two-shot models, I've scrapped the idea of two models in favor of a single SSD model.
Issue #2 resolved!

#### Unable to detect Toons

The trained SSD model does not detect Toons.
In fact, the Toons it does detect are classified as Cogs.
This issue is largely due to the massive dataset imbalance: 526 Cog samples and 148 Toon samples.
There are two solutions: Increase weights of Toon localization and classification during training or modify entire dataset to include only Cogs.

<font style="color:red">TODO: Insert image of undetected or wrongly detected Toon</font>

Recall that the goal of ToonVision is for a Toon to see Cogs.
Given that it's more important to detect Cogs, I will exclude Toons from the dataset.
I generated new TensorFlow record files which consist purely of Cogs and excluded any images that contained only Toons.
Issue #3 resolved!

#### Dataset inconsistencies

This is probably an uncommon issue.
When creating the binary and multi-class classification models, I opted to only include clear, non-obstructed Cog and Toon samples.
This means I did not put bounding boxes on entities that were occluded by another object.
However, the trained SSD model detects and classifies occluded Cogs!

<font style="color:red">TODO: Insert image of sample vs predictions</font>

*Why is it bad for the model to detect objects that I did not classify in the training set?*
It negatively affects the training loss.
More specifically, the localization loss increases during training.

*What's causing the loss increase?*
During training, the localization loss is calculated in part by how accurate the model's predicted bounding boxes compares to the ground truth bounding boxes.
In object detection terms, the bounding box accuracy is called the **Intersection over Union** (IoU).
IoU scores the overlap of the predicted box and the ground truth box.
The higher the IoU score, the higher the accuracy.
If there's no ground truth box, however, the IoU score will be zero and training loss will increase.

<font style="color:red">TODO: Insert image of IoU</font>

*How does this inconsistency affect training?*
The inconsistency did not affect the model's performance, but it decreased training performance and convergence.

<font style="color:red">TODO: Insert Tensorboard loss graphs</font>

*How can I resolve the issue?*
I would have to go through the dataset and label the non-labeled Cogs.
Alternatively, I could run the entire dataset through the model and compare the number of detected objects against the ground truth.
If there's a discrepancy, I can manually review the ground truth labels and correct them if needed.

In short, the model localizes objects that I did not declare as ground truth.
It slowed down training but did not affect performance too much.
Issue #4 explained, but unresolved!

---

## Prediction results

Enough about my issues, let's take a look at the model's predictions.
But first I need functions to run inference, retrieve the predicted bounding boxes, and visualize boxes and labels on the input image.

### Run inference

First load the image into a numpy array.

```python
def load_image_into_numpy_array(path, resize: tuple[int, int] = None):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path (this can be local or on colossus)

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, "rb").read()
    image = Image.open(BytesIO(img_data))
    if resize:
        image = image.resize(resize)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
```

```python
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop("num_detections"))
    output_dict = {
        key: value[0, :num_detections].numpy() for key, value in output_dict.items()
    }
    output_dict["num_detections"] = num_detections

    # detection_classes should be ints.
    output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)

    # Handle models with masks:
    if "detection_masks" in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict["detection_masks"],
            output_dict["detection_boxes"],
            image.shape[0],
            image.shape[1],
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return output_dict
```

### Retrieve predicted bounding boxes

```python
def get_highest_scoring_boxes(output_dict: dict, score_threshold: float = 0.5):
    # Output scores are sorted by descending values
    scores = np.array(output_dict["detection_scores"])
    scores = scores[scores > score_threshold]
    num_detections = len(scores)

    if num_detections > 0:
        scores = output_dict["detection_scores"][:num_detections]
        boxes = output_dict["detection_boxes"][:num_detections]
        classes = output_dict["detection_classes"][:num_detections]
        classes_str = np.array(
            [category_index[class_id]["name"] for class_id in classes]
        )

    return scores, boxes, classes, classes_str


def normal_box_to_absolute(image_np: np.ndarray, box: list[float]):
    """Generate absolute box coordinates from relative box coordinates

    Args:
        image_np (np.ndarray): Numpy array containing an image, from `load_image_into_numpy_array`
        box (list[float]): List containing the relative [ymin, xmin, ymax, xmax] values
            Values must be in range [0, 1]

    Returns:
        list[float]: List containing absolute coordinate values
    """
    y, x, _ = image_np.shape
    if box.ndim > 1:  # Input is a list of bounding boxes
        boxes = []
        for b in box:
            ymin, xmin, ymax, xmax = b
            boxes.append(
                np.array([y * ymin, x * xmin, y * ymax, x * xmax]).astype("int")
            )
        return np.array(boxes)
    else:  # Input is a single bounding box
        ymin, xmin, ymax, xmax = box
        return np.array([y * ymin, x * xmin, y * ymax, x * xmax]).astype("int")
```

### Visualize boxes and labels on the input image

```python
from object_detection.utils import visualization_utils as vis_util
img_path_test = "./tensorflow/workspace/training_demo/images/label"

img_fps = glob.glob(f"{img_path_test}/*.png")
# shuffle(img_fps)
for image_path in img_fps:
    # Create downscaled image array to decrease inference times
    image_np_downscale = load_image_into_numpy_array(image_path, resize=(1720, 720))
    output_dict = run_inference_for_single_image(model, image_np_downscale)
    img = vis_util.visualize_boxes_and_labels_on_image_array(
        image=image_np_downscale,
        boxes=output_dict["detection_boxes"],
        classes=output_dict["detection_classes"],
        scores=output_dict["detection_scores"],
        category_index=category_index,
        instance_masks=output_dict.get("detection_masks_reframed", None),
        use_normalized_coordinates=True,
        line_thickness=10,
    )
    # Create original size image array for accurate absolute bounding boxes annotations
    image_np = load_image_into_numpy_array(image_path)
    scores, boxes, classes, classes_str = get_highest_scoring_boxes(output_dict)
    # Create pascal VOC writer
    writer = Writer(path=image_path, width=image_np.shape[1], height=image_np.shape[0])
    # Add objects (class, xmin, ymin, xmax, ymax)
    for class_str, box in zip(classes_str, boxes):
        ymin, xmin, ymax, xmax = normal_box_to_absolute(image_np, box)
        writer.addObject(class_str, xmin, ymin, xmax, ymax)
    fn_xml = image_path.split("\\")[-1].replace(".png", ".xml")
    path_xml = f"{img_path_test}/{fn_xml}"
    writer.save(annotation_path=path_xml)

    print(image_path)
```

### Sample predictions

<font style="color:red">TODO: Insert image of sample predictions with bounding boxes</font>

---

## Save predicted annotations in PASCAL VOC format

Utilize a small Python package for reading and writing PASCAL VOC annotations.
Save annotations in the same directory as the sample image.

```python
from pascal_voc_writer import Writer

# Create pascal VOC writer
writer = Writer(path=image_path, width=image_np.shape[1], height=image_np.shape[0])

# Add objects (class, xmin, ymin, xmax, ymax)
for class_str, box in zip(classes_str, boxes):
    ymin, xmin, ymax, xmax = normal_box_to_absolute(image_np, box)
    writer.addObject(class_str, xmin, ymin, xmax, ymax)

fn_xml = image_path.split("\\")[-1].replace(".png", ".xml")
path_xml = f"tensorflow/workspace/training_demo/images/label/{fn_xml}"
writer.save(annotation_path=path_xml)
```

### Review, correct, and verify the predicted annotations

Not all of the model's predicted annotations will be accurate or correct.
I have to manually review all annotations, fix the bounding boxes, annotate the missing objects, and verify the image.
All of this is completed with [labelimg](https://github.com/heartexlabs/labelImg).

Once the annotations are verified, we can move the images to the "unprocessed" directory to extract the objects for our other classification models.

### Extract the annotated objects

Insert code snippets to extract objects from bounding boxes

---

## Build data pipelines to semi-autonomously grow a dataset

The new pipeline looks like this:

1. Take screenshot
1. Save to correct directory within `toonvision/img/raw/screenshots/`
1. Load model
1. Run inference
1. Save predicted annotations in PASCAL VOC format in same directory as its respective sample image
1. Manually review, correct, and verify the predicted annotations
1. Move verified images and their annotations to the `processed` directory

New process saves me time by assisting with the dataset annotation process.

### Future pipeline enhancements

- Run the extracted objects through a suit and name classifier to enhance the label from "cog" to "cog_bb_flunky"
    - Alternatively, create a multilabel classifier (suit, name, playground) and store in metadata file
    - Raise an error/alert me if the suit and name mismatch
    - Manually label the image
- Create metadata about the image
    - Number of Cogs
    - Number of Toons
    - Street name
    - Playground
    - Cog names and counts
    - Cog suits and counts
- Store metadata in SQLite database
- Visualize dataset over time to see how it grows

---

## References

1. Rich feature hierarchies for accurate object detection and semantic segmentation, [https://arxiv.org/abs/1311.2524](https://arxiv.org/abs/1311.2524)

2. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, [https://arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497)

3. SSD: Single Shot MultiBox Detector, [https://arxiv.org/abs/1512.02325](https://arxiv.org/abs/1512.02325)

4. You Only Look Once: Unified, Real-Time Object Detection, [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)

5. Fast R-CNN, [https://arxiv.org/abs/1504.08083](https://arxiv.org/abs/1504.08083)

6. Mask R-CNN, [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)

7. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors, [https://arxiv.org/abs/2207.02696](https://arxiv.org/abs/2207.02696)
