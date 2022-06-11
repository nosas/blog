<title>ToonVision: Binary and Multiclass Classification</title>

# ToonVision - Classification


This article is first in a series on **ToonVision**.

ToonVision is my personal computer vision project for teaching a machine how to see in [ToonTown Online](https://en.wikipedia.org/wiki/Toontown_Online) - an MMORPG created by Disney in 2002.
The ultimate goal is to teach a machine (nicknamed **OmniToon**) how to play ToonTown and create a self-sustaining ecosystem within the game where the bots progress through the game together.

In later articles, we'll dive into image segmentation and object detection.
For now, let's focus on real-time classification.

This article covers ...

- Binary classification: Toon vs Cog
- Multiclass classification: Cog suits (4 unique suits) and Cog names (32 unique names)


<details>
    <summary>Table of Contents</summary>

- [ToonVision - Classification](#toonvision---classification)
    - [Classification](#classification)
        - [Binary classification](#binary-classification)
        - [Multiclass classification](#multiclass-classification)
            - [Multiclass multilabel classification](#multiclass-multilabel-classification)
    - [ToonTown](#toontown)
        - [Toon](#toon)
        - [Cog](#cog)
        - [Why is it important for Toons to classify Cogs?](#why-is-it-important-for-toons-to-classify-cogs)
    - [The ToonVision dataset](#the-toonvision-dataset)
        - [Dataset considerations](#dataset-considerations)
        - [Filename and data folder structure](#filename-and-data-folder-structure)
        - [Data acquisition](#data-acquisition)
            - [Can we use GANs to synthesize additional data?](#can-we-use-gans-to-synthesize-additional-data)
        - [Data labeling](#data-labeling)
        - [Data extraction](#data-extraction)
        - [Data processing](#data-processing)
        - [Creating the datasets](#creating-the-datasets)
            - [Spitting the images into train, validate, and test](#spitting-the-images-into-train-validate-and-test)
    - [Compiling the model](#compiling-the-model)
        - [Loss function](#loss-function)
        - [Optimizer](#optimizer)
        - [Metrics](#metrics)
        - [Defining the model](#defining-the-model)
    - [Training the simple, baseline model](#training-the-simple-baseline-model)
    - [Training the model](#training-the-model)
        - [Preventing overfitting](#preventing-overfitting)
        - [Callbacks](#callbacks)
        - [Loss and accuracy plots](#loss-and-accuracy-plots)
        - [Model evaluation](#model-evaluation)
        - [Accuracy on never-before-seen images](#accuracy-on-never-before-seen-images)

</details>

---
## Classification

### Binary classification

### Multiclass classification

#### Multiclass multilabel classification

We could beef up the multiclass classification to a multilabel multiclass classification: Cog state/level/hp/name/suit.
But this adds unneeded complexity.
Let's keep it simple.
In the future, I will surely add the classification of the Cog's state: battle, patrolling, spawning, de-spawning, etc.

---
## ToonTown

### Toon

There are 11 unique animals:

- bear
- cat
- crocodile
- deer
- dog
- duck
- horse
- monkey
- mouse
- pig
- rabbit

Each animal can have a unique head shape, body length, and height.
Furthermore, each animal can have mismatching colors for its head, arms, and legs.

<font style="color:red">TODO: Image grid of animal with various sizes and colors</font>

### Cog

There are 4 Cog suits: Bossbot, Lawbot, Cashbot, and Sellbot.
Each suit has 8 Cogs, totaling 32 unique Cogs.

### Why is it important for Toons to classify Cogs?

ToonTasks, object avoidance

---
## The ToonVision dataset

There doesn't exist a dataset for ToonVision, so I'll be creating one from scratch.
The following sections will explain my process and results.

### Dataset considerations

- Images are split into training, validation, and test sets: 40% training, 20% validation, and 40% test.
- Images of Toons and Cogs must be...
    - Taken at various distances from each street, not playground
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
├───data
│   ├───test
│   │   ├───cog
│   │   └───toon
│   ├───train
│   │   ├───cog
│   │   └───toon
│   └───validate
│       ├───cog
│       └───toon
├───raw
│   ├───processed
│   └───screenshots
└───unsorted
    ├───cog
    └───toon
```

There's no need for a unique folder for each Cog suit because we can filter on the filename.

### Data acquisition

Acquiring data is simple: Walk around TT streets, take screenshots, and save them to the raw folder.
It's important to take screenshots from various distance and of different angles of each entity: front, back, and side.
Taking screenshots from up close is preferred.
When taken from far away, the entity's nametag covers the entity's head, thus causing us to crop the entity's head or include the nametag - neither are good options.

<font style="color:red">TODO: Insert example screenshot</font>

There were a few difficulties with acquiring data:

1. Entities are always moving unless in battle
1. Entities often obstruct other entities, which makes for less than ideal training data
1. Finding the desired entity is purely a matter of walking around the street and looking for the entity, there's no precision radar

These difficulties result in an imbalanced dataset that will be improved over time.

<font style="color:red">TODO: Insert dataset barchart</font>

#### Can we use GANs to synthesize additional data?

Yes, iff there was a GAN that could generate Toons and Cogs.
As far as I know, no GAN exists for generating ToonTown entities; perhaps I can take a swing at it later.

### Data labeling

I'm using [labelimg](https://github.com/tzutalin/labelImg) to draw labeled bounding boxes around Toons and Cogs.
Labels - also referred to as `obj_name` - follow the format:

- `cog_<bb|lb|cb|sb>_<name>_<index>`
- `toon_<animal>_<index>`

The cog labels contain shorthand notation (`<bb|lb|cb|sb>`) for each suit: Bossbot, Lawbot, Cashbot, and Sellbot, respectively.
This shorthand notation allows us to filter cog data by filename and create a classifier that can distinguish between the 4 suits.

Bounding boxes are saved in XML format - specifically [Pascal VOC XML](https://mlhive.com/2022/02/read-and-write-pascal-voc-xml-annotations-in-python) - alongside the image in the `raw/screenshots` directory.

```
img
├───data
│   ├───test
│   │   ├───cog
│   │   └───toon
│   ├───train
│   │   ├───cog
│   │   └───toon
│   └───validate
│       ├───cog
│       └───toon
├───raw
│   ├───processed
│   └───screenshots
│       ├   sample_img0.png
│       ├   sample_img0.xml
│       ├   sample_img1.png
│       └   sample_img1.xml
└───unsorted
    ├───cog
    └───toon
```

<font style="color:red">TODO: Insert image of labelimg and bounding boxes</font>

How the objects are labeled - how the bounding boxes are drawn - determines how the object will be extracted from the image.
It's important to draw bounding boxes such that the entity is snugly contained within the bounding box
Furthermore, we must exclude entity nametags in the bounding box because the classifier will learn to "cheat" by identifying objects from their nametag rather than features of the entity itself.

### Data extraction

The raw data (screenshot) is passed into the `data_processing.py` script.
The script utilizes functions in `img_utils.py` to extract objects from the images using the labeled bounding boxes found in the image's corresponding XML files.
Specifically, the data extraction workflow is as follows:

- Acquire bounding box dimensions and labels from the XML files
- Extract object (Toon or Cog) from the image using the dimensions and labels found in the XML files
- Save the cropped image of the object to the `img/unsorted` folder
- Move the raw image and its corresponding XML file to the `raw/processed` folder

*Why move the cropped image to unsorted and then processed?*

The unsorted images directory is used to maintain a counter (referred to as an index) for each label.
It gives me a glimpse of how many images are in each category by looking at the filenames in the `unsorted` directory.
If I want to add more images to the dataset, I would have place images from all datasets back into the `unsorted` directory in order to maintain the counter and avoid overwriting existing images.

Given that my dataset is so small, I can unsort and re-sort the images with ease.
But this is not at all scalable in the future and I will surely redesign this portion of the data pipeline.

```python
# %% Convert raw images to data images
def process_images(
    raw_images_dir: str = SCREENSHOTS_DIR,
    image_type: str = "png",
    move_images: bool = False,
) -> None:
    """Extract objects from raw images and save them to the unsorted img directory"""
    screenshots = glob(f"{raw_images_dir}/*.{image_type}", recursive=True)
    print(f"Found {len(screenshots)} screenshots")
    for img_path in screenshots:
        print(f"Processing {img_path}")
        xml_path = img_path.replace(f".{image_type}", ".xml")
        if path.exists(xml_path):
            # Extract objects' labels and bounding box dimensions from XML
            objs_from_xml = extract_objects_from_xml(xml_path)
            # Extract objects from images using XML data
            objs_from_img = extract_objects_from_img(img_path, objs_from_xml)
            # Save extracted objects to images, modify image name to include object index
            save_objects_to_img(objs_from_img, UNSORTED_DIR)
            # Move raw image to processed directory
            if move_images:
                for f in [img_path, xml_path]:
                    new_path = f.replace(raw_images_dir, PROCESSED_DIR)
                    print(f"    Moving {f} to {new_path}")
                    rename(f, new_path)
        else:
            print(f"    No XML file found for {img_path}")
```

### Data processing

The extracted objects are of various sizes because the screenshots were taken from various angles and distances.
Large objects are a result of the screenshot taken from up close, while small objects are a result of the screenshot taken from far away.

Overall, it would be ideal for the dataset to consist mostly of large, close-up objects because they contain more information about the object.
Small, far-away objects lose information about the object and are not as useful for training - we could simulate this loss of information through image augmentation by blurring the image.
Occlusion - an object is is a common problem with moving objects

<font style="color:red">TODO: Insert image comparing a small and large object</font>


### Creating the datasets

After the objects are extracted and placed in the `unsorted` folder, we can create the datasets.
First, we need to create a balanced datasets within the `data/[train|validate|test]` folders.
Remember that we're aiming for a 40/20/40 split of the dataset for training, validation, and testing, respectively.

#### Spitting the images into train, validate, and test

Before creating the datasets, we need to move images from `unsorted/[cog|toon]` to `data/[train|validate|test]/[cog|toon]`.

```python
def split_data(dry_run: bool = False):
    """Split the data into train(40%)/validate(20%)/test(40%) data sets"""
    for unsorted_dir in [UNSORTED_COG_DIR, UNSORTED_TOON_DIR]:
        cog_or_toon = unsorted_dir.split("/")[-1]
        # Get all images from unsorted_dir
        unsorted_images = glob(f"{unsorted_dir}/*.png")
        num_images = len(unsorted_images)

        # Split images into train/validate/test sets
        num_train = int(num_images * 0.4)
        num_validate = int(num_images * 0.2)
        num_test = num_images - num_train - num_validate
        print(num_train, num_validate, num_test)

        # # Shuffle filenames to randomize the order of the images
        shuffle(unsorted_images)
        train = unsorted_images[:num_train]
        validate = unsorted_images[num_train:-num_test]
        test = unsorted_images[-num_test:]

        # Move images to train/validate/test directories
        for images, dir_name in zip([train, validate, test], [TRAIN_DIR, VALIDATE_DIR, TEST_DIR]):
            for img_path in images:
                new_path = img_path.replace(unsorted_dir, f"{dir_name}/{cog_or_toon}")
                if dry_run:
                    print(f"Moving {img_path} to {new_path}")
                else:
                    rename(img_path, new_path)
```

We can visualize the balances of the datasets using the `plot_all_datasets` function in `data_visualization.py`.

<font style="color:red">TODO: Insert image of dataset balance</font>

The creation of datasets is straight-forward using keras:

```python
# %% Create datasets
from tensorflow.keras.image_dataset_from_directory

train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(600, 200),
    # batch_size=16
)
validation_dataset = image_dataset_from_directory(
    VALIDATE_DIR,
    image_size=(600, 200),
    # batch_size=16
)
test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=(600, 200),
    # batch_size=16
)
```

<font style="color:red">TODO: Insert sample images from train dataset</font>

## Compiling the model

Now that we've created the datasets, we can compile the model.
Compiling the model requires choosing a loss function, optimizer, and metrics to monitor the model's performance during training.

### Loss function

Our first model is classifying between two classes, therefore we'll use the `binary_crossentropy` loss function.
The later models will be more complex - classifying 4 or 32 classes - so we'll use the `categorical_crossentropy` loss function.

### Optimizer

There are a few guidelines to choosing an optimizer for classification problems.
I visualized the process by plotting optimizers' performances using the same model, dataset, hyperparameters, and number of epochs.

<font style="color:red">TODO: Insert plot</font>

A handful of the optimizers' losses flattened over the course of training as a result of a low learning rate or *vanishing gradients*.
SGD commonly encounters this problem, and it's often due to a low learning rate.

I increased the learning rate for all flattened optimizers and plotted the loss scores and accuracies again.

<font style="color:red">TODO: Insert plot with non-equal learning rates</font>

I could use additional callbacks, such as the learning rate scheduler, to gradually decrease the learning rate and improve the model's performance
Even more, adding *momentum* to the SGD optimizer could help the model reach global loss minimums and learn more effectively.
Alternatively, I could read an article which discusses the points above and use their suggestions to save time, but where's the fun in that?

### Metrics

It's a classification model, so we'll use the `binary_accuracy` metric.
For the later models, we'll use the `categorical_accuracy` metric.
That's about it.

### Defining the model

## Training the simple, baseline model

Before training the actual model, we need to define a simple baseline to beat.

## Training the model

### Preventing overfitting

Given that we have a small dataset, we can utilize a few techniques during training to prevent overfitting:

* **Data augmentation** - we can augment the images by randomly rotating, flipping, and cropping them.
* **Data balancing** - we can balance the datasets by balancing the number of objects in each dataset.
* **Dropout** - we can dropout some of the nodes in the model to prevent overfitting.
* **Regularization** - we can regularize the model by adding a penalty to the loss function.
* **Small learning rate** - we can use a small learning rate to prevent overfitting.
* **Reducing number of parameters** - too many parameters can cause overfitting.
* **Early stopping** - we can stop training the model if the model doesn't improve after a certain number of epochs.

### Callbacks

### Loss and accuracy plots

### Model evaluation

### Accuracy on never-before-seen images
