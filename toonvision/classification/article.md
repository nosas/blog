# ToonVision - Classification

This article is first in a series on **ToonVision**.

ToonVision is my personal computer vision project to teach a machine how to see in [ToonTown Online](https://en.wikipedia.org/wiki/Toontown_Online) - an MMORPG created by Disney in 2002.
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
        - [Data Acquisition](#data-acquisition)
            - [Can we use GANs to synthesize additional data?](#can-we-use-gans-to-synthesize-additional-data)
        - [Data Labeling](#data-labeling)
        - [Data Processing](#data-processing)

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

There are 11 number of animals:

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

### Cog

There are 4 Cog suits: Bossbot, Lawbot, Cashbot, and Sellbot.
Each suit has 8 Cogs, totaling 32 unique Cogs.

### Why is it important for Toons to classify Cogs?

ToonTasks, object avoidance

---
## Creating the dataset

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
│   └───processed
└───unsorted
    ├───cog
    └───toon
```

There's no need for a unique folder for each Cog suit because we can filter on the filename.

### Data Acquisition

Walk around TT, take screenshots, and save them to the raw folder.

#### Can we use GANs to synthesize additional data?

Yes, iff there was a GAN that could generate Toons and Cogs.
I'm not if it exists.
Also, the data is easy enough to gather manually by walking around TT.

### Data Labeling

Use [labelimg](https://github.com/tzutalin/labelImg) to create labeled bounding boxes around Toons and Cogs.
Labels follow the format:

- `cog_<bb|lb|cb|sb>_<name>`
- `toon_<animal>`

The cog labels contain shorthand notation (`<bb|lb|cb|sb>`) for each suit: BossBot, Lawbot, Cashbot, and Sellbot, respectively.
This shorthand notation will allow us to filter cog data by filename and create a classifier that can distinguish between the 4 suits.

Bounding boxes are saved in XML format - specifically [Pascal VOC XML](https://mlhive.com/2022/02/read-and-write-pascal-voc-xml-annotations-in-python) - alongside the image in the `raw` folder.

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
│   │   labelImg.exe
│   │   sample_img0.png
│   │   sample_img0.xml
│   │   sample_img1.png
│   │   sample_img1.xml
│   │
│   └───processed
```

### Data Processing

The raw data is passed into the `data_processing.py` script.
The script utilizes functions in `img_utils.py` to extract objects from the images using the bounding boxes and labels found in their corresponding XML files.
Specifically, the workflow is as follows:

- Acquire bounding box dimensions and labels from the XML files
- Extract object (Toon or Cog) from the image using the dimensions
- Save the cropped image of the object to the `unsorted` folder
- Move the raw image and its corresponding XML file to the `processed` folder

```python
# %% Convert raw images to data images
for img_path in glob(f"{RAW_DIR}/*.png"):
    print(f"Processing {img_path}")
    xml_path = img_path.replace(".png", ".xml")
    # Extract objects' bounding box dimensions from XML
    objs_from_xml = extract_objects_from_xml(xml_path)
    # Extract objects from images
    objs_from_img = extract_objects_from_img(img_path, objs_from_xml)
    # Save extracted objects to images
    save_objects_to_img(objs_from_img, UNSORTED_DIR)
    # Move raw image to processed directory
    for f in [img_path, xml_path]:
        new_path = f.replace(RAW_DIR, PROCESSED_DIR)
        print(f"    Moving {f} to {new_path}")
        rename(f, new_path)
```