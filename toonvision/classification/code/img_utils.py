# %% Import libraries
import cv2
import xml.etree.ElementTree as ET
from os import path, rename
from glob import glob


# %% Define utility functions
def extract_objects_from_xml(xml_path) -> tuple[str, int, int, int, int]:
    """Extract objects from XML file.


    Given a path to an XML file containing objects' bounding box
    dimensions, extract the objects and return a list of images.

    Args:
        xml_path (str): Absolute path to the XML file
    """
    # Load XML file
    xml_file = open(xml_path)
    xml_str = xml_file.read()
    xml_file.close()
    # Parse XML file
    tree = ET.fromstring(xml_str)
    objects = []
    for obj in tree.findall("object"):
        obj_name = obj.find("name").text
        obj_bndbox = obj.find("bndbox")
        obj_xmin = int(obj_bndbox.find("xmin").text)
        obj_ymin = int(obj_bndbox.find("ymin").text)
        obj_xmax = int(obj_bndbox.find("xmax").text)
        obj_ymax = int(obj_bndbox.find("ymax").text)
        objects.append((obj_name, obj_xmin, obj_ymin, obj_xmax, obj_ymax))
    return objects


def extract_objects_from_img(img_path, objs_from_xml) -> tuple[str, list]:
    """Extract objects from an image given a tuple of objects `from extract_object_from_xml()`

    Given a path to an image and objects' bounding box dimensions, extract the objects from the
    image and return a list of images.

    Args:
        img_path (str): Absolute path to the image
        objs_from_xml (tuple): Tuple containing obj name, xmin, ymin, xmax, ymax
    """
    # Load image
    img = cv2.imread(img_path)
    # Extract objects
    objects = []
    for obj in objs_from_xml:
        obj_name, obj_xmin, obj_ymin, obj_xmax, obj_ymax = obj
        obj_img = img[obj_ymin:obj_ymax, obj_xmin:obj_xmax]
        objects.append((obj_name, obj_img))
    return objects


def get_save_directory(obj_name):
    """Get the save directory for the object

    Given an object name, return the save directory for the object.

    Args:
        obj_name (str): Name of the object
    """
    if "cog" in obj_name:
        dir = "cog/"
    elif "toon" in obj_name:
        dir = "toon/"
    else:
        dir = "unknown/"
    return dir


def save_objects_to_img(objs_from_img, save_path):
    """Save objects to an image given a tuple of objects `from extract_object_from_img()`

    Given a tuple of objects and a path to save the objects to, save the objects to the
    image.

    Args:
        objs_from_img (tuple): Tuple containing obj name, obj img
        save_path (str): Absolute path of parent directory to save the objects to
    """
    for obj in objs_from_img:
        obj_name, obj_img = obj
        # save_dir can be "cog/", "toon/", or "unknown/"
        save_dir = get_save_directory(obj_name)
        # ! There must be a better way than this while loop
        img_num = 0
        filepath = f"{save_path}/{save_dir}/{obj_name}_{img_num}.png"
        while path.exists(filepath):
            # img_name can be "cog_<bb|cb|sb|lb>_<cog_name>_<img_num>.png"
            # img_name can be "toon_<animal>_<img_num>.png"
            # img_name can be "unknown_<img_num>.png"
            img_num += 1
            filepath = filepath.replace(
                f"{obj_name}_{img_num-1}", f"{obj_name}_{img_num}"
            )
        # Save image to filepath
        print(f"    Saving {obj_name} to {filepath}")
        cv2.imwrite(filepath, obj_img)


# %% Test functions with
IMG_DIR = "../img"
RAW_DIR = IMG_DIR + "/raw"
DATA_DIR = IMG_DIR + "/data"
UNSORTED_DIR = IMG_DIR + "/unsorted"
PROCESSED_DIR = RAW_DIR + "/processed"


def verify_folder_structure():
    """Verify that the folder structure exists and is correct

    :: Expected folder structure
        ├───data
        │   ├───cog
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
        │   └───data
        └───unsorted
            ├───cog
            └───toon
    """
    # NOTE: Skeleton function
    print("Verifying folder structure...")
    print(f"Raw directory: {RAW_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Unsorted directory: {UNSORTED_DIR}")
    print("Done!")


# %% Convert raw images to data images
for img_path in glob(f"{RAW_DIR}/*.png"):
    print(f"Processing {img_path}")
    xml_path = img_path.replace(".png", ".xml")
    # Extract objects' bounding box dimensions from XML
    objs_from_xml = extract_objects_from_xml(xml_path)
    # Extract objects from images
    objs_from_img = extract_objects_from_img(img_path, objs_from_xml)
    # Save extracted objects to images
    # save_objects_to_img(objs_from_img, UNSORTED_DIR)
    # Move raw image to processed directory
    for f in [img_path, xml_path]:
        new_path = f.replace(RAW_DIR, PROCESSED_DIR)
        print(f"    Moving {f} to {new_path}")
        rename(f, new_path)

# %% Sort images into train/validate/test split with 40%/20%/40%
