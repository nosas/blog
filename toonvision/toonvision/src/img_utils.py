# %% Import libraries
import re
import xml.etree.ElementTree as ET
from glob import glob
from os import path

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import layers


# %% Define utility functions
def extract_objects_from_xml(xml_path: str) -> list[tuple[str, int, int, int, int]]:
    """Extract objects from XML file.


    Given a path to an XML file containing objects' bounding box
    dimensions, extract the objects and return a list of images.

    Args:
        xml_path (str): Absolute path to the XML file

    Returns:
        list[tuple[str, int, int, int, int]]: List of tuples containing obj name, xmin, ymin, xmax, ymax
    """
    # Load XML file
    with open(xml_path, "r") as xml_file:
        xml_str = xml_file.read()
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


def extract_objects_from_img(
    img_path: str, objs_from_xml: tuple[str, int, int, int, int]
) -> list[tuple[str, list]]:
    """Extract objects from an image given a tuple of objects `from extract_object_from_xml()`

    Given a path to an image and objects' bounding box dimensions, extract the objects from the
    image and return a list of images.

    Args:
        img_path (str): Absolute path to the image
        objs_from_xml (tuple): Tuple containing obj name, xmin, ymin, xmax, ymax

    Returns:
        list[tuple[str, list]]: List of tuples containing obj name and obj img
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


def get_obj_name_from_filepath(filepath: str, file_ext: str = "png") -> str:
    """Given a filepath, return the full object name

    Args:
        filepath: Path to a file
        file_ext: File extension

    Returns:
        str: Full object name (ex: "cog_bb_flunky_1", "toon_cat_32")
    """
    regex_obj_name = re.compile(rf"((cog|toon)_.*)(\.{file_ext})?$")
    try:
        return regex_obj_name.search(filepath).group(0)
    except AttributeError as e:
        print("Could not find object name in filepath:", filepath)
        raise e


def get_obj_details_from_name(obj_name: str) -> dict:
    """Given an object name, return the object's binary label and class-specific details

    Args:
        obj_name: Full object name (ex: "cog_bb_flunky_1", "toon_cat_32")

    Returns:
        dict: Object details

    Example:
        >>> get_obj_details_from_name("cog_bb_flunky_1")
        {'binary': 'cog', 'suit': 'bb', 'name': 'flunky', 'animal': None, 'index': '1'}
        >>> get_obj_details_from_name("toon_cat_32")
        {'binary': 'toon', 'suit': None, 'name': None, 'animal': 'cat', 'index': '32'}
        >>> get_obj_details_from_name("cog_lb_backstabber")
        {'binary': 'cog', 'suit': 'lb', 'name': 'backstabber', 'animal': None, 'index': None}
    """
    # https://regex101.com/r/0UbU06/1
    regex_details = re.compile(
        r"""
    (?P<binary>cog|toon)_
    (
        ((?P<suit>bb|cb|lb|sb)_(?P<name>[a-zA-Z]+)) |
        (?P<animal>[a-zA-Z]+)
    )
    (
        _
        (?P<index>\d+)
        (?P<file_ext>\.png)?
    )?
    """,
        re.VERBOSE,
    )
    res = regex_details.search(obj_name)
    if res is None:
        raise ValueError(f"Could not parse object name: {obj_name}")
    details = res.groupdict()
    return details


def get_obj_details_from_filepath(filepath) -> dict:
    obj_name = get_obj_name_from_filepath(filepath)
    details = get_obj_details_from_name(obj_name)
    return details


def save_objects_to_img(
    objs_from_img: tuple[str, list],
    save_path: str,
    data_path: str,
    verbose: bool = False,
):
    """Save objects to an image given a tuple of objects `from extract_object_from_img()`

    Given a tuple of objects and a path to save the objects to, save the objects to the
    image.

    Args:
        objs_from_img (tuple): Tuple containing obj name, obj img
        save_path (str): Absolute path of parent directory to save the objects to, typically UNSORTED_DIR
        data_path (str): Absolute path of the data directory, typically DATA_DIR
    """
    for obj in objs_from_img:
        obj_name, obj_img = obj
        # save_dir can be "cog/", "toon/"
        obj_details = get_obj_details_from_name(obj_name)
        toon_or_cog = obj_details["binary"]

        sorted_obj_fps = glob(f"{data_path}/**/{obj_name}_*.png", recursive=True)
        unsorted_obj_fps = glob(f"{save_path}/{toon_or_cog}/{obj_name}_*.png")
        obj_fps = sorted_obj_fps + unsorted_obj_fps
        obj_fps = [fp.split("_")[-1] for fp in obj_fps]
        max_img_num = len(obj_fps)

        filename = f"{obj_name}_{max_img_num}.png"
        filepath = f"{save_path}/{toon_or_cog}/{filename}"
        if filename in obj_fps:
            print("    ERROR: Filename exists: ", filename)
        elif path.exists(filepath):
            # img_name can be "cog_<bb|cb|sb|lb>_<cog_name>_<img_num>.png"
            # img_name can be "toon_<animal>_<img_num>.png"
            print("    ERROR: Path exists: ", filepath)
        else:
            # Save image to filepath
            if verbose:
                print(f"    Saving {obj_name} to {filepath}")
            cv2.imwrite(filepath, obj_img)


def get_img_array_from_filepath(
    img_path: str, target_size: tuple[int, int, int] = (600, 200, 3)
) -> np.array:
    """Return the image array from the image's filepath.

    Args:
        img_path (str): Absolute filepath to an image
        target_size (np.array): Eg. (600, 200, 3) for 600h, 200w, 3ch

    Returns:
        np.array: Array representation of an image
    """
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    array = tf.keras.utils.img_to_array(img)
    return array


def get_image_augmentations() -> keras.Sequential:
    return keras.Sequential(
        [
            # Apply horizontal flipping to 50% of the images
            layers.RandomFlip("horizontal"),
            # Rotate the input image by some factor in range [-7.5%, 7.5%] or [-27, 27] in degrees
            layers.RandomRotation(0.075),
            # Zoom in or out by a random factor in range [-20%, 20%]
            layers.RandomZoom(0.2),
        ]
    )
