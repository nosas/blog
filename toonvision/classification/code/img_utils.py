# %% Import libraries
import cv2
import xml.etree.ElementTree as ET
from os import path


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


def get_save_directory(obj_name: str) -> str:
    """Given an object name, return the save directory for the object.

    Args:
        obj_name (str): Name of the object

    Returns:
        str: Save directory for the object
    """
    if obj_name.startswith("cog_"):
        dir = "cog/"
    elif obj_name.startswith("toon_"):
        dir = "toon/"
    else:
        dir = "unknown/"
    return dir


def save_objects_to_img(objs_from_img: tuple[str, list], save_path: str):
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
