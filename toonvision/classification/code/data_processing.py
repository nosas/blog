# %% Imports
from img_utils import (
    extract_objects_from_xml,
    extract_objects_from_img,
    save_objects_to_img,
)
from os import rename
from glob import glob

# %% Test functions with
IMG_DIR = "../img"
RAW_DIR = IMG_DIR + "/raw"
DATA_DIR = IMG_DIR + "/data"
UNSORTED_DIR = IMG_DIR + "/unsorted"
PROCESSED_DIR = RAW_DIR + "/processed"


def verify_folder_structure():
    """Verify that the folder structure exists and is correct

    :: Expected folder structure
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
        │   ├───data
        │   └───processed
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
def process_images(raw_images_dir: str = RAW_DIR, image_type: str = "png") -> None:
    for img_path in glob(f"{raw_images_dir}/*.{image_type}"):
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
