# %% Imports
from img_utils import (
    extract_objects_from_xml,
    extract_objects_from_img,
    save_objects_to_img,
)
from os import rename, path
from glob import glob

# %% Test functions with
IMG_DIR = "../img"
RAW_DIR = IMG_DIR + "/raw"
DATA_DIR = IMG_DIR + "/data"
UNSORTED_DIR = IMG_DIR + "/unsorted"
PROCESSED_DIR = RAW_DIR + "/processed"
SCREENSHOTS_DIR = RAW_DIR + "/screenshots"


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
def process_images(
    raw_images_dir: str = SCREENSHOTS_DIR,
    image_type: str = "png",
    move_images: bool = False,
) -> None:
    screenshots = glob(f"{raw_images_dir}/*/*.{image_type}")
    print(f"Found {len(screenshots)} screenshots")
    for img_path in screenshots:
        print(f"Processing {img_path}")
        xml_path = img_path.replace(f".{image_type}", ".xml")
        if path.exists(xml_path):
            # Extract objects' labels and bounding box dimensions from XML
            objs_from_xml = extract_objects_from_xml(xml_path)
            # Extract objects from images using XML data
            objs_from_img = extract_objects_from_img(img_path, objs_from_xml)
            # Save extracted objects to images
            save_objects_to_img(objs_from_img, UNSORTED_DIR)
            # Move raw image to processed directory
            if move_images:
                for f in [img_path, xml_path]:
                    new_path = f.replace(raw_images_dir, PROCESSED_DIR)
                    print(f"    Moving {f} to {new_path}")
                    rename(f, new_path)
        else:
            print(f"    No XML file found for {img_path}")
