# %% Imports
from img_utils import (
    extract_objects_from_xml,
    extract_objects_from_img,
    save_objects_to_img,
)
from os import rename, path
from glob import glob
from random import shuffle

# %% Test functions with
IMG_DIR = "../img"

RAW_DIR = IMG_DIR + "/raw"
DATA_DIR = IMG_DIR + "/data"

TRAIN_DIR = DATA_DIR + "/train"
VALIDATE_DIR = DATA_DIR + "/validate"
TEST_DIR = DATA_DIR + "/test"

PROCESSED_DIR = RAW_DIR + "/processed"
SCREENSHOTS_DIR = RAW_DIR + "/screenshots"

UNSORTED_DIR = IMG_DIR + "/unsorted"
UNSORTED_COG_DIR = UNSORTED_DIR + "/cog"
UNSORTED_TOON_DIR = UNSORTED_DIR + "/toon"


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


def unsort_data(dry_run: bool = False):
    for dir_name in [TRAIN_DIR, VALIDATE_DIR, TEST_DIR]:
        all_imgs = glob(f"{dir_name}/*/*.png")
        for img in all_imgs:
            new_path = img.replace("\\", '/').replace(dir_name, UNSORTED_DIR)
            if dry_run:
                print(f"Moving {img} to {new_path}")
            else:
                rename(img, new_path)


# # %% Split data
# split_data(dry_run=True)

# # %% Unsort data
# unsort_data(dry_run=True)

# %%
