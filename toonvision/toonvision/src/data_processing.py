# %% Imports
from img_utils import (
    extract_objects_from_xml,
    extract_objects_from_img,
    save_objects_to_img,
)
from os import rename, path
from glob import glob
from random import shuffle
from tensorflow.keras.utils import image_dataset_from_directory

# %% Global variables: directories
IMG_DIR = path.dirname(__file__).replace("src", "img").replace("\\", "/")

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
    filename_filter: str = "",
    verbose: bool = False,
) -> None:
    """Extract objects from raw images and save them to the unsorted img directory"""
    screenshots = glob(f"{raw_images_dir}/**/*.{image_type}", recursive=True)
    print(f"Found {len(screenshots)} screenshots in {raw_images_dir}")
    for img_path in screenshots:
        if filename_filter in img_path:  # TODO move to glob above
            if verbose:
                print(f"Processing {img_path}")
            xml_path = img_path.replace(f".{image_type}", ".xml")
            if path.exists(xml_path):
                # Extract objects' labels and bounding box dimensions from XML
                objs_from_xml = extract_objects_from_xml(xml_path)
                # Extract objects from images using XML data
                objs_from_img = extract_objects_from_img(img_path, objs_from_xml)
                # Save extracted objects to images, modify image name to include object index
                save_objects_to_img(objs_from_img, UNSORTED_DIR, DATA_DIR, verbose=verbose)
                # Move raw image to processed directory
                if move_images:
                    for f in [img_path, xml_path]:
                        new_path = f.replace(raw_images_dir, PROCESSED_DIR)
                        if verbose:
                            print(f"    Moving {f} to {new_path}")
                        rename(f, new_path)
            else:
                print(f"    No XML file found for {img_path}")


def split_data(split_ratio: list[float, float, float], dry_run: bool = False):
    """Split the data into train(60%)/validate(20%)/test(20%) data sets"""
    for unsorted_dir in [UNSORTED_COG_DIR, UNSORTED_TOON_DIR]:
        cog_or_toon = unsorted_dir.split("/")[-1]
        # Get all images from unsorted_dir
        unsorted_images = glob(f"{unsorted_dir}/*.png")
        num_images = len(unsorted_images)
        print(f"{unsorted_dir.split('toonvision')[-1]} # Unsorted images: ", num_images)

        # Split images into train/validate/test sets
        num_train = int(num_images * split_ratio[0])
        num_validate = int(num_images * split_ratio[1])
        num_test = num_images - num_train - num_validate
        print(f"{unsorted_dir.split('toonvision')[-1]} # Train, Val, Test: ", num_train, num_validate, num_test)

        # # Shuffle filenames to randomize the order of the images
        shuffle(unsorted_images)
        train = unsorted_images[:num_train]
        validate = unsorted_images[num_train:-num_test]
        test = unsorted_images[-num_test:]

        # Move images to train/validate/test directories
        for images, dir_name in zip(
            [train, validate, test], [TRAIN_DIR, VALIDATE_DIR, TEST_DIR]
        ):
            for img_path in images:
                new_path = img_path.replace(unsorted_dir, f"{dir_name}/{cog_or_toon}")
                if dry_run:
                    print(f"Moving {img_path} to {new_path}")
                else:
                    rename(img_path, new_path)


def create_datasets(
    image_size: tuple = (600, 200),
    batch_size: int = 32,
    shuffle: bool = True,
    split_ratio: list[float, float, float] = None,
):
    """Create binary (Cog vs Toon) datasets for training, validation, and testing

    Args:
        image_size (tuple, optional): Tuple of height and width. Defaults to (600, 200).
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Shuffle images in dataset. Defaults to True.
        split_ratio (list[float, float, float], optional): Train/val/test split. Defaults to None.

    Returns:
        tuple[keras.Dataset, keras.Dataset, keras.Dataset]: Train, validate, and test datasets.
    """
    if split_ratio:
        split_data(split_ratio=split_ratio, dry_run=False)

    ds_train = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    ds_validate = image_dataset_from_directory(
        VALIDATE_DIR,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    ds_test = image_dataset_from_directory(
        TEST_DIR,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return (ds_train, ds_validate, ds_test)


# def create_suit_datasets(
#     image_size: tuple = (600, 200),
#     batch_size: int = 32,
#     shuffle: bool = True,
#     split_ratio: list[float, float, float] = None,
# ):
#     """Create multiclass Cog suit datasets for training, validation, and testing

#     Args:
#         image_size (tuple, optional): Tuple of height and width. Defaults to (600, 200).
#         batch_size (int, optional): Number of samples per batch. Defaults to 32.
#         shuffle (bool, optional): Shuffle images in dataset. Defaults to True.
#         split_ratio (list[float, float, float], optional): Train/val/test split. Defaults to None.

#     Returns:
#         tuple[keras.Dataset, keras.Dataset, keras.Dataset]: Train, validate, and test datasets.
#     """
#     if split_ratio:
#         split_data(split_ratio=split_ratio, dry_run=False)

#     # ds_train = image_dataset_from_directory(
#     #     TRAIN_DIR + "/cog",
#     #     image_size=image_size,
#     #     batch_size=batch_size,
#     #     shuffle=shuffle,
#     # )
#     # ds_validate = image_dataset_from_directory(
#     #     VALIDATE_DIR + "/cog",
#     #     image_size=image_size,
#     #     batch_size=batch_size,
#     #     shuffle=shuffle,
#     # )
#     # ds_test = image_dataset_from_directory(
#     #     TEST_DIR + "/cog",
#     #     image_size=image_size,
#     #     batch_size=batch_size,
#     #     shuffle=shuffle,
#     # )
#     return (ds_train, ds_validate, ds_test)


def unsort_data(dry_run: bool = False):
    for dir_name in [TRAIN_DIR, VALIDATE_DIR, TEST_DIR]:
        all_imgs = glob(f"{dir_name}/*/*.png")
        for img in all_imgs:
            new_path = img.replace("\\", "/").replace(dir_name, UNSORTED_DIR)
            if dry_run:
                print(f"Moving {img} to {new_path}")
            else:
                rename(img, new_path)


# # %% Split data
# split_data(dry_run=True)

# # %% Unsort data
# unsort_data(dry_run=True)

# %%
