# %% Imports
from collections import Counter
from glob import glob
from os import path, rename

import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory

from img_utils import (
    extract_objects_from_img,
    extract_objects_from_xml,
    get_obj_details_from_filepath,
    get_obj_details_from_name,
    save_objects_to_img,
)

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

# Global variables: labels
with open(f"{RAW_DIR}/data/predefined_classes.txt", "r") as f:
    ALL_LABELS = [line for line in f.read().splitlines() if not line.startswith("=")]
ANIMALS = [label.split("_")[1] for label in ALL_LABELS if label.startswith("toon_")]
BINARY = ["cog", "toon"]

SUITS_LONG = ["bossbot", "lawbot", "cashbot", "sellbot"]
SUITS_SHORT = ["bb", "lb", "cb", "sb"]
MAP_INT_TO_SUIT = {i: suit for i, suit in enumerate(SUITS_SHORT)}
MAP_SUIT_TO_INT = {v: k for k, v in MAP_INT_TO_SUIT.items()}
MAP_SUIT_TO_ONEHOT = {
    suit: one_hot.astype(int)
    for suit, one_hot in zip(SUITS_SHORT, np.eye(len(SUITS_SHORT)))
}
MAP_SUIT_SHORT_LONG = {short: long for short, long in zip(SUITS_SHORT, SUITS_LONG)}

STREETS = ["br", "dd", "ddl", "dg", "mml", "ttc"]


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
    filename_filter: str = "*",
    verbose: bool = False,
) -> None:
    """Extract objects from raw images and save them to the unsorted img directory"""
    screenshots = glob(
        f"{raw_images_dir}/**/{filename_filter}.{image_type}", recursive=True
    )
    print(f"Found {len(screenshots)} screenshots in {raw_images_dir}")
    for img_path in screenshots:
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


def count_objects(data_dir: str = None, obj_names: list[str] = None) -> dict[str, dict]:
    """Count the objects in a data directory or list of object names

    Args:
        data_dir: Path to a data directory containing processed images
            Eg. UNSORTED_DIR + "/**/*.png"
        obj_names: List of object names

    Returns:
        dict: Dictionary of counter names and their counts

        :: Example
            {
                "all": count_all,
                "binary": count_binary,
                "suit": count_suit,
                "animal": count_animal,
            }
    """
    assert any(
        [data_dir is not None, obj_names is not None]
    ), "Must specify either data_dir or obj_names"

    def create_counters() -> tuple[Counter, Counter, Counter, Counter]:
        # Create counters
        count_all = Counter()  # All object names (32 cogs + 11 toons = 43 classes)
        count_binary = Counter()  # Cog or Toon (2 classes)
        count_suit = Counter()  # Bossbot, Lawbot, Cashbot, Sellbot (4 classes)
        count_animal = Counter()  # Toon animals (11 classes)
        # Initialize all counters to 0
        count_all.update({key: 0 for key in ALL_LABELS})
        count_binary.update({key: 0 for key in BINARY})
        count_suit.update({key: 0 for key in SUITS_LONG})
        count_animal.update({key: 0 for key in ANIMALS})
        return (count_all, count_binary, count_suit, count_animal)

    def update_counters(obj_details: dict) -> None:
        binary = obj_details["binary"]
        count_binary.update([binary])

        if binary == "cog":
            suit, name = obj_details["suit"], obj_details["name"]
            obj_formatted = f"{binary}_{suit}_{name}"
            count_suit.update([MAP_SUIT_SHORT_LONG.get(suit)])
        else:
            animal = obj_details["animal"]
            obj_formatted = f"{binary}_{animal}"
            count_animal.update([animal])
        count_all.update([obj_formatted])

    count_all, count_binary, count_suit, count_animal = create_counters()

    if data_dir:
        for filepath in glob(data_dir):
            obj_details = get_obj_details_from_filepath(filepath)
            update_counters(obj_details)
    else:
        for obj_name in obj_names:
            obj_details = get_obj_details_from_name(obj_name)
            update_counters(obj_details)

    return {
        "all": count_all,
        "binary": count_binary,
        "suit": count_suit,
        "animal": count_animal,
    }


def split_data(split_ratio: list[float, float, float]):
    """Split data into train, validate, and test directories

    Samples are split into train, validate, and test directories based on the split_ratio.
    The split_ratio is a list of ratios for the train, validate, and test directories. The sum of
    the ratios must be 1.

    Samples are split on a per-label basis, rather than the entire dataset being split. This way,
    we ensure a representative sample of each label is in each directory.

    Args:
        split_ratio (list[float, float, float]): Train/validate/test split ratio.
            Example: [0.6, 0.2, 0.2]
    """
    counters_all = count_objects(UNSORTED_DIR + "/**/*.png")["all"]
    # for label in counters_all:
    for label in counters_all:
        label_count = counters_all[label]
        img_fps = glob(UNSORTED_DIR + f"/**/{label}*.png")

        # Count the number of images per split
        num_train = int(label_count * split_ratio[0])
        num_validate = int(label_count * split_ratio[1])

        # Shuffle the images
        idx_shuffled = np.random.permutation(label_count)
        idx_train = idx_shuffled[:num_train]
        idx_validate = idx_shuffled[num_train : num_train + num_validate]
        idx_test = idx_shuffled[num_train + num_validate :]

        # Split images into train/validate/test sets
        train = [img_fps[i] for i in idx_train]
        validate = [img_fps[i] for i in idx_validate]
        test = [img_fps[i] for i in idx_test]

        # Move images to train/validate/test directories
        for images, dir_name in zip(
            [train, validate, test], [TRAIN_DIR, VALIDATE_DIR, TEST_DIR]
        ):
            for img_path in images:
                new_path = img_path.replace(UNSORTED_DIR, dir_name)
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
        split_data(split_ratio=split_ratio)

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


def create_suit_datasets(
    image_size: tuple = (600, 200),
    batch_size: int = 32,
    shuffle: bool = True,
    split_ratio: list[float, float, float] = None,
):
    """Create multiclass Cog suit datasets for training, validation, and testing

    Args:
        image_size (tuple, optional): Tuple of height and width. Defaults to (600, 200).
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Shuffle images in dataset. Defaults to True.
        split_ratio (list[float, float, float], optional): Train/val/test split. Defaults to None.

    Returns:
        tuple[keras.Dataset, keras.Dataset, keras.Dataset]: Train, validate, and test datasets.
    """
    if split_ratio:
        split_data(split_ratio=split_ratio)

    # ds_train = image_dataset_from_directory(
    #     TRAIN_DIR + "/cog",
    #     image_size=image_size,
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    # )
    # ds_validate = image_dataset_from_directory(
    #     VALIDATE_DIR + "/cog",
    #     image_size=image_size,
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    # )
    # ds_test = image_dataset_from_directory(
    #     TEST_DIR + "/cog",
    #     image_size=image_size,
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    # )
    # return (ds_train, ds_validate, ds_test)
    return None


def unprocess_data(dry_run: bool = False):
    """Move images and XML files from img/raw/processed to img/raw/screenshots"""
    img_fps = glob(f"{PROCESSED_DIR}/**/*.png", recursive=True)
    for fp in img_fps:
        new_fp = fp.replace("\\", "/").replace(PROCESSED_DIR, SCREENSHOTS_DIR)
        if dry_run:
            print(f"Moving {fp} to {new_fp}")
        else:
            rename(fp, new_fp)
            rename(fp.replace(".png", ".xml"), new_fp.replace(".png", ".xml"))


def unsort_data(dry_run: bool = False):
    """Move images from img/data/{train|validate|test} to img/data/unsorted"""
    for dir_name in [TRAIN_DIR, VALIDATE_DIR, TEST_DIR]:
        img_fps = glob(f"{dir_name}/*/*.png")
        for fp in img_fps:
            new_fp = fp.replace("\\", "/").replace(dir_name, UNSORTED_DIR)
            if dry_run:
                print(f"Moving {fp} to {new_fp}")
            else:
                rename(fp, new_fp)


def get_suits_from_dir(directories: list[str] = [DATA_DIR]) -> dict[str, tuple]:
    """Get filepaths and labels for all Cog suits in the directories. Defaults to entire dataset.

    Args:
        directories (list[str], optional): Directories to search for Cog suits. Defaults to [TRAIN_DIR, VALIDATE_DIR, TEST_DIR].

    Returns:
        dict[str, tuple]: Filepaths and labels for all Cog suits.
    """
    # filepaths = [fp for fp in walk(TRAIN_DIR + "/cog")]
    result = {}
    for dir in directories:
        filepaths = glob(dir + "/**/cog/*.png", recursive=True)
        obj_names = [
            fp.split("\\")[-1] for fp in filepaths
        ]  # Glob always returns double backslash
        obj_details = [get_obj_details_from_filepath(object) for object in obj_names]
        labels = [cog["suit"] for cog in obj_details]
        result[dir] = (filepaths, labels)
    return result


def suit_to_integer(suits: list[str]) -> list[int]:
    """Get integer labels for all suits in the list.

    Returns:
        list[int]: Integer labels for all Cog suits.
    """
    return [MAP_SUIT_TO_INT[suit] for suit in suits]


def integer_to_suit(suits: list[int]) -> list[str]:
    """Get string labels for all integers in the list.

    Returns:
        list[str]: String labels for all Cog suits.
    """
    return [MAP_INT_TO_SUIT[suit] for suit in suits]


def suit_to_onehot(suits: list[str]) -> list[np.ndarray]:
    """Get one-hot labels for all suits in the list.

    Args:
        suits (list[str]): List of Cog suits.

    Returns:
        list[np.ndarray]: One-hot labels for all Cog suits.

    Numpy alternative:
        n_suits = np.max(suits) + 1
        np.eye(n_suits)[suits]
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
    """
    return [MAP_SUIT_TO_ONEHOT[suit] for suit in suits]
