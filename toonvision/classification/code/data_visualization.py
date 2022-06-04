# %% Imports
from collections import Counter
from glob import glob
from statistics import mean

import matplotlib.pyplot as plt

from data_processing import RAW_DIR, UNSORTED_DIR
from img_utils import extract_objects_from_xml

# %% Global variables
with open(f"{RAW_DIR}/data/predefined_classes.txt", "r") as f:
    ALL_LABELS = [line for line in f.read().splitlines() if not line.startswith("=")]
ANIMALS = [label.split("_")[1] for label in ALL_LABELS if label.startswith("toon_")]
BINARY = ["cog", "toon"]
SUITS_LONG = ["bossbot", "lawbot", "cashbot", "sellbot"]
SUITS_SHORT = ["bb", "lb", "cb", "sb"]


# %% Define functions
def plot_suits_as_bar() -> None:
    # all_cogs = glob(f"{UNSORTED_DIR}/cog/*.png")
    # num_cogs = len(all_cogs)

    all_suits = [glob(f"{UNSORTED_DIR}/cog/cog_{suit}*.png") for suit in SUITS_SHORT]
    num_suits = [len(suit) for suit in all_suits]
    # all_suits_dict = dict(zip(SUITS, num_suits))
    # print(num_cogs, all_suits_dict)

    # Bar chart of all_suits_dict
    bars = plt.bar(SUITS_SHORT, num_suits)
    plt.title("Number of labels per suit")
    plt.xlabel("Suit")
    plt.bar_label(bars, num_suits)
    plt.axhline(y=160, color="green")
    plt.axhline(y=mean(num_suits), color="red", alpha=0.5, ls="dotted")
    plt.show()


def plot_toons_as_bar() -> None:
    # all_toons = glob(f"{UNSORTED_DIR}/toon/toon_*.png")
    # num_toons = len(all_toons)

    all_animals = [
        glob(f"{UNSORTED_DIR}/toon/toon_{animal}*.png") for animal in ANIMALS
    ]
    num_animals = [len(animal) for animal in all_animals]
    # all_animals_dict = dict(zip(ANIMALS, num_animals))
    # print(num_toons, all_animals_dict)

    # Bar chart of all_animals_dict
    bars = plt.bar(ANIMALS, num_animals)
    plt.title("Number of labels per animal")
    plt.xlabel("Animal")
    plt.xticks(rotation=-45)
    plt.bar_label(bars, num_animals)
    plt.axhline(y=58, color="green")
    plt.axhline(y=mean(num_animals), color="red", alpha=0.5)
    plt.show()


def plot_xml_data() -> None:
    def get_binary_label(obj_name: str) -> str:
        obj_type = obj_name.split("_")[0]
        if obj_type in BINARY:
            return obj_type
        return "unknown"

    def get_suit_label(obj_name: str) -> str:
        map = {short: long for short, long in zip(SUITS_SHORT, SUITS_LONG)}
        # Ex: Retrieve the value of the key "bb" from obj_name="cog_bb_flunky"
        return map.get(obj_name.split("_")[1], "unknown")

    def get_animal_label(obj_name: str) -> str:
        return obj_name.split("_")[1]

    def update_counters(obj_name: str) -> None:
        count_all.update([obj_name])
        count_binary.update([get_binary_label(obj_name)])
        if obj_name.startswith("cog"):
            count_suit.update([get_suit_label(obj_name)])
        if obj_name.startswith("toon"):
            count_animal.update([get_animal_label(obj_name)])

    def create_counters() -> tuple[Counter, Counter, Counter, Counter]:
        # Create counters
        count_all = Counter()  # All object names (32 + 11 = 43 classes)
        count_binary = Counter()  # Cog or Toon (2 classes)
        count_suit = Counter()  # Bossbot, Lawbot, Cashbot, Sellbot (4 classes)
        count_animal = Counter()  # Toon animals (11 classes)
        # Initialize all counters to 0
        count_all.update({key: 0 for key in ALL_LABELS})
        count_binary.update({key: 0 for key in BINARY})
        count_suit.update({key: 0 for key in SUITS_LONG})
        count_animal.update({key: 0 for key in ANIMALS})
        return (count_all, count_binary, count_suit, count_animal)

    count_all, count_binary, count_suit, count_animal = create_counters()

    all_xml = glob(f"{RAW_DIR}/screenshots/*/*.xml")
    for xml in all_xml:
        objs = extract_objects_from_xml(xml)
        for obj_name, _, _, _, _ in objs:
            update_counters(obj_name)

    # * Sort by highest value
    # sorted_all_labels = dict(count_all.most_common())
    # * Sort by key values
    # count_all_sorted = {key: count_all[key] for key in sorted(count_all.keys())}

    # Counter obj, title, desired_counter, axis,
    counts_and_titles = [
        (count_all, "Number of objects per label", 20),
        (count_binary, "Number of objects per binary label", 640),
        (count_suit, "Number of objects per suit label", 160),
        (count_animal, "Number of objects per animal label", 58),
    ]
    fig, ax = plt.subplots(
        4, 1, figsize=(5, 15), gridspec_kw={"height_ratios": [5, 0.75, 0.75, 1.25]}
    )
    fig.suptitle("Number of labels in unprocessed screenshots")
    for idx, (count_dict, title, desired_count) in enumerate(counts_and_titles):
        hbars = ax[idx].barh(y=list(count_dict.keys()), width=count_dict.values())
        ax[idx].invert_yaxis()
        ax[idx].axvline(x=mean(count_dict.values()), color="red", alpha=0.5)
        ax[idx].axvline(x=desired_count, color="green")
        ax[idx].set_title(title)
        ax[idx].bar_label(hbars, count_dict.values())
    plt.show()


def plot_image_sizes(img_dir: str) -> None:
    """Plot all image sizes in a directory"""
    # ! This poorly represents image sizes, but it's a start
    import matplotlib.pyplot as plt
    import numpy as np

    # Get image sizes
    sizes_height = []
    sizes_width = []
    for img_path in glob(f"{img_dir}/*/*.png"):
        img = plt.imread(img_path)
        sizes_height.append(img.shape[0])
        sizes_width.append(img.shape[1])
    # Plot histogram
    plt.hist(
        sizes_height,
        bins=100,
    )
    plt.title(f"Image sizes (height) in {img_dir}")
    plt.show()
    plt.hist(sizes_width, bins=100)
    plt.title(f"Image sizes (width) in {img_dir}")
    plt.show()
    print(np.mean(sizes_height), np.mean(sizes_width))


def plot_history(history: dict, name: str = "Model") -> None:
    """Plot the history (accuracy and loss) of a model"""
    import matplotlib.pyplot as plt
    import numpy as np

    max_accuracy = np.argmax(history["accuracy"]) + 1
    max_val_accuracy = np.argmax(history["val_accuracy"]) + 1
    min_loss = np.argmin(history["loss"]) + 1
    min_val_loss = np.argmin(history["val_loss"]) + 1
    num_epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.tight_layout()
    # Plot training & validation accuracy values
    axes[0].plot(num_epochs, history["accuracy"])
    axes[0].plot(num_epochs, history["val_accuracy"])
    axes[0].set_title(f"{name} accuracy")
    axes[0].axvline(
        x=max_accuracy,
        color="blue",
        alpha=0.5,
        ls="--",
    )
    axes[0].axvline(x=max_val_accuracy, color="orange", alpha=0.8, ls="--")
    axes[0].axhline(
        y=history["val_accuracy"][max_val_accuracy - 1], color="orange", alpha=0.8, ls="--"
    )
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_xticks(num_epochs[1::2])
    axes[0].legend(["Train", "Validation"], loc="upper left")
    axes[0].grid(axis="y")

    # Plot training & validation loss values
    axes[1].plot(num_epochs, history["loss"])
    axes[1].plot(num_epochs, history["val_loss"])
    axes[1].set_title(f"{name} loss")
    axes[1].axvline(x=min_loss, color="blue", alpha=0.5, ls="--")
    axes[1].axvline(x=min_val_loss, color="orange", alpha=0.8, ls="--")
    axes[1].axhline(
        y=history["val_loss"][min_val_loss - 1], color="orange", alpha=0.8, ls="--"
    )
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(["Train", "Validation"], loc="upper left")
    axes[1].set_xticks(num_epochs[1::2])
    axes[1].grid(axis="y")


def compare_histories(histories: list) -> None:
    """Plot and compare the histories (acc, val_acc, loss, val_loss) of multiple models

    The resulting plot is 4 subplots:
    1. Accuracy
    2. Validation accuracy
    3. Loss
    4. Validation loss

    Args:
        histories: List of tuples of (model: keras.Model, history: keras.callbacks.History)
    """
    _, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))
    for model, history in histories:
        for idx, key in enumerate(["accuracy", "val_accuracy", "loss", "val_loss"]):
            name = model.name.strip("toonvision_")
            axes[idx].plot(history.history[key], label=f"{name}")
            axes[idx].grid(axis="y")
            axes[idx].legend(loc="upper left")
            axes[idx].set_title(key)
    plt.tight_layout()
    plt.grid(axis="y")


# TODO Plot training/validation/test datasets
def plot_all_datasets():
    pass


# %% Plot data
# plot_suits_as_bar()
# plot_toons_as_bar()
# plot_xml_data()
