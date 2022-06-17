# %% Imports
import re
from collections import Counter
from glob import glob
from statistics import mean

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from data_processing import (
    RAW_DIR,
    TEST_DIR,
    TRAIN_DIR,
    UNSORTED_COG_DIR,
    UNSORTED_TOON_DIR,
    VALIDATE_DIR,
)
from img_utils import extract_objects_from_xml

# %% Global variables
with open(f"{RAW_DIR}/data/predefined_classes.txt", "r") as f:
    ALL_LABELS = [line for line in f.read().splitlines() if not line.startswith("=")]
ANIMALS = [label.split("_")[1] for label in ALL_LABELS if label.startswith("toon_")]
BINARY = ["cog", "toon"]
SUITS_LONG = ["bossbot", "lawbot", "cashbot", "sellbot"]
SUITS_SHORT = ["bb", "lb", "cb", "sb"]
SUITS_MAP = {short: long for short, long in zip(SUITS_SHORT, SUITS_LONG)}


# %% Define functions
def plot_suits_as_bar(img_dir: str = UNSORTED_COG_DIR) -> None:
    """Plot the number of cogs per suit in the unsorted directory"""
    # all_cogs = glob(f"{UNSORTED_DIR}/cog/*.png")
    # num_cogs = len(all_cogs)

    all_suits = [
        glob(f"{img_dir}/**/cog_{suit}*.png", recursive=True) for suit in SUITS_SHORT
    ]
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


def plot_toons_as_bar(img_dir: str = UNSORTED_TOON_DIR) -> None:
    """Plot the number of toons per animal in the unsorted directory"""
    # all_toons = glob(f"{UNSORTED_DIR}/toon/toon_*.png")
    # num_toons = len(all_toons)

    all_animals = [
        glob(f"{img_dir}/**/toon_{animal}*.png", recursive=True) for animal in ANIMALS
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
    """Plot object details retrieved from the screenshots' XML files"""
    obj_names = []
    for xml_path in glob(f"{RAW_DIR}/screenshots/*/*.xml"):
        for obj in extract_objects_from_xml(xml_path):
            obj_name, _, _, _, _ = obj
            obj_names.append(obj_name)

    # * Sort by highest value
    # sorted_all_labels = dict(count_all.most_common())
    # * Sort by key values
    # count_all_sorted = {key: count_all[key] for key in sorted(count_all.keys())}
    plot_counters(
        counters=count_objects(obj_names=obj_names),
        suptitle="Labels from screenshots' XML files",
    )


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
        y=history["val_accuracy"][max_val_accuracy - 1],
        color="orange",
        alpha=0.8,
        ls="--",
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


def compare_histories(histories: list, suptitle: str = "") -> None:
    """Plot and compare the histories (acc, val_acc, loss, val_loss) of multiple models

    The resulting plot is 4 subplots:
    1. Accuracy
    2. Validation accuracy
    3. Loss
    4. Validation loss

    Args:
        histories: List of tuples of (model: keras.Model, history: keras.callbacks.History)
    """
    # TODO - Can we add a plot for max acc and lowest loss?
    # TODO - Or should we just draw vertical lines for each max/min?
    # TODO - Is it possible to utilize `plot_history` here?
    _, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))
    for model, history in histories:
        num_epochs = range(0, len(history.history["loss"]))
        for idx, key in enumerate(["accuracy", "val_accuracy", "loss", "val_loss"]):
            name = model.name.strip("toonvision_")
            axes[idx].plot(history.history[key], label=f"{name}")
            axes[idx].legend()
            # axes[idx].set_xticks(num_epochs[::2])
            axes[idx].set_title(key)
    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()


def get_obj_name_from_filepath(filepath: str, file_ext: str = "png") -> str:
    """Given a filepath, return the full object name

    Args:
        filepath: Path to a file
        file_ext: File extension

    Returns:
        str: Full object name (ex: "cog_bb_flunky_1", "toon_cat_32")
    """
    regex_obj_name = re.compile(rf"((cog|toon)_.*)\.{file_ext}")
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


def count_objects(
    data_dir: str = None, obj_names: list[str] = None
) -> tuple[dict, dict, dict, dict]:
    """Count the objects in a data directory or list of object names

    Args:
        data_dir: Path to a data directory containing processed images
        obj_names: List of object names

    Returns:
        tuple: (count_all, count_binary, count_suit, count_animal)
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
            count_suit.update([SUITS_MAP.get(suit)])
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

    return (count_all, count_binary, count_suit, count_animal)


def plot_counters(counters: tuple[dict, dict, dict, dict], suptitle: str) -> None:
    count_all, count_binary, count_suit, count_animal = counters
    counts_and_titles = [
        (count_all, "Number of objects per label", 20),
        (count_binary, "Number of objects per binary label", 640),
        (count_suit, "Number of objects per suit label", 160),
        (count_animal, "Number of objects per animal label", 58),
    ]
    fig, ax = plt.subplots(
        4, 1, figsize=(5, 15), gridspec_kw={"height_ratios": [5, 0.75, 0.75, 1.25]}
    )
    fig.suptitle(suptitle, fontsize=20)
    for idx, (count_dict, title, desired_count) in enumerate(counts_and_titles):
        hbars = ax[idx].barh(y=list(count_dict.keys()), width=count_dict.values())
        ax[idx].invert_yaxis()
        ax[idx].axvline(x=mean(count_dict.values()), color="red", alpha=0.5)
        ax[idx].axvline(x=desired_count, color="green")
        ax[idx].set_title(title)
        ax[idx].bar_label(hbars, count_dict.values())
    plt.show()


def plot_datasets_all(text_color: str = "black") -> None:
    plt.figure(figsize=(5, 10))

    c_train = list(count_objects(data_dir=f"{TRAIN_DIR}/*/*.png"))
    train = c_train[0]
    labels = list(train.keys())
    bars_train = plt.barh(labels, train.values(), label="Train")
    plt.bar_label(bars_train, train.values(), label_type="center", color=text_color)

    c_validate = list(count_objects(data_dir=f"{VALIDATE_DIR}/*/*.png"))
    validate = c_validate[0]
    bars_validate = plt.barh(
        labels, validate.values(), label="Validate", left=list(train.values())
    )
    plt.bar_label(
        bars_validate, validate.values(), label_type="center", color=text_color
    )

    c_test = list(count_objects(data_dir=f"{TEST_DIR}/*/*.png"))
    test = c_test[0]
    bars_test = plt.barh(
        labels,
        test.values(),
        label="Test",
        left=np.add(list(train.values()), list(validate.values())),
    )
    plt.bar_label(bars_test, test.values(), label_type="center", color=text_color)

    plt.gca().invert_yaxis()
    plt.xlabel("Labels")
    plt.title("All labels per dataset")
    plt.grid(axis="x", linestyle="--")
    plt.legend(["Train", "Validate", "Test"])
    plt.show()


def plot_datasets_binary(text_color: str = "black") -> None:
    c_train = list(count_objects(data_dir=f"{TRAIN_DIR}/*/*.png"))
    train = c_train[1]
    labels = list(train.keys())
    bars_train = plt.barh(labels, train.values(), label="Train")
    plt.bar_label(bars_train, train.values(), label_type="center", color=text_color)

    c_validate = list(count_objects(data_dir=f"{VALIDATE_DIR}/*/*.png"))
    validate = c_validate[1]
    bars_validate = plt.barh(
        labels, validate.values(), label="Validate", left=list(train.values())
    )
    plt.bar_label(
        bars_validate, validate.values(), label_type="center", color=text_color
    )

    c_test = list(count_objects(data_dir=f"{TEST_DIR}/*/*.png"))
    test = c_test[1]
    bars_test = plt.barh(
        labels,
        test.values(),
        label="Test",
        left=np.add(list(train.values()), list(validate.values())),
    )
    plt.bar_label(bars_test, test.values(), label_type="center", color=text_color)

    plt.gca().invert_yaxis()
    plt.title("Binary labels per dataset")
    plt.grid(axis="x", linestyle="--")
    plt.legend(["Train", "Validate", "Test"])
    plt.show()


def plot_datasets_suits(text_color: str = "black"):
    c_train = list(count_objects(data_dir=f"{TRAIN_DIR}/*/*.png"))
    train = c_train[2]
    labels = list(train.keys())
    bars_train = plt.barh(labels, train.values(), label="Train")
    plt.bar_label(bars_train, train.values(), label_type="center", color=text_color)

    c_validate = list(count_objects(data_dir=f"{VALIDATE_DIR}/*/*.png"))
    validate = c_validate[2]
    bars_validate = plt.barh(
        labels, validate.values(), label="Validate", left=list(train.values())
    )
    plt.bar_label(
        bars_validate, validate.values(), label_type="center", color=text_color
    )

    c_test = list(count_objects(data_dir=f"{TEST_DIR}/*/*.png"))
    test = c_test[2]
    bars_test = plt.barh(
        labels,
        test.values(),
        label="Test",
        left=np.add(list(train.values()), list(validate.values())),
    )
    plt.bar_label(bars_test, test.values(), label_type="center", color=text_color)

    plt.gca().invert_yaxis()
    plt.xlabel("Suits")
    plt.title("Suit labels per dataset")
    plt.grid(axis="x", linestyle="--")
    plt.legend(["Train", "Validate", "Test"])
    plt.show()


def plot_datasets_animals(text_color: str = "black"):
    c_train = list(count_objects(data_dir=f"{TRAIN_DIR}/*/*.png"))
    train = c_train[3]
    labels = list(train.keys())
    bars_train = plt.barh(labels, train.values(), label="Train")
    plt.bar_label(bars_train, train.values(), label_type="center", color=text_color)

    c_validate = list(count_objects(data_dir=f"{VALIDATE_DIR}/*/*.png"))
    validate = c_validate[3]
    bars_validate = plt.barh(
        labels, validate.values(), label="Validate", left=list(train.values())
    )
    plt.bar_label(
        bars_validate, validate.values(), label_type="center", color=text_color
    )

    c_test = list(count_objects(data_dir=f"{TEST_DIR}/*/*.png"))
    test = c_test[3]
    bars_test = plt.barh(
        labels,
        test.values(),
        label="Test",
        left=np.add(list(train.values()), list(validate.values())),
    )
    plt.bar_label(bars_test, test.values(), label_type="center", color=text_color)

    plt.gca().invert_yaxis()
    plt.title("Animal labels per dataset")
    plt.grid(axis="x", linestyle="--")
    plt.legend(["Train", "Validate", "Test"])
    plt.show()


def draw_bounding_boxes(filepath: str, save_img: str = "") -> None:
    """Draw bounding boxes around objects in a raw screenshot.

    This function assumes there's a corresponding XML file containing the labeled bounding boxes in
    the same directory as the same filename.

    Args:
        filepath (str): Full path to the raw screenshot.
        save_img (str, optional): Full path to save the labeled image to. Defaults to "".
    """
    xml_path = filepath.replace(".png", ".xml")
    objs = extract_objects_from_xml(xml_path)

    plt.figure(figsize=(21, 9))
    # Display the image
    plt.imshow(Image.open(filepath), aspect="auto")

    # Add the patch to the Axes
    for obj in objs:
        label, xmin, ymin, xmax, ymax = obj
        height = ymax - ymin
        width = xmax - xmin
        plt.gca().add_patch(
            Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=3,
                edgecolor="r",
                facecolor="none",
            )
        )
        text_coords = (xmin - 75, ymin - 20)
        text = plt.text(
            text_coords[0],
            text_coords[1],
            label,
            fontsize=14,
            color="white",
            weight="bold",
        )
        text.set_path_effects([PathEffects.withStroke(linewidth=4, foreground="black")])
    plt.axis("off")

    if save_img:
        plt.savefig(save_img, bbox_inches="tight", pad_inches=0)


# %% Plot the histories
def plot_histories(
    axes,
    model_name: str,
    histories: list,
    color: str,
    alpha_runs: float = 0.15,
    alpha_mean: float = 0.85,
    index_slice: tuple = (0, -1),
) -> None:
    """Plot the history (accuracy and loss on the validation set) of a model as a line chart"""
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    idx_start, idx_end = index_slice

    for history in histories:
        num_epochs = range(1, len(history["loss"]) + 1)
        acc.append(history["accuracy"])
        val_acc.append(history["val_accuracy"])
        loss.append(history["loss"])
        val_loss.append(history["val_loss"])

        # Plot training & validation accuracy values
        axes[0][0].plot(
            num_epochs[idx_start:idx_end],
            history["accuracy"][idx_start:idx_end],
            color=color,
            alpha=alpha_runs,
        )
        axes[0][1].plot(
            num_epochs[idx_start:idx_end],
            history["val_accuracy"][idx_start:idx_end],
            color=color,
            alpha=alpha_runs,
        )
        # Plot training & validation loss values
        axes[1][0].plot(
            num_epochs[idx_start:idx_end],
            history["loss"][idx_start:idx_end],
            color=color,
            alpha=alpha_runs,
        )
        axes[1][1].plot(
            num_epochs[idx_start:idx_end],
            history["val_loss"][idx_start:idx_end],
            color=color,
            alpha=alpha_runs,
        )

    # Average of the histories
    avg_history = {
        "accuracy": np.mean(acc, axis=0),
        "val_accuracy": np.mean(val_acc, axis=0),
        "loss": np.mean(loss, axis=0),
        "val_loss": np.mean(val_loss, axis=0),
    }

    # Plot training & validation accuracy values
    axes[0][0].plot(
        num_epochs[idx_start:idx_end],
        avg_history["accuracy"][idx_start:idx_end],
        color=color,
        alpha=alpha_mean,
        label=model_name,
    )
    axes[0][1].plot(
        num_epochs[idx_start:idx_end],
        avg_history["val_accuracy"][idx_start:idx_end],
        color=color,
        alpha=alpha_mean,
        label=model_name,
    )
    axes[0][0].set_title("Accuracy")
    axes[0][1].set_title("Val Accuracy")
    for a in axes[0]:
        a.set_ylabel("Accuracy")
        a.legend()
    # Plot training & validation loss values
    axes[1][0].plot(
        num_epochs[idx_start:idx_end],
        avg_history["loss"][idx_start:idx_end],
        color=color,
        alpha=alpha_mean,
        label=model_name,
    )
    axes[1][1].plot(
        num_epochs[idx_start:idx_end],
        avg_history["val_loss"][idx_start:idx_end],
        color=color,
        alpha=alpha_mean,
        label=model_name,
    )
    axes[1][0].set_title("Loss")
    axes[1][1].set_title("Val Loss")
    for a in axes[1]:
        a.set_ylabel("Loss")
        a.set_xlabel("Epoch")
        a.legend()


def plot_evaluations_box(axes, evaluations_all: list, colors: list[str]) -> None:
    """Plot the evaluations (accuracy and loss on the test set) of a model as a box chart"""
    # Transpose each history so row1 contains all accuracies, row0 all losses
    # Transpose again once all accuracies are collected so that the rows are the runs
    # Result: A model's accuracies are located in the same column, but the runs are in different rows
    # Model data being per-column is necessary for the boxplot, read the documentation
    all_acc = np.array(
        [np.array(e).transpose()[1] for _, e in evaluations_all]
    ).transpose()
    all_loss = np.array(
        [np.array(e).transpose()[0] for _, e in evaluations_all]
    ).transpose()
    model_names = [e[0] for e in evaluations_all]

    # Plot test accuracy values
    # Set patch_artist to True so we can use the box's facecolor
    bp_acc = axes[0].boxplot(all_acc, notch=False, sym="o", patch_artist=True)
    bp_loss = axes[1].boxplot(all_loss, notch=False, sym="o", patch_artist=True)

    for ax, (bp, label) in enumerate([(bp_acc, "Accuracy"), (bp_loss, "Loss")]):
        axes[ax].set_title(f"Test {label}")
        axes[ax].set_ylabel(f"{label}")
        axes[ax].set_xticks([])  # Remove the x-axis labels
        axes[ax].xaxis.grid(False)
        axes[ax].yaxis.grid(
            True, linestyle="-", which="major", color="lightgrey", alpha=0.5
        )
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
        # Add the legend AFTER setting the colors so the colors in the legend are accurate
        axes[ax].legend(bp["boxes"], model_names)


def plot_evaluations_line_deprecated(
    axes,
    model_name: str,
    evaluations: list,
    color: str,
    alpha_runs: float = 0.15,
    index_slice: tuple = (0, -1),
) -> None:
    """Plot the evaluations (accuracy and loss on the test set) of a model as a line chart"""
    idx_start, idx_end = index_slice

    num_runs = range(1, len(evaluations) + 1)
    accuracies = [eval[1] for eval in evaluations]
    losses = [eval[0] for eval in evaluations]

    # Plot test accuracy values
    axes[0][2].plot(
        num_runs[idx_start:idx_end],
        accuracies[idx_start:idx_end],
        color=color,
        alpha=alpha_runs,
        label=model_name,
    )
    # Plot test loss values
    axes[1][2].plot(
        num_runs[idx_start:idx_end],
        losses[idx_start:idx_end],
        color=color,
        alpha=alpha_runs,
        label=model_name,
    )

    axes[0][2].set_title("Test Accuracy")
    for a in axes[0]:
        a.set_ylabel("Accuracy")
        a.legend()
        a.grid(axis="y")

    axes[1][2].set_title("Test Loss")
    axes[1][2].set_xlabel("Run")
    for a in axes[1]:
        a.set_ylabel("Loss")
        a.legend()
        a.grid(axis="y")


# %% Plot data
# plot_suits_as_bar()
# plot_toons_as_bar()
# plot_xml_data()
# plot_datasets_all()
# plot_datasets_binary()
# plot_datasets_suits()
# plot_datasets_animals()
