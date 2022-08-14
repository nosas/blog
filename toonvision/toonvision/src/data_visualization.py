# %% Imports
from collections import Counter
from glob import glob
from statistics import mean

import keras
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data_processing import (
    ALL_LABELS,
    ANIMALS,
    BINARY,
    PROCESSED_DIR,
    STREETS,
    SUITS_LONG,
    SUITS_SHORT,
    TEST_DIR,
    TRAIN_DIR,
    UNSORTED_COG_DIR,
    UNSORTED_TOON_DIR,
    VALIDATE_DIR,
    count_objects,
)
from img_utils import extract_objects_from_xml

# %% Global variables
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
COLORS_STREETS = [
    "#228e99",  # The Brrrgh
    "#a14e4a",  # Donald's Dock
    "#363251",  # Donald's Dreamland
    "#2a873b",  # Daisy Gardens
    "#9b3d7f",  # Minnie's Melodyland
    "#c06635",  # ToonTown Central
]
# Set default plot style and colors
plt.style.use("dark_background")
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLORS)


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
    for xml_path in glob(f"{PROCESSED_DIR}/**/*.xml", recursive=True):
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


def plot_history(
    history: dict,
    name: str = "Model",
    multiclass: bool = False,
    loss_ylim: tuple[int, int] = None,
    includes_validation: bool = True,
    onehot: bool = False,
) -> None:
    """Plot the history (accuracy and loss) of a model"""
    import matplotlib.pyplot as plt
    import numpy as np

    accuracy_str = "accuracy" if not multiclass else "sparse_categorical_accuracy"
    accuracy_str = accuracy_str if not onehot else "categorical_accuracy"

    max_accuracy = np.argmax(history[accuracy_str]) + 1
    min_loss = np.argmin(history["loss"]) + 1
    num_epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    fig.tight_layout()

    # Plot training accuracy values
    axes[0].plot(num_epochs, history[accuracy_str], label="Train acc")
    # Plot the maximum training accuracies as vertical and horizontal lines
    axes[0].axvline(
        x=max_accuracy,
        color="#e377c2",
        alpha=0.5,
        ls="--",
        label="Max acc",
    )
    axes[0].set_title(f"{name} accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(axis="y", alpha=0.5, color="lightgrey")

    # Plot training loss values
    axes[1].plot(num_epochs, history["loss"], label="Train loss")
    # Plot the minimum training loss as vertical and horizontal lines
    axes[1].axvline(x=min_loss, color="#e377c2", alpha=0.5, ls="--", label="Min loss")
    axes[1].set_title(f"{name} loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(axis="y", alpha=0.5, color="lightgrey")
    if loss_ylim:
        axes[1].set_ylim(*loss_ylim)

    if includes_validation:  # Plot validation accuracy values if included in history

        val_accuracy_str = "val_" + accuracy_str
        max_val_accuracy = np.argmax(history[val_accuracy_str]) + 1
        min_val_loss = np.argmin(history["val_loss"]) + 1

        # Plot validation accuracy values
        axes[0].plot(num_epochs, history[val_accuracy_str], label="Val acc")
        # Plot the maximum validation accuracies as vertical and horizontal lines
        axes[0].axvline(x=max_val_accuracy, color="#2ca02c", alpha=0.8, ls="--")
        axes[0].axhline(
            y=history[val_accuracy_str][max_val_accuracy - 1],
            color="#2ca02c",
            alpha=0.8,
            ls="--",
            label="Max val acc",
        )
        # Plot validation loss values
        axes[1].plot(num_epochs, history["val_loss"], label="Val loss")
        # Plot the minimum validation loss as vertical and horizontal lines
        axes[1].axvline(
            x=min_val_loss, color="#2ca02c", alpha=0.8, ls="--", label="Min val loss"
        )
        axes[1].axhline(
            y=history["val_loss"][min_val_loss - 1],
            color="#2ca02c",
            alpha=0.8,
            ls="--",
        )

    axes[0].legend()
    axes[1].legend()


def compare_histories(
    histories: list, suptitle: str = "", multiclass: bool = False
) -> None:
    """Plot and compare the histories (acc, val_acc, loss, val_loss) of multiple models

    The resulting plot is 4 subplots:
    1. Accuracy
    2. Validation accuracy
    3. Loss
    4. Validation loss

    Args:
        histories: List of tuples of (model_name: str, history: dict[str, str, str, str])
    """
    _, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))
    accuracy_str = "accuracy" if not multiclass else "sparse_categorical_accuracy"
    val_accuracy_str = "val_" + accuracy_str
    for model_name, history in histories:
        # num_epochs = range(0, len(history["loss"]))
        for idx, key in enumerate([accuracy_str, val_accuracy_str, "loss", "val_loss"]):
            name = model_name.replace("toonvision_", "")
            axes[idx].plot(history[key], label=f"{name}")
            axes[idx].legend()
            # axes[idx].set_xticks(num_epochs[::2])
            axes[idx].set_title(key)
    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()


def plot_counters(counters: dict[str, dict], suptitle: str) -> None:
    gridspec_kw = dict(width_ratios=[1, 1.2], height_ratios=[1, 1, 1])
    fig, ax = plt.subplot_mosaic(
        [["left", "upper right"], ["left", "middle right"], ["left", "lower right"]],
        gridspec_kw=gridspec_kw,
        figsize=(10, 8),
        dpi=100,
    )

    counts_and_titles = [
        (counters["all"], "Objects per label", 30, "left"),
        (counters["binary"], "Objects per binary label", 960, "upper right"),
        (counters["suit"], "Objects per suit label", 240, "middle right"),
        (counters["animal"], "Objects per animal label", 58, "lower right"),
    ]

    fig.suptitle(suptitle, fontsize=20)
    for count_dict, title, desired_count, subplot_key in counts_and_titles:
        hbars = ax[subplot_key].barh(
            y=list(count_dict.keys()), width=count_dict.values()
        )
        ax[subplot_key].invert_yaxis()
        ax[subplot_key].axvline(x=mean(count_dict.values()), color="red", alpha=0.5)
        ax[subplot_key].axvline(x=desired_count, color="green")
        ax[subplot_key].set_title(title)
        ax[subplot_key].bar_label(hbars, count_dict.values())

    fig.tight_layout()
    fig.show()


def plot_datasets_all(text_color: str = "black") -> None:
    plt.figure(figsize=(5, 10), dpi=100)

    c_train = count_objects(data_dir=f"{TRAIN_DIR}/*/*.png")
    train = c_train["all"]
    labels = list(train.keys())
    bars_train = plt.barh(labels, train.values(), label="Train")
    plt.bar_label(bars_train, train.values(), label_type="center", color=text_color)

    c_validate = count_objects(data_dir=f"{VALIDATE_DIR}/*/*.png")
    validate = c_validate["all"]
    bars_validate = plt.barh(
        labels, validate.values(), label="Validate", left=list(train.values())
    )
    plt.bar_label(
        bars_validate, validate.values(), label_type="center", color=text_color
    )

    c_test = count_objects(data_dir=f"{TEST_DIR}/*/*.png")
    test = c_test["all"]
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
    c_train = count_objects(data_dir=f"{TRAIN_DIR}/*/*.png")
    train = c_train["binary"]
    labels = list(train.keys())
    bars_train = plt.barh(labels, train.values(), label="Train")
    plt.bar_label(bars_train, train.values(), label_type="center", color=text_color)

    c_validate = count_objects(data_dir=f"{VALIDATE_DIR}/*/*.png")
    validate = c_validate["binary"]
    bars_validate = plt.barh(
        labels, validate.values(), label="Validate", left=list(train.values())
    )
    plt.bar_label(
        bars_validate, validate.values(), label_type="center", color=text_color
    )

    c_test = count_objects(data_dir=f"{TEST_DIR}/*/*.png")
    test = c_test["binary"]
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
    c_train = count_objects(data_dir=f"{TRAIN_DIR}/*/*.png")
    train = c_train["suit"]
    labels = list(train.keys())
    bars_train = plt.barh(labels, train.values(), label="Train")
    plt.bar_label(bars_train, train.values(), label_type="center", color=text_color)

    c_validate = count_objects(data_dir=f"{VALIDATE_DIR}/*/*.png")
    validate = c_validate["suit"]
    bars_validate = plt.barh(
        labels, validate.values(), label="Validate", left=list(train.values())
    )
    plt.bar_label(
        bars_validate, validate.values(), label_type="center", color=text_color
    )

    c_test = count_objects(data_dir=f"{TEST_DIR}/*/*.png")
    test = c_test["suit"]
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
    c_train = count_objects(data_dir=f"{TRAIN_DIR}/*/*.png")
    train = c_train["animal"]
    labels = list(train.keys())
    bars_train = plt.barh(labels, train.values(), label="Train")
    plt.bar_label(bars_train, train.values(), label_type="center", color=text_color)

    c_validate = count_objects(data_dir=f"{VALIDATE_DIR}/*/*.png")
    validate = c_validate["animal"]
    bars_validate = plt.barh(
        labels, validate.values(), label="Validate", left=list(train.values())
    )
    plt.bar_label(
        bars_validate, validate.values(), label_type="center", color=text_color
    )

    c_test = count_objects(data_dir=f"{TEST_DIR}/*/*.png")
    test = c_test["animal"]
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
    loss_ylim: tuple[int, int] = None,
    multiclass: bool = False,
    includes_validation: bool = False,
    onehot: bool = False,
) -> None:
    """Plot the history (accuracy and loss on the validation set) of a model as a line chart"""
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    idx_start, idx_end = index_slice
    accuracy_str = "accuracy" if not multiclass else "sparse_categorical_accuracy"
    accuracy_str = accuracy_str if not onehot else "categorical_accuracy"

    for history in histories:
        num_epochs = range(1, len(history["loss"]) + 1)
        acc.append(history[accuracy_str])
        loss.append(history["loss"])
        # Plot training & validation accuracy values
        axes[0][0].plot(
            num_epochs[idx_start:idx_end],
            history[accuracy_str][idx_start:idx_end],
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
        if loss_ylim:
            # axes[1][1].set_ylim(0, 1)
            axes[1][1].set_ylim(*loss_ylim)

    # Average of the histories
    avg_history = {
        accuracy_str: np.mean(acc, axis=0),
        "loss": np.mean(loss, axis=0),
    }

    # Plot average training & validation accuracy values
    axes[0][0].plot(
        num_epochs[idx_start:idx_end],
        avg_history[accuracy_str][idx_start:idx_end],
        color=color,
        alpha=alpha_mean,
        label=model_name,
    )
    axes[0][0].set_title("Accuracy")

    # Plot training & validation loss values
    axes[1][0].plot(
        num_epochs[idx_start:idx_end],
        avg_history["loss"][idx_start:idx_end],
        color=color,
        alpha=alpha_mean,
        label=model_name,
    )
    axes[1][0].set_title("Loss")

    if includes_validation:
        val_accuracy_str = "val_" + accuracy_str
        val_acc.append(history[val_accuracy_str])
        val_loss.append(history["val_loss"])
        axes[0][1].plot(
            num_epochs[idx_start:idx_end],
            history[val_accuracy_str][idx_start:idx_end],
            color=color,
            alpha=alpha_runs,
        )
        axes[1][1].plot(
            num_epochs[idx_start:idx_end],
            history["val_loss"][idx_start:idx_end],
            color=color,
            alpha=alpha_runs,
        )

        avg_history[val_accuracy_str] = np.mean(val_acc, axis=0)
        avg_history["val_loss"] = np.mean(val_loss, axis=0)
        axes[0][1].plot(
            num_epochs[idx_start:idx_end],
            avg_history[val_accuracy_str][idx_start:idx_end],
            color=color,
            alpha=alpha_mean,
            label=model_name,
        )
        axes[0][1].set_title("Val Accuracy")
        axes[1][1].plot(
            num_epochs[idx_start:idx_end],
            avg_history["val_loss"][idx_start:idx_end],
            color=color,
            alpha=alpha_mean,
            label=model_name,
        )
        axes[1][1].set_title("Val Loss")

    for a in axes[0]:
        a.set_ylabel("Accuracy")
        a.legend()
    for a in axes[1]:
        a.set_ylabel("Loss")
        a.set_xlabel("Epoch")
        a.legend()


def plot_evaluations_box(
    axes, evaluations_all: list, colors: list[str] = COLORS
) -> None:
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
            plt.setp(bp["medians"], color="white")
        # Add the legend AFTER setting the colors so the colors in the legend are accurate
        axes[ax].legend(bp["boxes"], model_names)


def plot_wrong_predictions(
    wrong_predictions: list[tuple[str, str, float, float]],
    model_name: str,
    show_num_wrong: int = 5,
) -> None:
    # Plot the wrong predictions by highest error rate (most wrong)
    wrong = np.array(wrong_predictions)
    wrong = wrong[wrong[:, 3].argsort()]  # Sort ascending by error rate
    wrong = wrong[::-1]  # Reverse the order so the most wrong is on top

    # Plot the wrong predictions
    plt.figure(figsize=(10, 5), dpi=100)
    for i in range(show_num_wrong):
        plt.subplot(1, show_num_wrong, i + 1)
        try:
            plt.imshow(
                keras.preprocessing.image.load_img(wrong[i][0], target_size=(600, 200))
            )
            label = wrong[i][1]
            # Nested as heck because `predict_image` returns an array[float] instead of float
            accuracy = f"{wrong[i][2][0][0]:.2f}"
            error = f"{wrong[i][3][0][0]:.2f}"
            plt.title(f"{label}\n(E:{error}, A:{accuracy})")
        except IndexError:
            # If there are less than `show_num_wrong` wrong predictions,
            # the remaining plots will be empty
            plt.imshow(np.zeros((600, 200, 3)))
        plt.axis("off")
    plt.suptitle(
        f" {len(wrong)} Wrong predictions: {model_name}",
    )
    plt.tight_layout()
    plt.show()


def plot_wrong_predictions_multiclass(
    wrong_predictions: list[tuple[str, str, str]],
    model_name: str,
    show_num_wrong: int = 5,
) -> None:
    wrong = np.asarray(wrong_predictions)

    # Plot the wrong predictions
    plt.figure(figsize=(10, 5), dpi=100)
    for i in range(show_num_wrong):
        plt.subplot(1, show_num_wrong, i + 1)
        try:
            img_fp, label_wrong, label_actual = wrong[i]
            plt.imshow(
                keras.preprocessing.image.load_img(img_fp, target_size=(600, 200))
            )
            plt.title(f"(P:{label_wrong}, A:{label_actual})")
        except IndexError:
            # If there are less than `show_num_wrong` wrong predictions,
            # the remaining plots will be empty
            plt.imshow(np.zeros((600, 200, 3)))
        plt.axis("off")
    plt.suptitle(
        f" {len(wrong)} Wrong predictions: {model_name}",
    )
    plt.tight_layout()
    plt.show()


def get_street_counters() -> dict[str, dict[str, dict]]:
    obj_names = {}
    for street in STREETS:
        obj_names[street] = []
        for xml_path in glob(f"{PROCESSED_DIR}/{street}/*.xml", recursive=True):
            for obj in extract_objects_from_xml(xml_path):
                obj_name, _, _, _, _ = obj
                obj_names[street].append(obj_name)

    street_counters = {
        street: count_objects(obj_names=obj_names[street]) for street in STREETS
    }
    return street_counters


def _help_plot_streets(
    ax: plt.Axes, counter_labels: list[str], counter_key: str, figsize=(6, 4)
):
    """Helper function to plot counters for each street

    Args:
        ax (plt.Axes): The axes to plot the counter on
        counter_labels (list[str]): The labels for the counter
        counter_key (str): The key for the counter to be plotted.
            Key must be in ['all', 'binary', 'suit', 'animal']
        figsize: The size of the figure

    Returns:
        ax: The axes with the plotted counter
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)

    counter = Counter()
    # Initialize all counters to 0
    counter.update({label: 0 for label in counter_labels})
    street_counters = get_street_counters()

    for i, street in enumerate(street_counters):
        street_counter = street_counters[street][counter_key]
        labels = list(street_counter.keys())
        bars = ax.barh(
            labels,
            street_counter.values(),
            color=COLORS_STREETS[i],
            left=list(counter.values()),
        )
        bar_labels = ax.bar_label(
            bars, list(street_counter.values()), label_type="center"
        )
        # TODO Remove if statement below when dataset is balanced
        if counter_key == "binary":  # Reduce crowding of labels for COUNTER_BINARY
            bar_labels = [
                label.set_text(None)
                for label in bar_labels
                if int(label.get_text()) < 20
            ]
        else:
            bar_labels = [
                label.set_text(None)
                for label in bar_labels
                if int(label.get_text()) == 0
            ]

        counter.update(street_counter)
    ax.grid(axis="x", linestyle="--", color="lightgrey", alpha=0.5)
    ax.invert_yaxis()
    return ax


def plot_streets_all(ax: plt.Axes = None, show_legend: bool = False) -> None:
    ax = _help_plot_streets(
        ax, counter_labels=ALL_LABELS, counter_key="all", figsize=(4, 8)
    )
    ax.set_title("Labels per street")
    if show_legend:
        ax.legend(STREETS)
    return ax


def plot_streets_binary(ax: plt.Axes = None, show_legend: bool = False) -> None:
    ax = _help_plot_streets(ax, counter_labels=BINARY, counter_key="binary")
    ax.set_title("Binary per street")
    if show_legend:
        ax.legend(STREETS)
    return ax


def plot_streets_suits(ax: plt.Axes = None, show_legend: bool = False) -> None:
    ax = _help_plot_streets(ax, counter_labels=SUITS_LONG, counter_key="suit")
    ax.set_title("Suits per street")
    if show_legend:
        ax.legend(STREETS)
    return ax


def plot_streets_animals(ax: plt.Axes = None, show_legend: bool = False) -> None:
    ax = _help_plot_streets(ax, counter_labels=ANIMALS, counter_key="animal")
    ax.set_title("Animals per street")
    if show_legend:
        ax.legend(STREETS)

    return ax


def plot_streets() -> None:
    gridspec_kw = dict(width_ratios=[1, 1.2], height_ratios=[1, 1, 1])
    fig, ax = plt.subplot_mosaic(
        [["left", "upper right"], ["left", "middle right"], ["left", "lower right"]],
        gridspec_kw=gridspec_kw,
        figsize=(10, 8),
        dpi=100,
    )

    counts_and_titles = [
        (plot_streets_all, "left"),
        (plot_streets_binary, "upper right"),
        (plot_streets_suits, "middle right"),
        (plot_streets_animals, "lower right"),
    ]

    fig.suptitle("Labels grouped by street", fontsize=20)
    for func_plot, subplot_key in counts_and_titles:
        func_plot(ax[subplot_key])

    fig.tight_layout()
    fig.legend(STREETS, loc="lower left")
    fig.show()


def plot_confusion_matrix(
    predictions: list[str],
    targets: list[str],
    display_labels: list[str],
    title: str = "",
) -> None:
    """Plot the confusion matrix for a list of predictions and targets"""

    matrix = confusion_matrix(y_pred=predictions, y_true=targets, labels=display_labels)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=display_labels
    )

    display.plot(include_values=True)
    display.ax_.set_title(title)

    fig = display.ax_.get_figure()
    fig.set_figwidth(8)
    fig.set_figheight(8)

    plt.show()


# %% Plot data
# plot_suits_as_bar()
# plot_toons_as_bar()
# plot_datasets_all()
# plot_datasets_binary()
# plot_datasets_suits()
# plot_datasets_animals()
# plot_streets_all()
# plot_streets_binary()
# plot_streets_suits()
# plot_streets_animals(show_legend=True)

# %%
# plot_xml_data
# plot_streets()
