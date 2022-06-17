# %% Imports
from glob import glob

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

from data_processing import (
    DATA_DIR,
    SCREENSHOTS_DIR,
    TEST_DIR,
    TRAIN_DIR,
    UNSORTED_DIR,
    VALIDATE_DIR,
    create_datasets,
    process_images,
    split_data,
    unsort_data,
)
from data_visualization import (
    compare_histories,
    plot_datasets_all,
    plot_datasets_animals,
    plot_datasets_binary,
    plot_datasets_suits,
    plot_history,
    plot_image_sizes,
    plot_suits_as_bar,
    plot_toons_as_bar,
    plot_xml_data,
)
from model_utils import make_model, make_model_optimized, predict_image

# %% Convert all images in screenshots directory to data images
# process_images()
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Fri-Jun-10")
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Sat-Jun-11")

# %% Split unsorted images into train, validate, and test sets
unsort_data()
ds_train, ds_validate, ds_test = create_datasets(split_ratio=[0.6, 0.2, 0.2])

# %% Plot bar of suits
plot_suits_as_bar(img_dir=DATA_DIR)

# # %% Plot bar of toons
plot_toons_as_bar(img_dir=DATA_DIR)

# # %% Plot xml data
plot_xml_data()

# # %% Plot all image sizes in unsorted directory
plot_image_sizes(TRAIN_DIR)

# %% Plot the balance of the datasets
plot_datasets_suits()
plot_datasets_animals()
plot_datasets_binary()
plot_datasets_all()

# %% Compile all models
learning_rate = 0.001
optimizers = [
    tf.keras.optimizers.SGD(learning_rate=learning_rate),
    tf.keras.optimizers.Adam(learning_rate=learning_rate),
    tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
    tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
    tf.keras.optimizers.Adamax(learning_rate=learning_rate),
    tf.keras.optimizers.Nadam(learning_rate=learning_rate),
    tf.keras.optimizers.Ftrl(learning_rate=learning_rate),
]

models_all = []
for opt in optimizers:
    model = make_model_optimized(name="toonvision_" + opt._name)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    models_all.append(model)

# %% Print model summary
print(model.summary())

# %% Train model
histories_all = []  # List of tuples: tuple[model, history]
for model in models_all:
    # Define the training callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f"{model.name}.keras", save_best_only=True, monitor="val_loss"
        )
    ]
    history = model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_validate,
        callbacks=callbacks,
    )
    histories_all.append((model, history))

# %% Compare training histories for all optimizers
compare_histories(histories_all)

# %% Remove optimizers with vanishing gradients
vanishing_gradients = ["Adadelta", "Adagrad", "Ftrl", "SGD"]
histories_vanishing = []
histories_not_vanishing = []
for model, history in histories_all:
    if model.name.strip("toonvision_") not in vanishing_gradients:
        histories_not_vanishing.append((model, history))
    else:
        histories_vanishing.append((model, history))

# %% Compare training histories for optimizers without vanishing gradients
compare_histories(
    histories_not_vanishing, suptitle="Optimizers without vanishing gradients"
)

# %% Compare training histories for optimizers with vanishing gradients
compare_histories(histories_vanishing, suptitle="Optimizers with vanishing gradients")

# %% Test all models
for model in models_all:
    model = keras.models.load_model(f"{model.name}.keras")
    test_loss, test_accuracy = model.evaluate(ds_test, verbose=0)
    print(f"Test Acc, Loss: {test_accuracy:.2f} {test_loss:.2f} {model.name}")


# %% Create function to repeatedly train models
def train_model(
    model: keras.Model,
    datasets: tuple,
    epochs: int,
    callbacks: list,
) -> tuple[list[dict], list[dict]]:
    # Define the training callbacks
    ds_train, ds_validate, ds_test = datasets
    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_validate,
        callbacks=callbacks,
    )
    evaluation = model.evaluate(ds_test, verbose=False)
    return (history.history, evaluation)


# %% Train model 5 times and plot the average of the histories
def make_baseline_comparisons(
    epochs: int,
    num_runs: int,
    model_kwargs: dict,
    learning_rate: float = 0.0001,
):
    histories_all = {model["name"]: [] for model in model_kwargs}
    evaluations_all = {model["name"]: [] for model in model_kwargs}
    for _ in range(num_runs):
        unsort_data()
        datasets = create_datasets(split_ratio=[0.6, 0.2, 0.2])
        for kwargs in model_kwargs:
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=f"toonvision_{kwargs['name']}.keras",
                    save_best_only=True,
                    monitor="val_loss",
                )
            ]
            model = make_model_optimized(**kwargs)
            if "baseline" in kwargs["name"]:
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            else:
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_rate, decay=1e-6
                )
            model.compile(
                loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
            history, evaluation = train_model(
                model=model,
                datasets=datasets,
                epochs=epochs,
                callbacks=callbacks,
            )
            # plot_history(histories, name=model.name)
            histories_all[model.name].append(history)
            evaluations_all[model.name].append(evaluation)

    # Convert dictionaries to tuples
    histories = [
        (model["name"], histories_all[model["name"]]) for model in model_kwargs
    ]
    evaluations = [
        (model["name"], evaluations_all[model["name"]]) for model in model_kwargs
    ]

    return histories, evaluations


# %% Plot the histories
def plot_histories(
    axes,
    model_name: str,
    histories: list,
    color: str,
    alpha_runs: float = 0.15,
    alpha_mean: float = 0.85,
    index_slice: tuple = (0, -1),
):
    """Plot the history (accuracy and loss) of a model"""
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
    for a in axes[0][:2]:
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
    for a in axes[1][:2]:
        a.set_ylabel("Loss")
        a.set_xlabel("Epoch")
        a.legend()


def plot_evaluations_line(
    axes,
    model_name: str,
    evaluations: list,
    color: str,
    alpha_runs: float = 0.15,
    index_slice: tuple = (0, -1),
):
    """Plot the history (accuracy and loss) of a model"""
    idx_start, idx_end = index_slice

    num_runs = range(1, len(evaluations) + 1)
    accuracies = [eval[1] for eval in evaluations]
    losses = [eval[0] for eval in evaluations]

    # Plot training & validation accuracy values
    axes[0][2].plot(
        num_runs[idx_start:idx_end],
        accuracies[idx_start:idx_end],
        color=color,
        alpha=alpha_runs,
        label=model_name,
    )
    # Plot training & validation loss values
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


def plot_evaluations_box(axes, evaluations_all: list, colors: list[str]):
    all_acc = np.array(
        [np.array(e).transpose()[1] for _, e in evaluations_all]
    ).transpose()
    all_loss = np.array(
        [np.array(e).transpose()[0] for _, e in evaluations_all]
    ).transpose()
    model_names = [e[0] for e in evaluations_all]

    bp_acc = axes[0, 2].boxplot(all_acc, notch=False, sym="o", patch_artist=True)
    bp_loss = axes[1, 2].boxplot(all_loss, notch=False, sym="o", patch_artist=True)
    axes[1, 2].set_xlabel("\nModel")

    for ax, (bp, label) in enumerate([(bp_acc, "Accuracy"), (bp_loss, "Loss")]):
        # axes[ax, 2].set_xticks(xticks_range, model_names, rotation=15)
        axes[ax, 2].set_title(f"Test {label}")
        axes[ax, 2].set_ylabel(f"{label}")
        axes[ax, 2].set_xticks([])
        axes[ax, 2].xaxis.grid(False)
        axes[ax, 2].yaxis.grid(
            True, linestyle="-", which="major", color="lightgrey", alpha=0.5
        )
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
        axes[ax, 2].legend(bp["boxes"], model_names)


# %% Compare training histories against baseline models
data_augmentation = keras.Sequential(
    [
        # Apply horizontal flipping to 50% of the images
        layers.RandomFlip("horizontal"),
        # Rotate the input image by some factor in range [-20%, 20%] or [-72, 72] in degrees
        layers.RandomRotation(0.2),
        # Zoom in or out by a random factor in range [-30%, 30%]
        layers.RandomZoom(0.3),
    ]
)
model_kwargs = [
    {"name": "baseline"},
    {"name": "augmentation", "augmentation": data_augmentation},
    {"name": "dropout80", "dropout": 0.80},
    {
        "name": "augmentation_dropout",
        "augmentation": data_augmentation,
        "dropout": 0.80,
    },
]

# %% Train each model for 100 epochs, and repeat it 10 times
histories_all, evaluations_all = make_baseline_comparisons(
    epochs=25, num_runs=50, learning_rate=0.001, model_kwargs=model_kwargs
)

# %% Plot the histories
plt.figure(figsize=(35, 15), dpi=1200)
plt.style.use("dark_background")
fig, axes = plt.subplots(2, 3, figsize=(35, 15))
for row in range(axes.shape[0]):
    for column in range(axes.shape[1]):
        axes[row, column].xaxis.grid(False)
        axes[row, column].yaxis.grid(
            True, linestyle="-", which="major", color="lightgrey", alpha=0.5
        )
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for color, (model_name, histories) in zip(colors, histories_all):
    plot_histories(
        axes,
        model_name=model_name,
        histories=histories,
        color=color,
        alpha_runs=0.10,
        alpha_mean=0.99,
    )
plot_evaluations_box(axes, evaluations_all, colors)
fig.show()


# %% Export the histories_all to a JSON file
import json

with open("histories_all.json", "w") as f:
    json.dump(histories_all, f)
with open("evaluations_all.json", "w") as f:
    json.dump(evaluations_all, f)


# %% Plot wrong predictions
preds_wrong = []
for model_name, _ in histories_all:
    model = keras.models.load_model(f"toonvision_{model_name}.keras")
    preds = []

    for fn in glob(f"{DATA_DIR}/**/cog/*.png", recursive=True):
        label, pred = predict_image(fn, model)
        if label == "Toon":
            preds.append((fn, label, pred, abs(pred - 0.5)))

    for fn in glob(f"{DATA_DIR}/**/toon/*.png", recursive=True):
        label, pred = predict_image(fn, model)
        if label == "Cog":
            preds.append((fn, label, pred, abs(pred - 0.5)))

    preds_wrong.append((model.name, preds))

# %% Plot the wrong predictions by highest error rate (most wrong)
for (model_name, pred_wrong) in preds_wrong:
    wrong = np.array(pred_wrong)
    wrong = wrong[wrong[:, 3].argsort()]
    wrong = wrong[::-1]

    # %% Plot the wong predictions by highest error rate (most wrong)
    plt.figure(figsize=(15, 15))
    for i in range(len(wrong[:10])):
        plt.subplot(3, 5, i + 1)
        plt.imshow(
            keras.preprocessing.image.load_img(wrong[i][0], target_size=(600, 200))
        )
        label = wrong[i][1]
        accuracy = f"{wrong[i][2][0][0]:.2f}"
        error = f"{wrong[i][3][0][0]:.2f}"
        plt.title(f"{label}\n(E:{error}, A:{accuracy})")
        plt.axis("off")
    plt.suptitle(f"Wrong predictions {len(wrong)} {model_name}")
    plt.tight_layout()
    plt.show()

# %%
