# %% Imports
from glob import glob

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers

from data_processing import (
    DATA_DIR,
    TRAIN_DIR,
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
    plot_evaluations_box,
    plot_histories,
    plot_history,
    plot_image_sizes,
    plot_suits_as_bar,
    plot_toons_as_bar,
    plot_wrong_predictions,
    plot_xml_data,
)
from model_utils import (
    make_baseline_comparisons,
    make_model_optimized,
    predict_image,
    get_wrong_predictions,
)

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
LR = 0.001
plt.style.use("dark_background")

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
optimizers = [
    tf.keras.optimizers.SGD(learning_rate=LR),
    tf.keras.optimizers.Adam(learning_rate=LR),
    tf.keras.optimizers.RMSprop(learning_rate=LR),
    tf.keras.optimizers.Adadelta(learning_rate=LR),
    tf.keras.optimizers.Adagrad(learning_rate=LR),
    tf.keras.optimizers.Adamax(learning_rate=LR),
    tf.keras.optimizers.Nadam(learning_rate=LR),
    tf.keras.optimizers.Ftrl(learning_rate=LR),
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
evaluations_all = []  # List of tuples: tuple[model, evaluation]
for model in models_all:
    history = model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_validate,
    )
    histories_all.append((model, history))
    evaluations_all.append((model, model.evaluate(ds_test, verbose=0)))

# %% Test all models
for model, (test_loss, test_accuracy) in evaluations_all:
    print(f"Test Acc, Loss: {test_accuracy:.2f} {test_loss:.2f} {model.name}")

# %% Compare training histories for all optimizers
plt.figure(figsize=(30, 30), dpi=1200)
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
for color, (model, history) in zip(COLORS, histories_all):
    plot_histories(
        axes=axes,
        model_name=model.name,
        histories=[history.history],
        color=color,
        alpha_runs=0.10,
        alpha_mean=0.99,
    )
fig.show()

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
    {
        "name": "optimized",
        "augmentation": data_augmentation,
        "dropout": 0.90,
    },
    {
        "name": "optimized_1e-4",
        "augmentation": data_augmentation,
        "dropout": 0.90,
    },
    {
        "name": "optimized_1e-5",
        "augmentation": data_augmentation,
        "dropout": 0.90,
    },
]
optimizers = [
    tf.keras.optimizers.Adam(learning_rate=LR),
    tf.keras.optimizers.Adam(learning_rate=LR),
    tf.keras.optimizers.Adam(learning_rate=LR, decay=1e-4),
    tf.keras.optimizers.Adam(learning_rate=LR, decay=1e-5),
]
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=f"toonvision_{kwargs['name']}.keras",
        save_best_only=True,
        monitor="val_loss",
    )
    for kwargs in model_kwargs
]

# %% Train each model for 100 epochs, and repeat it 10 times
histories_all, evaluations_all = make_baseline_comparisons(
    epochs=25,
    num_runs=200,
    model_kwargs=model_kwargs,
    callbacks=callbacks,
    optimizers=optimizers,
)

# %% Plot the histories
plt.figure(figsize=(30, 30), dpi=1200)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for color, (model_name, histories) in zip(COLORS, histories_all):
    plot_histories(
        axes=axes,
        model_name=model_name,
        histories=histories,
        color=color,
        alpha_runs=0.05,
        alpha_mean=0.99,
    )
fig.show()

# %% Plot the evaluations
plt.figure(figsize=(30, 30), dpi=1200)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# plot_evaluations_box(axes, evaluations_all, colors)
plot_evaluations_box(axes=axes, evaluations_all=evaluations_all, colors=COLORS)
fig.show()

# %% Export the histories_all to a JSON file
# import json

# with open("histories_all.json", "w") as f:
#     json.dump(histories_all, f)
# with open("evaluations_all.json", "w") as f:
#     json.dump(evaluations_all, f)

# %% Retrieve and plot wrong predictions
# for model_name, _ in histories_all:
#     model = keras.models.load_model(f"toonvision_{model_name}.keras")
#     wrong_predictions = get_wrong_predictions(model)
#     plot_wrong_predictions(wrong_predictions, model_name)

# %% Retrieve and plot wrong predictions
unsort_data()
ds_train, ds_validate, ds_test = create_datasets(split_ratio=[0.6, 0.2, 0.2])
for model_name, run in [
    ("baseline", 19),
    ("optimized", 21),
    ("optimized_1e-4", 21),
    ("optimized_1e-5", 1),
]:
    model = keras.models.load_model(f"./models/toonvision_{model_name}_run{run}.keras")
    evaluation = model.evaluate(ds_test, verbose=False)
    print(f"{model_name} run {run}: {evaluation[1]:.2f} {evaluation[0]:.2f}")
    wrong_predictions = get_wrong_predictions(model)
    plot_wrong_predictions(wrong_predictions, model_name)

# %%
