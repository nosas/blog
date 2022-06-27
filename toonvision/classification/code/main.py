# %% Imports
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
    COLORS,
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
    make_model,
    make_model_original,
    make_model_padding,
    get_wrong_predictions,
    get_average_history,
)

LR = 0.001

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
# plt.clf()
# plot_image_sizes(TRAIN_DIR)

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
    model = make_model(name="toonvision_" + opt._name)
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
plt.figure(figsize=(10, 10), dpi=100)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for color, (model, history) in zip(COLORS, histories_all):
    plot_histories(
        axes=axes,
        histories=[history.history],
        model_name=model.name.replace("toonvision_", ""),
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
        histories_not_vanishing.append((model.name, history.history))
    else:
        histories_vanishing.append((model.name, history.history))

# %% Compare training histories for optimizers without vanishing gradients
# compare_histories(
#     histories_not_vanishing, suptitle="Optimizers without vanishing gradients"
# )
plt.figure(figsize=(10, 10), dpi=100)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for color, (model_name, history) in zip(COLORS, histories_not_vanishing):
    plot_histories(
        axes=axes,
        histories=[history],
        model_name=model_name.replace("toonvision_", ""),
        color=color,
        alpha_runs=0.10,
        alpha_mean=0.99,
    )
fig.show()

# %% Compare training histories for optimizers with vanishing gradients
compare_histories(histories_vanishing, suptitle="Optimizers with vanishing gradients")

# %% Define data augmentation and models to use
data_augmentation = keras.Sequential(
    [
        # Apply horizontal flipping to 50% of the images
        layers.RandomFlip("horizontal"),
        # Rotate the input image by some factor in range [-7.5%, 7.5%] or [-27, 27] in degrees
        layers.RandomRotation(0.075),
        # Zoom in or out by a random factor in range [-20%, 20%]
        layers.RandomZoom(0.2),
    ]
)
model_kwargs = [
    {
        "model_func": make_model_original,
        "kwargs": {
            "name": "base_orig",
        },
    },
    {
        "model_func": make_model_padding,
        "kwargs": {
            "name": "base_pad",
        },
    },
    {
        "model_func": make_model_original,
        "kwargs": {
            "name": "opt_orig",
            "augmentation": data_augmentation,
            "dropout": 0.90,
        },
    },
    {
        "model_func": make_model_padding,
        "kwargs": {
            "name": "opt_pad",
            "dropout": 0.90,
        },
    },
]

train_kwargs = [
    {"optimizer": tf.keras.optimizers.Adam(learning_rate=LR)},
    {"optimizer": tf.keras.optimizers.Adam(learning_rate=LR)},
    {"optimizer": tf.keras.optimizers.Adam(learning_rate=LR, decay=1e-5)},
    {"optimizer": tf.keras.optimizers.Adam(learning_rate=LR, decay=1e-5)},
]

# %% Train each model for 25 epochs, and repeat it for 200 runs
histories_all, evaluations_all = make_baseline_comparisons(
    epochs=25,
    num_runs=100,
    model_kwargs=model_kwargs,
    train_kwargs=train_kwargs,
)

# %% Plot the histories
plt.figure(figsize=(10, 10), dpi=100)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
h = [histories_all[0], histories_all[-1]]
for color, (model_name, histories) in zip(COLORS, histories_all):
    plot_histories(
        axes=axes,
        model_name=model_name,
        histories=histories,
        color=color,
        alpha_runs=0.03,
        alpha_mean=0.99,
    )
fig.show()

# %% Plot the evaluations
plt.figure(figsize=(10, 10), dpi=100)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
e = [evaluations_all[0], evaluations_all[-1]]
# plot_evaluations_box(axes, evaluations_all, colors)
plot_evaluations_box(axes=axes, evaluations_all=evaluations_all, colors=COLORS)
fig.show()

# %% Export the histories_all to a JSON file
# import json

# with open("histories_all_new_pad.json", "w") as f:
#     json.dump(histories_all, f)
# with open("evaluations_all_new_pad.json", "w") as f:
#     json.dump(evaluations_all, f)

# %% Load the histories_all from a JSON file
# import json

# with open("histories_all.json", "r") as f:
#     histories_all = json.load(f)
# with open("evaluations_all.json", "r") as f:
#     evaluations_all = json.load(f)


# %% Plot the average history of the baseline model
baseline_histories = histories_all[0][1]
run = 99 + 1
plot_history(baseline_histories[0], f"Baseline (run #{run-1})")
baseline_avg = get_average_history(baseline_histories)
plot_history(baseline_avg, "Baseline (average)")

# %% Plot the average history of the optimized model
optimized_histories = histories_all[-1][1]
run = 2
plot_history(optimized_histories[run], f"Optimized 1e-5 (run #{run + 1})")
optimized_avg = get_average_history(optimized_histories)
plot_history(optimized_avg, "Optimized 1e-5 (average)")

# %% Compare the baseline to the optimized
# compare_histories(
#     [("baseline", baseline_histories), ("optimized", optimized_histories)],
#     suptitle="Baseline vs Optimized",
# )

# %% Compare the baseline to the optimized (average)
compare_histories(
    [("baseline", baseline_avg), ("optimized", optimized_avg)],
    suptitle="Baseline vs Optimized",
)

# %% Retrieve and plot wrong predictions
# for model_name, _ in histories_all:
#     model = keras.models.load_model(f"toonvision_{model_name}.keras")
#     wrong_predictions = get_wrong_predictions(model)
#     plot_wrong_predictions(wrong_predictions, model_name)

# %% Retrieve wrong predictions for each model
unsort_data()
ds_train, ds_validate, ds_test = create_datasets(split_ratio=[0.6, 0.2, 0.2])
wrong_predictions = []

for model_name, run in [
    ("base_orig", 23),
    ("base_pad", 7),
    ("opt_orig", 17),
    ("opt_pad", 2),
]:
    model = keras.models.load_model(f"./models/toonvision_{model_name}_run{run}.keras")
    evaluation = model.evaluate(ds_test, verbose=False)
    print(f"{model_name} run {run}: {evaluation[1]:.2f} {evaluation[0]:.2f}")
    wrong_predictions.append((model_name, get_wrong_predictions(model)))

# %% Plot the wrong predictions
for model_name, wrong_preds in wrong_predictions:
    plot_wrong_predictions(wrong_preds, model_name, show_num_wrong=5)


# %% Load the optimized model
model = keras.models.load_model("./models/toonvision_opt_pad_run2.keras")

# %% Load an example image and turn it into a tensor
import numpy as np


def get_img_array(img_path, target_size):
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    array = tf.keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


# fname = '../img/data\\train\\cog\\cog_lb_bloodsucker_15.png'
# img_tensor = get_img_array(fname, target_size=(600, 200))
# img_tensor = ds_train.take(1)
# img_tensor = img_tensor[0]
for image, label in ds_train.take(1):
    img_tensor = image.numpy().astype("uint8")
    plt.imshow(img_tensor[0])
    plt.axis("off")
    plt.show()

# %% Visualize the model's intermediate convnet outputs
from tensorflow.keras import layers

layer_outputs = []
layer_names = []
for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        # if isinstance(layer, (layers.Conv2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# Get the activations of the last convolutional layer
activations_conv2d = activations[-1]
# Display the ninth channel of the activation
# This channel appears to encode the black pixels of the image
# plt.matshow(activations_conv2d[0, :, :, 8], cmap="viridis")

# %% Plot a complete visualization of all activations in the network
images_per_row = 2
for layer_name, layer_activation in zip(layer_names[:], activations[:]):
    n_features = layer_activation.shape[-1]
    r, c = layer_activation.shape[1], layer_activation.shape[2]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((images_per_row * (r + 1) - 1, (c + 1) * n_cols - 1))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy()
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[
                row * (r + 1) : (row + 1) * r + row,
                col * (c + 1) : (col + 1) * c + col,
            ] = channel_image

    # scale = r * (1.25)./r
    # scale = r * (1.25)./r
    scale = 0.1
    plt.figure(figsize=(scale * 600, scale * 200))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis")


# %%
