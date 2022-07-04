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
    # ("base_orig", 23),
    # ("base_pad", 7),
    # ("opt_orig", 17),
    # ("opt_pad", 2),
    ("optimized_1e-5", 12),
]:
    model = keras.models.load_model(f"./models/toonvision_{model_name}_run{run}.keras")
    evaluation = model.evaluate(ds_test, verbose=False)
    print(f"{model_name} run {run}: {evaluation[1]:.2f} {evaluation[0]:.2f}")
    wrong_predictions.append((model_name, get_wrong_predictions(model)))

# %% Plot the wrong predictions
for model_name, wrong_preds in wrong_predictions:
    plot_wrong_predictions(wrong_preds, model_name, show_num_wrong=5)


# %% Load the optimized model
model = keras.models.load_model("./models/toonvision_optimized_1e-5_run12.keras")

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
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# Get the activations of the last convolutional layer
activations_conv2d = activations[-1]
# Display the ninth channel of the activation
# This channel appears to encode the black pixels of the image
# plt.matshow(activations_conv2d[0, :, :, 9], cmap="viridis")

# %% Plot a complete visualization of all activations (feature maps) in the network
feature_maps = list(zip(layer_names[1:], activations[1:]))
n_maps = len(feature_maps)

for layer_name, layer_activation in feature_maps:
    n_features = layer_activation.shape[-1]
    img_size = (layer_activation.shape[1], layer_activation.shape[2])
    image_belt = np.zeros((img_size[0], img_size[1] * n_features))
    x1 = 0  # Starting x-coordinate for each feature map
    for i in range(n_features):
        # Normalize the image so it's human-visible
        feature_image = layer_activation[0, :, :, i]
        feature_image -= feature_image.mean()
        feature_image /= feature_image.std()
        feature_image *= 64
        feature_image += 128
        feature_image = np.clip(feature_image, 0, 255).astype("uint8")
        # Plot the image on the horizontal belt
        image_belt[:, x1 : x1 + img_size[1]] = feature_image
        # Increment the x-coordinate for the next feature map
        x1 += img_size[1]

    scale = 25.0 / n_features
    plt.figure(figsize=(scale * n_features, 2 * scale), dpi=100)
    plt.title(
        f"{layer_name}: {layer_activation.shape[1]}x{layer_activation.shape[2]} img_size, {n_features} features"
    )
    plt.grid(False)
    plt.axis("off")
    plt.imshow(image_belt, aspect="auto", cmap="inferno")
    # plt.imshow(image_belt, aspect="auto", cmap="gist_earth")


# %% Plot a complete visualization of all activations (feature maps) in the network
fig, ax = plt.subplots(n_maps, figsize=(10, 8), dpi=100)
for row, (layer_name, layer_activation) in enumerate(feature_maps):
    plt.subplot(n_maps, 1, row + 1)
    n_features = layer_activation.shape[-1]
    img_size = (layer_activation.shape[1], layer_activation.shape[2])
    image_belt = np.zeros((img_size[0], img_size[1] * n_features))
    x1 = 0  # Starting x-coordinate for each feature map
    for i in range(n_features):
        # Normalize the image so it's human-visible
        feature_image = layer_activation[0, :, :, i]
        feature_image -= feature_image.mean()
        feature_image /= feature_image.std()
        feature_image *= 64
        feature_image += 128
        feature_image = np.clip(feature_image, 0, 255).astype("uint8")
        # Plot the image on the horizontal belt
        image_belt[:, x1 : x1 + img_size[1]] = feature_image
        # Increment the x-coordinate for the next feature map
        x1 += img_size[1]

    scale = 25.0 / n_features
    # ax[row].figure(figsize=(scale * n_features, 2 * scale))
    plt.title(
        f"{layer_name}: {layer_activation.shape[1]}x{layer_activation.shape[2]} img_size, {n_features} features",
        fontsize=10,
    )
    plt.grid(False)
    plt.axis("off")
    ax[row].imshow(image_belt, cmap="inferno")
# plt.tight_layout()

# %% Plot wrong predictions
evaluation = model.evaluate(ds_test, verbose=False)
print(f"{model.name}  {evaluation[1]:.2f} {evaluation[0]:.2f}")
plot_wrong_predictions(get_wrong_predictions(model), model.name, show_num_wrong=5)

# %% Create a feature extractor model
layer_name = layer_names[-1]  # Second convolutional layer
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
# activation = feature_extractor(img_tensor)


# %% Use the feature extractor
def get_feature_extractor(layer_name):
    layer = model.get_layer(name=layer_name)
    feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
    return feature_extractor


def compute_loss(image, filter_index, feature_extractor) -> float:
    """Return a scalar quantifying how much a given input image "activates" a layer's filter"""
    # Get the activation of the selected filter
    activation = feature_extractor(image)
    filter_activation = activation[:, :, :, filter_index]
    # Compute the loss
    loss = tf.reduce_mean(filter_activation)
    return loss


@tf.function
def gradient_ascent_step(image, filter_index, learning_rate, feature_extractor):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, filter_index, feature_extractor)
    # Compute the gradient of the loss with respect to the image
    grads = tape.gradient(loss, image)
    # Smoothen the gradient descent process, ensure magnitude of updates are within same range
    grads = tf.math.l2_normalize(grads)
    # Update the image
    # Move the image in a direction that activates our target more strongly
    image += learning_rate * grads
    return image


def generate_filter_pattern(filter_index, feature_extractor) -> np.ndarray:
    iterations = 30
    learning_rate = 10.0
    image = tf.random.uniform(minval=0.4, maxval=0.6, shape=(1, 600, 200, 3))
    for i in range(iterations):
        image = gradient_ascent_step(
            image, filter_index, learning_rate, feature_extractor
        )
    return image[0].numpy()


def deprocess_image(image):
    # Normalize image values within [0, 255] range
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    image = image[25:-25, 25:-25, :]  # Center crop to remove border artifacts
    return image


# %% Generate a filter pattern for a given filter
filter_index = -16
filter_pattern = generate_filter_pattern(filter_index, feature_extractor)
filter_pattern = deprocess_image(filter_pattern)
plt.figure(figsize=(10, 10))
plt.imshow(filter_pattern)


# %% Generate a grid of images for all filter patterns
def generate_filter_grid(layer):
    feature_extractor = get_feature_extractor(layer.name)

    all_images = []
    for filter_index in range(layer.output.shape[-1]):
        filter_pattern = generate_filter_pattern(filter_index, feature_extractor)
        filter_pattern = deprocess_image(filter_pattern)
        all_images.append(filter_pattern)

    margin = 5
    n = len(all_images)
    img_per_row = 8
    num_rows = int(np.ceil(n / img_per_row))

    cropped_width = 200 - 25 * 2
    cropped_height = 600 - 25 * 2

    width = img_per_row * cropped_width + (img_per_row - 1) * margin
    height = num_rows * cropped_height + (num_rows - 1) * margin

    stitched_filters = np.zeros((height, width, 3))
    for col in range(img_per_row):
        for row in range(num_rows):
            filter_index = col * num_rows + row
            filter_pattern = all_images[filter_index]

            row_start = (cropped_height + margin) * row
            row_end = row_start + cropped_height
            col_start = (cropped_width + margin) * col
            col_end = col_start + cropped_width
            stitched_filters[row_start:row_end, col_start:col_end, :] = filter_pattern
    tf.keras.utils.save_img(f"filters_{layer.name}.png", stitched_filters)


# %% Generate a grid of images for each layer
for layer_name in layer_names[:]:
    layer = model.get_layer(name=layer_name)
    generate_filter_grid(layer)

# %% Make a gif of all the layers' filter patterns
import imageio


def make_gif(layer_names, filename):
    images = []
    for layer_name in layer_names:
        images.append(imageio.imread(f"../img/filters_{layer_name}.png"))
    imageio.mimsave(f"../img/{filename}.gif", images, duration=1)


# make_gif(layer_names[1:3], filename="filters_first_two")  # First two layers
# make_gif(layer_names[3:], filename="filters_remaining")  # Remaining layers

# %% Get filenames of wrong predictions
fns = []
for w in wrong_predictions[0][1]:
    fns.append(w[0])

# %% Display the image
from glob import glob
from random import choice

img_fp = choice(glob("../img/data/**/cog_*.png", recursive=True))
# img_fp = '../img/data\\test\\toon\\toon_cat_20.png'
# img_fp = '../img/data\\train\\toon\\toon_dog_12.png'
img = tf.keras.utils.load_img(img_fp, target_size=(600, 200))
img_title = img_fp.split("\\")[-1].replace(".png", "")
plt.title(img_title)
plt.axis("off")
plt.imshow(img)

# %% Expand dimensions of the model to (1, 600, 200, 3), a single-item batch of images
img = np.expand_dims(img, axis=0)

# %% Predict the class of the image
pred = model.predict(img)
pred_class = "Toon" if pred[0][0] > 0.5 else "Cog"
pred_class, pred[0][0]  # (0.2666575, 'Cog')

# %% Set up a model that returns the last convolutional layer's output
last_conv_layer_name = "max_pooling2d_211"
classifier_layer_names = ["flatten_52", "dropout_38", "dense_52"]
last_conv_layer = model.get_layer(name=last_conv_layer_name)
last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

# %% Reapply the classifier to the last convolutional layer's output
classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in classifier_layer_names:
    x = model.get_layer(name=layer_name)(x)
classifier_model = keras.Model(classifier_input, x)


# %% Retrieve the gradients of the top predicted class
with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img)
    tape.watch(last_conv_layer_output)
    pred = classifier_model(last_conv_layer_output)
    top_class_channel = pred[:, 0]

grads = tape.gradient(top_class_channel, last_conv_layer_output)

# %% Gradient pooling and chanel-importance weighting
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
last_conv_layer_output = last_conv_layer_output.numpy()[0]
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]
heatmap = np.mean(last_conv_layer_output, axis=-1)

# %% Heatmap post-processing: normalize and scale to [0, 1]
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap, cmap="jet")

# %% Superimpose the heatmap on the original image
import matplotlib.cm as cm

img = tf.keras.utils.load_img(img_fp, target_size=(600, 200))
img = tf.keras.utils.img_to_array(img)
heatmap = np.uint(255 * heatmap)

jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
plt.title(img_title)
plt.axis("off")
plt.imshow(superimposed_img)


# %% Create functions to streamline heatmap generation
def generate_heatmap(
    img_fp: str,
    model: keras.Model,
    layer_name_last_conv: str,
    layers_classifier: list[str],
) -> tuple[np.array, np.array]:
    img = tf.keras.utils.load_img(img_fp, target_size=(600, 200))
    img = np.expand_dims(img, axis=0)

    # %% Set up a model that returns the last convolutional layer's output
    last_conv_layer = model.get_layer(name=layer_name_last_conv)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # %% Reapply the classifier to the last convolutional layer's output
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in layers_classifier:
        x = model.get_layer(name=layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # %% Retrieve the gradients of the top predicted class
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img)
        tape.watch(last_conv_layer_output)
        pred = classifier_model(last_conv_layer_output)
        top_class_channel = pred[:, 0]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # %% Gradient pooling and chanel-importance weighting
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # %% Heatmap post-processing: normalize and scale to [0, 1]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # %% Superimpose the heatmap on the original image
    import matplotlib.cm as cm

    img = tf.keras.utils.load_img(img_fp, target_size=(600, 200))
    img = tf.keras.utils.img_to_array(img)
    unscaled_heatmap = np.uint(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[unscaled_heatmap]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    return heatmap, superimposed_img


# %% Select an image to generate a heatmap for
# toon_cat_20: Face, glove, hat, and backpack activation
# toon_duck_9: Face and glove activation
# toon_crocodile_2: Hat and shoes activation
# toon_mouse_3: Fairy wings and glove activation
img_fp = choice(glob("../img/data/**/toon_duck_9.png", recursive=True))
img = tf.keras.utils.load_img(img_fp, target_size=(600, 200))
img_title = img_fp.split("\\")[-1].replace(".png", "")
plt.title(img_title)
plt.axis("off")
plt.imshow(img)

# %% Generate the heatmaps and the superimposed image
layer_names = np.array([layer.name for layer in model.layers])
layer_name_last_conv = layer_names[-4]
layers_classifier = layer_names[-3:]
heatmap, superimposed_img = generate_heatmap(
    img_fp=img_fp,
    model=model,
    layer_name_last_conv=layer_name_last_conv,
    layers_classifier=layers_classifier,
)

# %% Display the original image, heatmap, and superimposed image
pred = model.predict(np.expand_dims(img, axis=0))[0][0]
pred_label = "Toon" if pred > 0.5 else "Cog"

figure, ax = plt.subplots(1, 3, figsize=(5, 5), dpi=100)
ax[0].imshow(img)
ax[0].set_title("Original", y=1.01)
ax[1].matshow(heatmap)
ax[1].set_title("Heatmap", y=1.01)
ax[2].imshow(superimposed_img)
ax[2].set_title("Superimposed", y=1.01)

for ax_idx in range(len(ax)):
    ax[ax_idx].axis("off")
figure.suptitle(f"{img_title}, {pred_label}, {pred:.2f}")
figure.tight_layout()


# %%
