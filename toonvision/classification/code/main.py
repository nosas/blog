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
    process_images,
    split_data,
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
from model_utils import make_model, predict_image

# %% Convert all images in screenshots directory to data images
process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Fri-Jun-10")
process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Sat-Jun-11")

# %% Split unsorted images into train, validate, and test sets
# split_data()

# %% Plot bar of suits
plot_suits_as_bar(img_dir=DATA_DIR)

# # %% Plot bar of toons
plot_toons_as_bar(img_dir=DATA_DIR)

# # %% Plot xml data
plot_xml_data()

# # %% Plot all image sizes in unsorted directory
plot_image_sizes(TRAIN_DIR)


# %% Read images into keras dataset
train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(600, 200),
    # batch_size=16
)
validation_dataset = image_dataset_from_directory(
    VALIDATE_DIR,
    image_size=(600, 200),
    # batch_size=16
)
test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=(600, 200),
    # batch_size=16
)


# %% Plot the balance of the datasets
plot_datasets_suits()
plot_datasets_animals()
plot_datasets_binary()
plot_datasets_all()

# %% Define data augmentations
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

# %% Compile all models
learning_rate = 0.0001
optimizers = [
    # tf.keras.optimizers.SGD(learning_rate=learning_rate * 100),
    # tf.keras.optimizers.Adam(learning_rate=learning_rate),
    # tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    # tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
    # tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
    # tf.keras.optimizers.Adamax(learning_rate=learning_rate),
    tf.keras.optimizers.Nadam(learning_rate=learning_rate),
    # tf.keras.optimizers.Ftrl(learning_rate=learning_rate),
]
models_all = []
for opt in optimizers:
    model = make_model(name="toonvision_" + opt._name)
    model_augmentation = make_model(
        name=f"toonvision_{opt._name}_augmentation", augmentation=data_augmentation
    )
    model_augmentation_dropout = make_model(
        name=f"toonvision_{opt._name}_augmentation_dropout",
        augmentation=data_augmentation,
        dropout=0.75,
    )

    models = [model, model_augmentation, model_augmentation_dropout]
    for model in models:
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
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        callbacks=callbacks,
    )
    histories_all.append((model, history))

# %% Plot training and validation loss
for model, history in histories_all:
    plot_history(history=history.history, name=model.name)

# %% Test model
for model in models_all:
    model = keras.models.load_model(f"{model.name}.keras")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"{model.name} Test Loss: {test_loss:.2f}")
    print(f"{model.name} Test Accuracy: {test_accuracy:.2f}")

# %% Compare training histories for all optimizers
compare_histories(histories_all)

# %% Predict on unseen images
preds_wrong = []
for model in models_all:
    model = keras.models.load_model(f"{model.name}.keras")
    preds = []

    for fn in glob(f"{DATA_DIR}/**/cog/*.png", recursive=True):
        label, pred = predict_image(fn, model)
        if label == "Toon":
            print(f"{fn}, {label} {pred}")
            preds.append((fn, label, pred, abs(pred - 0.5)))

    for fn in glob(f"{DATA_DIR}/**/toon/*.png", recursive=True):
        label, pred = predict_image(fn, model)
        if label == "Cog":
            print(f"{fn}, {label} {pred}")
            preds.append((fn, label, pred, abs(pred - 0.5)))

    preds_wrong.append((model.name, preds))

# %% Plot the wrong predictions by highest error rate (most wrong)
for (model_name, pred_wrong) in preds_wrong:
    wrong = np.array(pred_wrong)
    wrong = wrong[wrong[:, 3].argsort()]
    wrong = wrong[::-1]

    # %% Plot the wong predictions by highest error rate (most wrong)
    plt.figure(figsize=(15, 15))
    for i in range(len(wrong[:15])):
        plt.subplot(3, 5, i + 1)
        plt.imshow(
            keras.preprocessing.image.load_img(wrong[i][0], target_size=(600, 200))
        )
        label = wrong[i][1]
        error = f"{wrong[i][3][0][0]:.2f}"
        accuracy = f"{wrong[i][2][0][0]:.2f}"
        plt.title(f"{label}\n(E:{error}, A:{accuracy})")
        plt.axis("off")
    plt.suptitle(f"Wrong predictions {len(wrong)} {model_name}")
    plt.tight_layout()
    plt.show()

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
compare_histories(histories_not_vanishing)

# %% Compare training histories for optimizers with vanishing gradients
compare_histories(histories_vanishing)
