# %% Imports
import keras
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf

from data_processing import (
    TEST_DIR,
    TRAIN_DIR,
    VALIDATE_DIR,
    process_images,
    split_data,
    SCREENSHOTS_DIR
)
from data_visualization import (
    plot_history,
    plot_image_sizes,
    plot_suits_as_bar,
    plot_toons_as_bar,
    plot_xml_data,
    compare_histories,
)
from model_utils import make_model

# %% Convert all images in screenshots directory to data images
process_images(raw_images_dir=SCREENSHOTS_DIR + "/br", move_images=False)

# %% Plot bar of suits
# plot_suits_as_bar()

# # %% Plot bar of toons
# plot_toons_as_bar()

# # %% Plot xml data
# plot_xml_data()

# # %% Plot all image sizes in unsorted directory
# plot_image_sizes(TRAIN_DIR)

# %% Split unsorted images into train, validate, and test sets
# from data_processing import split_data


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

# %% Define data augmentations
# data_augmentation = keras.Sequential(
#     [
#         # Apply horizontal flipping to 50% of the images
#         layers.RandomFlip("horizontal"),
#         # Rotate the input image by some factor in range [-10%, 10%] or [-36, 36] in degrees
#         layers.RandomRotation(0.1),
#         # Zoom in or out by a random factor in range [-20%, 20%]
#         layers.RandomZoom(0.2),
#     ]
# )

# %% Create models
# model = make_model(name="toonvision")
# model_augmentation = make_model(
#     name="toonvision_augmentation", augmentation=data_augmentation
# )
# model_augmentation_dropout = make_model(
#     name="toonvision_augmentation_dropout", augmentation=data_augmentation, dropout=0.5
# )
# all_models = [model, model_augmentation, model_augmentation_dropout]

# %% View model summary
# all_models[0].summary()

# %% Compile all models
learning_rate = 0.0001
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
    model = make_model(name="toonvision_" + opt._name)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    models_all.append(model)
# optimizer = keras.optimizers.adam_v2.Adam(learning_rate=0.0001)
# for model in all_models:
#     model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

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
# for model, history in all_histories:
#     plot_history(history=history.history, name=model.name)

# %% Test model
for model in models_all:
    model = keras.models.load_model(f"{model.name}.keras")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"{model.name} Test Loss: {test_loss:.2f}")
    print(f"{model.name} Test Accuracy: {test_accuracy:.2f}")

# %% Predict on unseen images
# TODO: Add code to predict on unseen images


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
compare_histories(histories_not_vanishing)

# %% Compare training histories for optimizers with vanishing gradients
compare_histories(histories_vanishing)

# %%
