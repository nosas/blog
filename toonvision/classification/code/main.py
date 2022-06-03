# %% Imports
import matplotlib.pyplot as plt
import keras
from keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

from data_processing import (TEST_DIR, TRAIN_DIR, VALIDATE_DIR, process_images,
                             split_data)
from data_visualization import (plot_history, plot_image_sizes,
                                plot_suits_as_bar, plot_toons_as_bar,
                                plot_xml_data)
from model_utils import make_model

# %% Convert all images in screenshots directory to data images
# process_images(move_images=False)

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
data_augmentation = keras.Sequential(
    [
        # Apply horizontal flipping to 50% of the images
        layers.RandomFlip("horizontal"),
        # Rotate the input image by some factor in range [-10%, 10%] or [-36, 36] in degrees
        layers.RandomRotation(0.1),
        # Zoom in or out by a random factor in range [-20%, 20%]
        layers.RandomZoom(0.2)
    ]
)

# %% Create models
model = make_model(
    name="toonvision"
)
model_augmentation = make_model(
    name="toonvision_augmentation",
    augmentation=data_augmentation
)
model_augmentation_dropout = make_model(
    name="toonvision_augmentation_dropout",
    augmentation=data_augmentation,
    dropout=0.5
)
all_models = [model, model_augmentation, model_augmentation_dropout]

# %% View model summary
all_models[0].summary()

# %% Compile all models
for model in all_models:
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# %% Train model
all_histories = []
for model in all_models:
    # Define the training callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f"{model.name}.keras",
            save_best_only=True,
            monitor="val_loss"
        )
    ]
    history = model.fit(
        train_dataset,
        epochs=20,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    all_histories.append((model.name, history))

# %% Plot training and validation loss
plot_history(history.history)

# %% Test model
for model_name, model in all_models:
    model = keras.models.load_model(f"{model_name}.keras")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"{model_name} Test Loss: {test_loss:.2f}")
    print(f"{model_name} Test Accuracy: {test_accuracy:.2f}")

# %% Predict on unseen images
# TODO: Add code to predict on unseen images
