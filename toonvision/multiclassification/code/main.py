# %% Imports
from glob import glob

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_processing import (
    DATA_DIR,
    MAP_INT_TO_SUIT,
    MAP_SUIT_TO_INT,
    TRAIN_DIR,
    create_suit_datasets,
    get_suits_from_dir,
    process_images,
    split_data,
    suit_to_integer,
    suit_to_onehot,
    unsort_data,
)
from data_visualization import (
    COLORS,
    compare_histories,
    plot_datasets_all,
    plot_datasets_animals,
    plot_datasets_binary,
    plot_datasets_suits,
    plot_histories,
    plot_history,
    plot_suits_as_bar,
    plot_toons_as_bar,
    plot_xml_data,
)
from img_utils import get_obj_details_from_filepath
from keras import layers
from model_utils import make_multiclass_model_original

LR = 0.0001

# %% Convert all images in screenshots directory to data images
process_images(move_images=True)
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Fri-Jun-10")
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Sat-Jun-11")

# %% Split unsorted images into train, validate, and test sets
unsort_data()
ds_train, ds_validate, ds_test = create_suit_datasets(split_ratio=[0.8, 0.1, 0.1])

# %% Plot bar of suits
plot_suits_as_bar(img_dir=DATA_DIR)

# # %% Plot bar of toons
plot_toons_as_bar(img_dir=DATA_DIR)

# # %% Plot xml data
plot_xml_data()

# %% Plot the balance of the datasets
plot_datasets_suits()
plot_datasets_animals()
plot_datasets_binary()
plot_datasets_all()


# %% Create the dataset
def img_to_array(img_path, height=600, width=200):
    """Process an image, given a filepath, and return a numpy array"""
    img = tf.keras.utils.load_img(img_path, target_size=(height, width))
    # img = tf.image.decode_png(tf.io.read_file(img_path), channels=3)
    # img.set_shape(tf.TensorShape([None, None, None]))
    img = tf.image.resize(img, [height, width])
    # img = tf.cast(img, tf.float32) / 255.0
    return img


filepaths, labels = get_suits_from_dir(directories=[TRAIN_DIR])[TRAIN_DIR]
train_labels = np.asarray(suit_to_integer(labels), dtype=np.float32)
train_labels_onehot = np.asarray(suit_to_onehot(labels), dtype=np.float32)
train_images = np.asarray([img_to_array(fp) for fp in filepaths], dtype=np.float32)
batch_size = 32


# %% Compile all models
optimizers = [
    tf.keras.optimizers.SGD(learning_rate=LR),
    tf.keras.optimizers.Adam(),
    tf.keras.optimizers.RMSprop(),
    tf.keras.optimizers.Adadelta(learning_rate=LR),
    tf.keras.optimizers.Adagrad(learning_rate=LR),
    tf.keras.optimizers.Adamax(learning_rate=LR),
    tf.keras.optimizers.Nadam(learning_rate=LR),
    tf.keras.optimizers.Ftrl(learning_rate=LR),
]

models_all = []
for opt in optimizers:
    model = make_multiclass_model_original(name="tv_" + opt._name)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    models_all.append(model)

# %% Print model summary
print(model.summary())

# %% Train all model
histories_all = []  # List of tuples: tuple[model, history]
# evaluations_all = []  # List of tuples: tuple[model, evaluation]
for model in models_all:
    history = model.fit(
        train_images,
        train_labels,
        epochs=25,
        validation_split=0.2,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    histories_all.append((model, history))
    # evaluations_all.append((model, model.evaluate(ds_test, verbose=0)))


# %% Plot the training history
plot_history(history.history, name=model.name, multiclass=True)

# %% Plot all training histories
plt.figure(figsize=(10, 10), dpi=100)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for color, (model, hist) in zip(COLORS, histories_all):
    plot_histories(
        axes=axes,
        histories=[hist.history],
        model_name=model.name.replace("tv_", ""),
        color=color,
        alpha_runs=0.10,
        alpha_mean=0.75,
        multiclass=True,
    )


# %% Predict an image
def get_img_array(img_path, target_size):
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    array = tf.keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


fp = DATA_DIR + "\\test\\cog\\cog_sb_mrhollywood_41.png"
pred = model.predict(get_img_array(fp, (600, 200, 3)))
l_actual = get_obj_details_from_filepath(fp)["suit"]
l_pred = MAP_INT_TO_SUIT[pred.argmax()]  # Given an int, get suit
l_actual, l_pred
