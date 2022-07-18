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
    TEST_DIR,
    TRAIN_DIR,
    VALIDATE_DIR,
    create_suit_datasets,
    get_suits_from_dir,
    integer_to_suit,
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
    plot_streets,
    plot_suits_as_bar,
    plot_toons_as_bar,
    plot_wrong_predictions_multiclass,
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
create_suit_datasets(split_ratio=[0.7, 0.2, 0.1])

# %% Plot bar of suits
# plot_suits_as_bar(img_dir=DATA_DIR)

# # # %% Plot bar of toons
# plot_toons_as_bar(img_dir=DATA_DIR)

# # %% Plot xml data
# plot_xml_data()

# # % Plot street data
# plot_streets()

# %% Plot the balance of the datasets
# plot_datasets_suits()
# plot_datasets_animals()
# plot_datasets_binary()
# plot_datasets_all()


# %% Create the dataset
def img_to_array(img_path, height=600, width=200):
    """Process an image, given a filepath, and return a numpy array"""
    img = tf.keras.utils.load_img(img_path, target_size=(height, width))
    # img = tf.image.decode_png(tf.io.read_file(img_path), channels=3)
    # img.set_shape(tf.TensorShape([None, None, None]))
    img = tf.image.resize(img, [height, width])
    # img = tf.cast(img, tf.float32) / 255.0
    return img


batch_size = 64
filepaths, labels = get_suits_from_dir(directories=[TRAIN_DIR])[TRAIN_DIR]
train_fps, train_labels = filepaths, labels
train_labels = np.asarray(suit_to_integer(labels), dtype=np.float32)
train_labels_onehot = np.asarray(suit_to_onehot(labels), dtype=np.float32)
train_images = np.asarray([img_to_array(fp) for fp in train_fps], dtype=np.float32)

filepaths, labels = get_suits_from_dir(directories=[VALIDATE_DIR])[VALIDATE_DIR]
val_fps, val_labels = filepaths, labels
val_labels = np.asarray(suit_to_integer(labels), dtype=np.float32)
val_labels_onehot = np.asarray(suit_to_onehot(labels), dtype=np.float32)
val_images = np.asarray([img_to_array(fp) for fp in val_fps], dtype=np.float32)

# %% Shuffle the data
np.random.seed(42)
p = np.random.permutation(len(train_images))
train_images = train_images[p]
train_labels = train_labels[p]
p = np.random.permutation(len(val_images))
val_images = val_images[p]
val_labels = val_labels[p]

# %% Create validation set and training set
# val_labels = train_labels[:num_validate]
# val_images = train_images[:num_validate]
# train_labels = train_labels[num_validate:]
# train_images = train_images[num_validate:]

# %% Display a sample from the validation set
idx = np.random.randint(len(val_images))
plt.title(f"{val_labels[idx]}, {MAP_INT_TO_SUIT[val_labels[idx]]}")
plt.imshow(val_images[idx] / 255)

# %% Compile all models
optimizers = [
    tf.keras.optimizers.SGD(learning_rate=LR),
    tf.keras.optimizers.Adam(),
    tf.keras.optimizers.RMSprop(),
    # tf.keras.optimizers.Adadelta(),
    # tf.keras.optimizers.Adagrad(),
    tf.keras.optimizers.Adamax(),
    # tf.keras.optimizers.Nadam(),
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
    print(f"Training model {model.name}")
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=20,
        batch_size=batch_size,
        # shuffle=True,
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


fp = DATA_DIR + "\\test\\cog\\cog_lb_legaleagle_13.png"
pred = model.predict(get_img_array(fp, (600, 200, 3)))
l_actual = get_obj_details_from_filepath(fp)["suit"]
l_pred = MAP_INT_TO_SUIT[pred.argmax()]  # Given an int, get suit
l_actual, l_pred

# %% Plot wrong predictions on the TEST_SET

# Get the test set
filepaths, labels = get_suits_from_dir(directories=[TEST_DIR])[TEST_DIR]
test_fps, test_labels = filepaths, labels
test_labels = np.asarray(suit_to_integer(labels), dtype=np.float32)
test_labels_onehot = np.asarray(suit_to_onehot(labels), dtype=np.float32)
test_images = np.asarray([img_to_array(fp) for fp in test_fps], dtype=np.float32)

# %% Get the Adamax model
model = models_all[-1]

# %% Predict the test set
predictions = model.predict(test_images)
preds_int = np.asarray([np.argmax(p) for p in predictions], dtype=np.int32)
preds_str = integer_to_suit(preds_int)

# %% Get the wrong predictions as a True/False array
mask = preds_int.astype("int") != test_labels.astype("int")
wrong_idxs = np.argwhere(mask).transpose().flatten()

wrong_fps = [test_fps[i] for i in wrong_idxs]
wrong_preds = [preds_str[i] for i in wrong_idxs]
wrong_actual = integer_to_suit(test_labels[i] for i in wrong_idxs)
wrong = list(zip(wrong_fps, wrong_preds, wrong_actual))


# %% Plot the wrong predictions
for model in models_all:
    predictions = model.predict(test_images)
    preds_int = np.asarray([np.argmax(p) for p in predictions], dtype=np.int32)
    preds_str = integer_to_suit(preds_int)

    # %% Get the wrong predictions as a True/False array
    mask = preds_int.astype("int") != test_labels.astype("int")
    wrong_idxs = np.argwhere(mask).transpose().flatten()

    wrong_fps = [test_fps[i] for i in wrong_idxs]
    wrong_preds = [preds_str[i] for i in wrong_idxs]
    wrong_actual = integer_to_suit(test_labels[i] for i in wrong_idxs)
    wrong = list(zip(wrong_fps, wrong_preds, wrong_actual))

    plot_wrong_predictions_multiclass(wrong, model_name=model.name, show_num_wrong=10)
# %% Get the wrong predictions

# ! Wrongly-labeled image
# ttr-screenshot-Sun-May-29-22-18-51-2022-704947.png
# ttr-screenshot-Sun-May-29-22-18-51-2022-704947.xml