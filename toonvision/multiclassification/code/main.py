# %% Imports
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_processing import (
    TEST_DIR,
    create_suit_datasets,
    get_suits_from_dir,
    integer_to_suit,
    process_images,
    unsort_data,
)
from data_visualization import (
    COLORS,
    plot_histories,
    plot_history,
    plot_wrong_predictions_multiclass,
)
from img_utils import get_image_augmentations
from model_utils import make_multiclass_model_original, make_multiclass_model_padding
from kerastuner import RandomSearch

LR = 0.001

# %% Convert all images in screenshots directory to data images
process_images(move_images=True)
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Fri-Jun-10")
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Sat-Jun-11")

# %% Split unsorted images into train, validate, and test sets
unsort_data()
ds_train, ds_validate, ds_test = create_suit_datasets(split_ratio=[0.6, 0.2, 0.2])
batch_size = 64

# %% Plot bar of suits
# plot_suits_as_bar(img_dir=DATA_DIR)

# # # %% Plot bar of toons
# plot_toons_as_bar(img_dir=DATA_DIR)

# # %% Plot xml data
# plot_xml_data()

# # % Plot street balance
# plot_streets()

# # % Plot dataset balance
# plot_datasets()

# %% Create the dataset
train_images, train_labels = ds_train
val_images, val_labels = ds_validate
test_images, test_labels = ds_test

# %% Shuffle the data
np.random.seed(42)
p = np.random.permutation(len(train_images))
train_images = train_images[p]
train_labels = train_labels[p]
p = np.random.permutation(len(val_images))
val_images = val_images[p]
val_labels = val_labels[p]
p = np.random.permutation(len(test_images))
test_images = test_images[p]
test_labels = test_labels[p]
# Retrieve filepaths of test images, to be used in plotting of wrong predictions
test_fps, _ = get_suits_from_dir(directories=[TEST_DIR])[TEST_DIR]
test_fps = np.array(test_fps)[
    p
]  # Apply permutation to match the order of the test dataset

# %% Display a sample from the validation set
idx = np.random.randint(len(val_images))
label_int, label_str = val_labels[idx], integer_to_suit([int(val_labels[idx])])[0]
plt.title(f"{label_int}, {label_str}")
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

# %% Train all models
histories_all = []  # List of tuples: tuple[model, history]
# evaluations_all = []  # List of tuples: tuple[model, evaluation]
for model in models_all:
    print(f"Training model {model.name}")
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=25,
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

# %% Plot wrong predictions on the test set
for model in models_all:
    # %% Predict the test set
    predictions = model.predict(test_images)
    preds_int = np.asarray([np.argmax(p) for p in predictions], dtype=np.int32)
    preds_str = integer_to_suit(preds_int)

    # %% Get the wrong predictions as a True/False array
    mask = preds_int.astype(int) != test_labels.astype(int)
    wrong_idxs = np.argwhere(mask).transpose().flatten()

    wrong_fps = [test_fps[i] for i in wrong_idxs]
    wrong_preds = [preds_str[i] for i in wrong_idxs]
    wrong_actual = integer_to_suit(test_labels[i] for i in wrong_idxs)
    wrong = list(zip(wrong_fps, wrong_preds, wrong_actual))

    plot_wrong_predictions_multiclass(wrong, model_name=model.name, show_num_wrong=10)

# %% Compile the optimized models
models_all = []
for opt in [tf.keras.optimizers.Adam(), tf.keras.optimizers.RMSprop()]:
    model = make_multiclass_model_padding(
        name="opt_tv_" + opt._name, augmentation=get_image_augmentations(), dropout=0.5
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=opt,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    models_all.append(model)
print(model.summary())


# %% Train all models
histories_all = []  # List of tuples: tuple[model, history]
for model in models_all:
    print(f"Training model {model.name}")
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=100,
        batch_size=batch_size,
        # shuffle=True,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f"models/{model.name}.keras",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            ),
        ],
    )
    histories_all.append((model, history))
    # evaluations_all.append((model, model.evaluate(ds_test, verbose=0)))

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
        loss_ylim=(0, 1),
    )

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


# %%
for model, history in histories_all:
    plot_history(
        history=history.history, name=model.name, multiclass=True, loss_ylim=(0, 1)
    )
# %%
model.evaluate(test_images, test_labels, batch_size=batch_size)

# %%


def model_builder(hp):
    model = keras.Sequential(
        [
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.Conv2D(
                filters=hp.Int("conv_1_filters", min_value=4, max_value=16, step=2),
                kernel_size=hp.Choice("conv_1_kernel_size", values=[3, 5]),
                activation="relu",
                padding="same",
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_1_size", min_value=1, max_value=4, step=1),
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_2_size", min_value=1, max_value=4, step=1),
            ),
            keras.layers.Dropout(
                rate=hp.Float("dropout_1_rate", min_value=0.0, max_value=0.8, step=0.1),
            ),
            keras.layers.Conv2D(
                filters=hp.Int("conv_2_filters", min_value=4, max_value=16, step=2),
                kernel_size=hp.Choice("conv_2_kernel_size", values=[3, 5]),
                activation="relu",
                padding="same",
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_3_size", min_value=1, max_value=4, step=1),
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_4_size", min_value=1, max_value=4, step=1),
            ),
            keras.layers.Flatten(),
            keras.layers.Dropout(
                rate=hp.Float("dropout_2_rate", min_value=0.0, max_value=0.8, step=0.1),
            ),
            keras.layers.Dense(units=4, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


# %%
tuner = RandomSearch(
    model_builder,
    objective="val_loss",
    max_trials=50,
    executions_per_trial=3,
    directory="models",
    project_name="tuner_multiclass_cog",
    seed=42,
)
tuner.search_space_summary()

# %%
tuner.search(
    train_images,
    train_labels,
    epochs=25,
    batch_size=batch_size,
    validation_data=(val_images, val_labels),
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)
# %%
# model = tuner.get_best_models(num_models=1)[0]
params = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(params)

# %% Train the best model
hist = model.fit(
    np.concatenate([train_images, val_images], axis=0),
    np.concatenate([train_labels, val_labels], axis=0),
    epochs=50,
    batch_size=batch_size,
    # shuffle=True,
    verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"models/tuned_{model.name}.keras",
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ],
)

# %%
model.summary()

# %% Plot all training histories
plt.figure(figsize=(10, 10), dpi=100)
plot_history(
    history=hist.history,
    name=model.name.replace("tv_", ""),
    multiclass=True,
    loss_ylim=(0, 1),
    includes_validation=False,
)

# %% Plot the wrong predictions
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
# %%
