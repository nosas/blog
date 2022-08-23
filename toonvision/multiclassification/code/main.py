# %% Imports
from glob import glob

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_processing import (
    SUITS_LONG,
    SUITS_SHORT,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATE_DIR,
    create_suit_datasets,
    get_suits_from_dir,
    integer_to_suit,
    onehot_to_suit,
    process_images,
    suit_to_integer,
    unsort_data,
)
from data_visualization import (
    COLORS,
    create_image_grid,
    plot_confusion_matrix,
    plot_histories,
    plot_history,
    plot_prediction_confidence,
    plot_streets_suits,
    plot_wrong_predictions_multiclass,
)
from img_utils import get_image_augmentations
from keras_tuner import BayesianOptimization, Hyperband, RandomSearch
from model_utils import (
    make_multiclass_model_original,
    make_multiclass_model_padding,
    make_multiclass_model_tuned,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

BATCH_SIZE = 64
LR = 0.001
SEED = 42
np.random.seed(SEED)

ONEHOT = True
label_decoder = integer_to_suit if not ONEHOT else onehot_to_suit
LOSS = (
    tf.keras.losses.SparseCategoricalCrossentropy()
    if not ONEHOT
    else tf.keras.losses.CategoricalCrossentropy()
)
METRICS = [
    tf.keras.metrics.SparseCategoricalAccuracy()
    if not ONEHOT
    else tf.keras.metrics.CategoricalAccuracy(),
]
if ONEHOT:
    METRICS += [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

# %% Convert all images in screenshots directory to data images
process_images(move_images=True)
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Fri-Jun-10")
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Sat-Jun-11")

# %% Split unsorted images into train, validate, and test sets
unsort_data()
ds_train, ds_validate, ds_test = create_suit_datasets(
    split_ratio=[0.6, 0.2, 0.2], onehot=ONEHOT
)

# %% Plot bar of suits
# plot_suits_as_bar(img_dir=DATA_DIR)

# # # %% Plot bar of toons
# plot_toons_as_bar(img_dir=DATA_DIR)

# # %% Plot xml data
# plot_xml_data()

# # % Plot street balance
# plot_streets_suits()

# # % Plot dataset balance
# plot_datasets()

# %% Create the dataset
train_images, train_labels = ds_train
val_images, val_labels = ds_validate
test_images, test_labels = ds_test

# %% Shuffle the data
p = np.random.permutation(len(train_images))
train_images = train_images[p]
train_labels = train_labels[p]
# Retrieve filepaths of test images, to be used in plotting of wrong predictions
train_fps, _ = get_suits_from_dir(directories=[TRAIN_DIR])[TRAIN_DIR]
train_fps = np.array(train_fps)[p]  # Apply permutation to match ds_train's ordering

p = np.random.permutation(len(val_images))
val_images = val_images[p]
val_labels = val_labels[p]
# Retrieve filepaths of test images, to be used in plotting of wrong predictions
val_fps, _ = get_suits_from_dir(directories=[VALIDATE_DIR])[VALIDATE_DIR]
val_fps = np.array(val_fps)[p]  # Apply permutation to match ds_val's ordering

p = np.random.permutation(len(test_images))
test_images = test_images[p]
test_labels = test_labels[p]
# Retrieve filepaths of test images, to be used in plotting of wrong predictions
test_fps, _ = get_suits_from_dir(directories=[TEST_DIR])[TEST_DIR]
test_fps = np.array(test_fps)[p]  # Apply permutation to match ds_test's ordering

# %% Display a sample from the validation set
# Should be a walking Cashbot-MoneyBags
idx = np.random.randint(len(val_images))
label_enc, label_str = val_labels[idx], label_decoder([val_labels[idx]])[0]
plt.title(f"{label_enc}, {label_str}")
plt.imshow(val_images[idx] / 255)

# %% Compile the optimized models
models_all = []


for opt in [tf.keras.optimizers.Adam()]:
    model = make_multiclass_model_padding(
        name="toonvision_multiclass_baseline", dropout=0.0
    )
    model.compile(
        loss=LOSS,
        optimizer=opt,
        metrics=METRICS,
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
        batch_size=BATCH_SIZE,
        # shuffle=True,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
        ],
    )
    histories_all.append((model, history))

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
        onehot=ONEHOT,
    )

# %% Plot the training history of all models
for model, history in histories_all:
    plot_history(
        history=history.history,
        name=model.name,
        multiclass=True,
        loss_ylim=(0, 1),
        onehot=ONEHOT,
    )
    # Evaluate the model against the test set
    print(model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE))

# %% Plot the wrong predictions
for model in models_all:
    predictions = model.predict(test_images)
    preds_int = np.asarray([np.argmax(p) for p in predictions], dtype=np.int32)
    preds_str = integer_to_suit(preds_int)

    # Get the wrong predictions as a True/False array
    # mask = np.argwhere(tf.keras.metrics.categorical_accuracy(test_labels, predictions) == 0)
    mask = preds_str != label_decoder(test_labels)
    wrong_idxs = np.argwhere(mask).transpose().flatten()

    wrong_fps = [test_fps[i] for i in wrong_idxs]
    wrong_preds = [preds_str[i] for i in wrong_idxs]
    wrong_actual = label_decoder(test_labels[i] for i in wrong_idxs)
    wrong = list(zip(wrong_fps, wrong_preds, wrong_actual))

    plot_wrong_predictions_multiclass(wrong, model_name=model.name, show_num_wrong=10)

# %% Plot the precision, recall and F1-score

# Print f1, precision, and recall scores
print(precision_score(label_decoder(test_labels), preds_str, average="macro"))
print(recall_score(label_decoder(test_labels), preds_str, average="macro"))
print(f1_score(label_decoder(test_labels), preds_str, average="macro"))
print(classification_report(label_decoder(test_labels), preds_str))
# Print confusion matrix
print(confusion_matrix(label_decoder(test_labels), preds_str))
plot_confusion_matrix(
    predictions=wrong_preds,
    targets=wrong_actual,
    display_labels=SUITS_SHORT,
    title=f"Wrong Predictions: {model.name}",
)
plot_confusion_matrix(
    predictions=preds_str,
    targets=label_decoder(test_labels),
    display_labels=SUITS_SHORT,
    title=f"Confusion matrix: {model.name}",
)


# %% Create a model builder to tune the hyperparameters
def model_builder(hp):
    model = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.Conv2D(
                filters=hp.Int("conv_1_filters", min_value=4, max_value=16, step=4),
                kernel_size=3,
                activation="relu",
                padding="same",
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_1_size", min_value=2, max_value=4, step=1),
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_2_size", min_value=1, max_value=4, step=1),
            ),
            keras.layers.Dropout(
                rate=hp.Float("dropout_1_rate", min_value=0.0, max_value=0.5, step=0.1),
            ),
            keras.layers.Conv2D(
                filters=hp.Int("conv_2_filters", min_value=4, max_value=20, step=4),
                kernel_size=3,
                activation="relu",
                padding="same",
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_3_size", min_value=2, max_value=4, step=1),
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_4_size", min_value=1, max_value=4, step=1),
            ),
            keras.layers.Flatten(),
            keras.layers.Dropout(
                rate=hp.Float("dropout_2_rate", min_value=0.0, max_value=0.9, step=0.1),
            ),
            keras.layers.Dense(units=4, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=LOSS,
        metrics=METRICS,
    )
    return model


# %%
tuner = RandomSearch(
    model_builder,
    objective="val_loss",
    max_trials=25,
    executions_per_trial=1,
    directory="models",
    project_name="multiclass_performance",
    seed=SEED,
    overwrite=False,  # Set to False to load previous trials
)
tuner.search_space_summary()

# %%
tuner.search(
    train_images,
    train_labels,
    epochs=75,
    batch_size=BATCH_SIZE,
    validation_data=(val_images, val_labels),
    verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard("./tb_logs/multiclass_performance/"),
    ],
)
# %%
# model = tuner.get_best_models(num_models=1)[0]
# model.build(input_shape=(None, 600, 200, 3))
# 1, 4, 7
# 10, 13, 14, 14, 17, 18
best_hps = tuner.get_best_hyperparameters(num_trials=25)[2:3]
for hp_id, hp in enumerate(best_hps):
    model = tuner.hypermodel.build(hp)
    # hp.values

    # %% Train the best model
    hist = model.fit(
        np.concatenate((train_images, val_images)),
        np.concatenate((train_labels, val_labels)),
        epochs=100,
        batch_size=BATCH_SIZE,
        # shuffle=True,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True
            ),
        ],
    )

    # Plot the wrong predictions
    predictions = model.predict(test_images)
    preds_int = np.asarray([np.argmax(p) for p in predictions], dtype=np.int32)
    preds_str = integer_to_suit(preds_int)

    # Get the wrong predictions as a True/False array
    mask = preds_str != label_decoder(test_labels)
    wrong_idxs = np.argwhere(mask).transpose().flatten()
    num_wrong = len(wrong_idxs)
    # if num_wrong < 5:
    # Print the tuned model's summary
    model.summary()

    print(hp_id, ":", num_wrong, "wrong predictions")

    wrong_fps = [test_fps[i] for i in wrong_idxs]
    wrong_preds = [preds_str[i] for i in wrong_idxs]
    wrong_actual = label_decoder(test_labels[i] for i in wrong_idxs)
    wrong = list(zip(wrong_fps, wrong_preds, wrong_actual))
    plot_wrong_predictions_multiclass(
        wrong, model_name=f"random_final_{hp_id}", show_num_wrong=5
    )

    # Print f1, precision, and recall scores
    print(precision_score(label_decoder(test_labels), preds_str, average="macro"))
    print(recall_score(label_decoder(test_labels), preds_str, average="macro"))
    print(f1_score(label_decoder(test_labels), preds_str, average="macro"))
    print(classification_report(label_decoder(test_labels), preds_str))
    # Print confusion matrix
    print(confusion_matrix(label_decoder(test_labels), preds_str))
    plot_confusion_matrix(
        predictions=wrong_preds,
        targets=wrong_actual,
        display_labels=SUITS_SHORT,
        title=f"Wrong Predictions: {model.name}",
    )
    plot_confusion_matrix(
        predictions=preds_str,
        targets=label_decoder(test_labels),
        display_labels=SUITS_SHORT,
        title=f"All Predictions: {model.name}",
    )
    # del model

# %%
model.save("./models/tuned_randomsearch_tv_Adam_3.keras")


# %% Create a model builder to tune the hyperparameters
def model_builder_bayesian(hp):
    model = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.Conv2D(
                filters=hp.Int("conv_1_filters", min_value=8, max_value=16, step=4),
                kernel_size=hp.Int("conv_1_kernel", min_value=3, max_value=5, step=2),
                activation="relu",
                padding="valid",
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_1_size", min_value=2, max_value=4, step=1),
            ),
            keras.layers.Dropout(
                rate=hp.Float("dropout_1_rate", min_value=0.0, max_value=0.2, step=0.1),
            ),
            keras.layers.Conv2D(
                filters=hp.Int("conv_2_filters", min_value=4, max_value=16, step=4),
                kernel_size=hp.Int("conv_2_kernel", min_value=3, max_value=5, step=2),
                activation="relu",
                padding="valid",
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_3_size", min_value=2, max_value=4, step=1),
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_4_size", min_value=1, max_value=4, step=1),
            ),
            keras.layers.Flatten(),
            keras.layers.Dropout(
                rate=hp.Float("dropout_2_rate", min_value=0.0, max_value=0.7, step=0.1),
            ),
            keras.layers.Dense(units=4, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-3, 1e-4])
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


# %%
tuner = BayesianOptimization(
    hypermodel=model_builder_bayesian,
    objective="val_loss",
    max_trials=500,
    executions_per_trial=3,
    directory="models",
    project_name="tuned_multiclass_bayesian_final",
    seed=SEED,
)
tuner.search_space_summary()

# %%
tuner.search(
    train_images,
    train_labels,
    epochs=100,
    batch_size=BATCH_SIZE,
    validation_data=(val_images, val_labels),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard("./tb_logs/bayesian_final/"),
    ],
)
# %%
# model = tuner.get_best_models(num_models=1)[0]
# model.build(input_shape=(None, 600, 200, 3))
params = tuner.get_best_hyperparameters(num_trials=5)[1]
model = tuner.hypermodel.build(params)
params.values

# %% Train the best model
hist = model.fit(
    np.concatenate((train_images, val_images)),
    np.concatenate((train_labels, val_labels)),
    epochs=100,
    batch_size=BATCH_SIZE,
    # shuffle=True,
    verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        ),
    ],
)

# Print the tuned model's summary
model.summary()

# Plot all training histories
plt.figure(figsize=(10, 10), dpi=100)
plot_history(
    history=hist.history,
    name=model.name.replace("tv_", ""),
    multiclass=True,
    loss_ylim=(0, 1),
    includes_validation=False,
)

# Plot the wrong predictions
predictions = model.predict(test_images)
preds_int = np.asarray([np.argmax(p) for p in predictions], dtype=np.int32)
preds_str = label_decoder(preds_int)

# Get the wrong predictions as a True/False array
mask = preds_int.astype("int") != test_labels.astype("int")
wrong_idxs = np.argwhere(mask).transpose().flatten()

wrong_fps = [test_fps[i] for i in wrong_idxs]
wrong_preds = [preds_str[i] for i in wrong_idxs]
wrong_actual = label_decoder(test_labels[i] for i in wrong_idxs)
wrong = list(zip(wrong_fps, wrong_preds, wrong_actual))

plot_wrong_predictions_multiclass(wrong, model_name=model.name, show_num_wrong=10)


# %% Create a model builder to tune the hyperparameters
def model_builder_hyperband(hp):
    model = keras.Sequential(
        name="hyperband",
        layers=[
            keras.layers.RandomFlip("horizontal"),
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.Conv2D(
                filters=16,
                kernel_size=hp.Choice("conv_1_kernel_size", [3, 5]),
                activation="relu",
                padding=hp.Choice("padding_1", ["same", "valid"]),
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_1_size", min_value=2, max_value=4, step=1),
            ),
            keras.layers.Dropout(
                rate=hp.Float("dropout_1_rate", min_value=0.0, max_value=0.3, step=0.1),
            ),
            keras.layers.Conv2D(
                filters=12,
                kernel_size=hp.Choice("conv_2_kernel_size", [3, 5]),
                activation="relu",
                padding=hp.Choice("padding_2", ["same", "valid"]),
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_3_size", min_value=2, max_value=4, step=1),
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_4_size", min_value=1, max_value=4, step=1),
            ),
            keras.layers.Flatten(),
            keras.layers.Dropout(
                rate=hp.Float("dropout_2_rate", min_value=0.0, max_value=0.8, step=0.1),
            ),
            keras.layers.Dense(units=4, activation="softmax"),
        ],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


# %%

tuner = Hyperband(
    hypermodel=model_builder_hyperband,
    objective="val_loss",
    max_epochs=100,
    executions_per_trial=1,
    directory="models",
    project_name="tuned_multiclass_hyperband3",
    seed=SEED,
)
tuner.search_space_summary()

# %%
tuner.search(
    train_images,
    train_labels,
    epochs=100,
    batch_size=BATCH_SIZE,
    validation_data=(val_images, val_labels),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard("./tb_logs/hyperband3/"),
    ],
)
# %%
# model = tuner.get_best_models(num_models=1)[0]
# model.build(input_shape=(None, 600, 200, 3))
params = tuner.get_best_hyperparameters(num_trials=7)[6]
model = tuner.hypermodel.build(params)
params.values

# %% Train the best model
hist = model.fit(
    np.concatenate((train_images, val_images)),
    np.concatenate((train_labels, val_labels)),
    epochs=100,
    batch_size=BATCH_SIZE,
    # shuffle=True,
    verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        ),
    ],
)

# Print the tuned model's summary
model.summary()

# Plot all training histories
plt.figure(figsize=(10, 10), dpi=100)
plot_history(
    history=hist.history,
    name=model.name.replace("tv_", ""),
    multiclass=True,
    loss_ylim=(0, 1),
    includes_validation=False,
)

#  Plot the wrong predictions
predictions = model.predict(test_images)
preds_int = np.asarray([np.argmax(p) for p in predictions], dtype=np.int32)
preds_str = label_decoder(preds_int)

# Get the wrong predictions as a True/False array
mask = preds_int.astype("int") != test_labels.astype("int")
wrong_idxs = np.argwhere(mask).transpose().flatten()

wrong_fps = [test_fps[i] for i in wrong_idxs]
wrong_preds = [preds_str[i] for i in wrong_idxs]
wrong_actual = label_decoder(test_labels[i] for i in wrong_idxs)
wrong = list(zip(wrong_fps, wrong_preds, wrong_actual))

plot_wrong_predictions_multiclass(wrong, model_name=model.name, show_num_wrong=10)


# %% Create a model builder to tune the hyperparameters
def model_builder_hyperband_final(hp):
    model = keras.Sequential(
        name="hyperband",
        layers=[
            keras.layers.RandomFlip("horizontal"),
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.Conv2D(
                filters=16,
                kernel_size=3,
                activation="relu",
                padding="valid",
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_1_size", min_value=3, max_value=4, step=1),
            ),
            keras.layers.Conv2D(
                filters=12,
                kernel_size=5,
                activation="relu",
                padding="valid",
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_3_size", min_value=2, max_value=4, step=1),
            ),
            keras.layers.MaxPooling2D(
                pool_size=hp.Int("pool_4_size", min_value=1, max_value=4, step=1),
            ),
            keras.layers.Flatten(),
            keras.layers.Dropout(
                rate=hp.Float("dropout_rate", min_value=0.3, max_value=0.8, step=0.1),
            ),
            keras.layers.Dense(units=4, activation="softmax"),
        ],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=LOSS,
        metrics=METRICS,
    )
    return model


# %%

tuner = Hyperband(
    hypermodel=model_builder_hyperband_final,
    objective="val_loss",
    max_epochs=100,
    executions_per_trial=2,
    directory="models",
    project_name="tuned_multiclass_hyperband_final2",
    seed=SEED,
)
tuner.search_space_summary()

# %%
tuner.search(
    train_images,
    train_labels,
    epochs=100,
    batch_size=BATCH_SIZE,
    validation_data=(val_images, val_labels),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard("./tb_logs/hyperband_final/"),
    ],
)

# %%
# model = tuner.get_best_models(num_models=1)[0]
# model.build(input_shape=(None, 600, 200, 3))
params = tuner.get_best_hyperparameters(num_trials=10)[0]
model = tuner.hypermodel.build(params)
params.values

# %% Train the best model
hist = model.fit(
    np.concatenate((train_images, val_images)),
    np.concatenate((train_labels, val_labels)),
    epochs=100,
    batch_size=BATCH_SIZE,
    # shuffle=True,
    verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        ),
    ],
)

# Print the tuned model's summary
model.summary()

# Plot all training histories
plt.figure(figsize=(10, 10), dpi=100)
plot_history(
    history=hist.history,
    name=model.name.replace("tv_", ""),
    multiclass=True,
    loss_ylim=(0, 1),
    includes_validation=False,
    onehot=ONEHOT,
)

# Plot the wrong predictions
# model = keras.models.load_model("./models/tuned_hyperband2_tv_Adam.keras")

predictions = model.predict(test_images)
preds_int = np.asarray([np.argmax(p) for p in predictions], dtype=np.int32)
preds_str = integer_to_suit(preds_int)

# Get the wrong predictions as a True/False array
# mask = np.argwhere(tf.keras.metrics.categorical_accuracy(test_labels, predictions) == 0)
mask = preds_str != label_decoder(test_labels)
wrong_idxs = np.argwhere(mask).transpose().flatten()

wrong_fps = [test_fps[i] for i in wrong_idxs]
wrong_preds = [preds_str[i] for i in wrong_idxs]
wrong_actual = label_decoder(test_labels[i] for i in wrong_idxs)
wrong = list(zip(wrong_fps, wrong_preds, wrong_actual))

plot_wrong_predictions_multiclass(wrong, model_name=model.name, show_num_wrong=10)

# %%
model.save("./models/tuned_hyperband_final_tv_Adam.keras")

# %% Plot the wrong predictions

for model_fp in glob("./models/*.keras"):
    model = tf.keras.models.load_model(model_fp)
    # model.summary()
    predictions = model.predict(test_images)
    preds_int = np.asarray([np.argmax(p) for p in predictions], dtype=np.int32)
    preds_str = integer_to_suit(preds_int)

    # Get the wrong predictions as a True/False array
    mask = preds_str != label_decoder(test_labels)
    wrong_idxs = np.argwhere(mask).transpose().flatten()

    wrong_fps = [test_fps[i] for i in wrong_idxs]
    wrong_preds = [preds_str[i] for i in wrong_idxs]
    wrong_actual = label_decoder(test_labels[i] for i in wrong_idxs)
    wrong = list(zip(wrong_fps, wrong_preds, wrong_actual))
    model_name = model_fp.split("\\")[-1].replace(".keras", "")
    plot_wrong_predictions_multiclass(wrong, model_name=model_name, show_num_wrong=10)
    print(
        classification_report(
            y_true=label_decoder(test_labels), y_pred=preds_str, labels=SUITS_SHORT
        )
    )
    plot_confusion_matrix(
        predictions=wrong_preds,
        targets=wrong_actual,
        display_labels=SUITS_SHORT,
        title=f"Wrong Predictions: {model.name}",
    )
    plot_confusion_matrix(
        predictions=preds_str,
        targets=label_decoder(test_labels),
        display_labels=SUITS_SHORT,
        title=f"All Predictions: {model.name}",
    )

# %% Plot the wrong predictions
model = keras.models.load_model("./models/tuned_hyperband_final_tv_Adam.keras")
# model = make_multiclass_model_tuned(name="toonvision_multiclass_tuned")
# model.compile(
#     loss=LOSS,
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     metrics=METRICS
# )
# print(f"Training model {model.name}")
# history = model.fit(
#     np.concatenate((train_images, val_images)),
#     np.concatenate((train_labels, val_labels)),
#     epochs=250,
#     # batch_size=BATCH_SIZE,
#     # shuffle=True,
#     verbose=0,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor="loss", patience=15, restore_best_weights=True
#         ),
#     ],
# )
# plot_history(
#     history=history.history,
#     name=model.name,
#     multiclass=True,
#     loss_ylim=(0, 1),
#     includes_validation=False,
#     onehot=ONEHOT,
# )
# # Evaluate the model against the test set
# print(model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE))

predictions = model.predict(test_images)
preds_int = np.asarray([np.argmax(p) for p in predictions], dtype=np.int32)
preds_str = integer_to_suit(preds_int)

# Get the wrong predictions as a True/False array
# mask = np.argwhere(tf.keras.metrics.categorical_accuracy(test_labels, predictions) == 0)
mask = preds_str != label_decoder(test_labels)
wrong_idxs = np.argwhere(mask).transpose().flatten()

wrong_fps = [test_fps[i] for i in wrong_idxs]
wrong_preds = [preds_str[i] for i in wrong_idxs]
wrong_actual = label_decoder(test_labels[i] for i in wrong_idxs)
wrong = list(zip(wrong_fps, wrong_preds, wrong_actual))

plot_wrong_predictions_multiclass(
    wrong, model_name="toonvision_multiclass_tuned", show_num_wrong=2
)
print(
    classification_report(
        y_true=label_decoder(test_labels), y_pred=preds_str, labels=SUITS_SHORT
    )
)
plot_confusion_matrix(
    predictions=wrong_preds,
    targets=wrong_actual,
    display_labels=SUITS_SHORT,
    title=f"Wrong Predictions: {model.name}",
)
plot_confusion_matrix(
    predictions=preds_str,
    targets=label_decoder(test_labels),
    display_labels=SUITS_SHORT,
    title=f"All Predictions: {model.name}",
)

# %% Plot the top 10 least confident predictions
# Get the indices of the 10 least confident predictions
lc_preds_values = np.amax(predictions, axis=1)
lc_preds_idxs = np.argsort(lc_preds_values)
lc10_preds_idxs = lc_preds_idxs[:10]
# sorted(lc_preds_values)[:10] == np.amax(predictions[lc10_preds_idxs], axis=1)
# Get the corresponding filepath, prediction
lc10_fps = test_fps[lc10_preds_idxs]
lc10_fps_short = [fp.split("\\")[-1].strip(".png") for fp in lc10_fps]
lc10_preds = predictions[lc10_preds_idxs]

# %% Plot the least confident predictions
results = {lc10_fps_short[i]: lc10_preds[i] for i in range(len(lc10_fps_short))}

colors = ["brown", "darkblue", "darkgreen", "maroon"]
fig, ax = plot_prediction_confidence(
    results=results,
    category_colors=colors,
    title="Confidence levels for least confident predictions",
)
plt.show()


# %% Plot confidence intervals for wrong predictions
wrong_fps_short = [fp.split("\\")[-1].strip(".png") for fp in wrong_fps]
results = {
    wrong_fps_short[i]: predictions[wrong_idxs][i] for i in range(len(wrong_preds))
}

fig, ax = plot_prediction_confidence(
    results=results,
    category_colors=colors,
    title="Confidence levels for wrong predictions",
)
plt.show()

# %% Create image grid given image filepaths
create_image_grid(lc10_fps[:5], ncols=5, title="Least confident samples")


# %% Create functions to streamline heatmap generation
def generate_heatmap(
    img_fp: str,
    model: keras.Model,
    layer_name_last_conv: str = "",
    class_id: int = None,
) -> tuple[np.array, np.array]:
    img = tf.keras.utils.load_img(img_fp, target_size=(600, 200))
    img = np.expand_dims(img, axis=0)

    layers_conv = [
        layer
        for layer in model.layers
        if "conv" in layer.name or "pooling" in layer.name
    ]
    layers_classifier = [
        layer
        for layer in model.layers
        if "flatten" in layer.name or "dropout" in layer.name or "dense" in layer.name
    ]
    if not layer_name_last_conv:
        layer_name_last_conv = layers_conv[-1].name  # actually a MaxPooling2D layer

    # Set up a model that returns the last convolutional layer's output
    last_conv_layer = model.get_layer(name=layer_name_last_conv)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Reapply the classifier to the last convolutional layer's output
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in layers_classifier:
        x = model.get_layer(name=layer.name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Retrieve the gradients of the class
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img)
        tape.watch(last_conv_layer_output)
        pred = classifier_model(last_conv_layer_output)
        if class_id is None:
            class_id = np.argmax(pred)  # Predicted class
        top_class_channel = pred[:, class_id]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # Gradient pooling and channel-importance weighting
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # Heatmap post-processing: normalize and scale to [0, 1]
    heatmap = np.maximum(heatmap, 0)
    if not (heatmap == 0).all():
        heatmap = heatmap / np.max(heatmap)

    # Superimpose the heatmap on the original image
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


# %% Display the original image, heatmap, and superimposed image
img_fps = [
    "../../toonvision/img/data/test\\cog\\cog_cb_loanshark_24.png",  # Wrong pred #1
    "../../toonvision/img/data/test\\cog\\cog_lb_spindoctor_19.png",  # Wrong pred #2
    "../../toonvision/img/data/test\\cog\\cog_sb_namedropper_16.png",  # Least confident #1
    "../../toonvision/img/data/test\\cog\\cog_sb_telemarketer_8.png",  # Least confident #2
]

for img_fp in img_fps:
    # img_fp = choice(img_fps)
    img = tf.keras.utils.load_img(img_fp, target_size=(600, 200))
    img_title = img_fp.split("\\")[-1].replace(".png", "")
    plt.title(img_title)
    plt.axis("off")
    plt.imshow(img)

    # Generate heatmap and superimposed image

    pred = model.predict(np.expand_dims(img, axis=0))

    heatmaps = []
    super_imgs = []
    figs = []
    axs = []
    for class_id in range(4):
        heatmap, superimposed_img = generate_heatmap(
            img_fp=img_fp,
            model=model,
            class_id=class_id
        )
        figure, ax = plt.subplots(1, 3, figsize=(5, 5), dpi=100)
        ax[0].imshow(img)
        ax[0].set_title("Original", y=1.01)
        ax[1].matshow(heatmap)
        ax[1].set_title("Heatmap", y=1.01)
        ax[2].imshow(superimposed_img)
        ax[2].set_title("Superimposed", y=1.01)
        plt.tight_layout()
        heatmaps.append(heatmap)
        super_imgs.append(superimposed_img)
        figs.append(figure)
        axs.append(ax)

        for ax_idx in range(len(ax)):
            ax[ax_idx].axis("off")
        figure.suptitle(
            f"{img_title}, P:{integer_to_suit([np.argmax(pred)])[0]}, A:{integer_to_suit([class_id])[0]}"
        )

    # Create 2x2 grid of heatmaps and superimposed images
    figure, ax = plt.subplots(2, 4, figsize=(10, 10), dpi=100)
    suit_pred = integer_to_suit([np.argmax(pred)])[0]
    for i, suit_actual in enumerate(SUITS_SHORT):
        ax[0, i].imshow(heatmaps[i])
        ax[0, i].set_title(f"Heatmap {i}: {suit_actual}", y=1.01)
        ax[1, i].imshow(super_imgs[i])
        ax[1, i].set_title(f"Superimposed {i}: {suit_actual}", y=1.01)
        ax[0, i].axis("off")
        ax[1, i].axis("off")

    figure.suptitle(f"{img_title}, Predicted as {integer_to_suit([np.argmax(pred)])[0]}")
    plt.tight_layout()
    plt.show()
