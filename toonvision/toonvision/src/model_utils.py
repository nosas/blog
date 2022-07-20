# %% Imports and functions
from glob import glob

import keras
import numpy as np

# from tensorflow import keras
from keras import layers

from data_processing import DATA_DIR, create_datasets, unsort_data


def make_multiclass_model_original(
    name: str = "",
    augmentation: keras.Sequential = None,
    dropout: float = 0.0,
) -> keras.Model:
    inputs = keras.Input(shape=(600, 200, 3))
    if augmentation:
        x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(filters=8, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=8, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(units=4, activation="softmax")(x)
    model = keras.Model(name=name, inputs=inputs, outputs=outputs)
    return model


def make_multiclass_model_padding(
    name: str = "",
    augmentation: keras.Sequential = None,
    dropout: float = 0.0,
) -> keras.Model:
    inputs = keras.Input(shape=(600, 200, 3))
    if augmentation:
        x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=4, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(units=4, activation="softmax")(x)
    model = keras.Model(name=name, inputs=inputs, outputs=outputs)
    return model


def make_model_padding(
    name: str = "",
    augmentation: keras.Sequential = None,
    dropout: float = 0.0,
) -> keras.Model:
    inputs = keras.Input(shape=(600, 200, 3))
    if augmentation:
        x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=8, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(name=name, inputs=inputs, outputs=outputs)
    return model


def make_model_original(
    name: str = "",
    augmentation: keras.Sequential = None,
    dropout: float = 0.0,
) -> keras.Model:
    inputs = keras.Input(shape=(600, 200, 3))
    if augmentation:
        x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(name=name, inputs=inputs, outputs=outputs)
    return model


def make_model(
    model_func: callable,
    kwargs: dict = None,
) -> keras.Model:
    return model_func(**(kwargs or {}))


def predict_image(filename: str, model: keras.Model) -> tuple[str, np.array]:
    img = keras.preprocessing.image.load_img(filename, target_size=(600, 200))
    ia = keras.preprocessing.image.img_to_array(img)
    # plt.imshow(ia/255.)
    ia = np.expand_dims(ia, axis=0)
    classes = model.predict(ia)
    label = "Toon" if classes[0] > 0.5 else "Cog"
    # print(f"{label} : {filename}")
    return (label, classes)


# %% Create function to repeatedly train models
def train_model(
    model: keras.Model,
    datasets: tuple,
    epochs: int,
    callbacks: list = None,
) -> tuple[list[dict], tuple[float, float]]:
    """Train a model on a dataset and return the history and evaluation of the model"""
    ds_train, ds_validate, ds_test = datasets
    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_validate,
        callbacks=callbacks,
    )
    evaluation = model.evaluate(ds_test, verbose=False)
    return (history.history, evaluation)


# %% Train model 5 times and plot the average of the histories
def get_average_history(histories: list[dict]) -> dict:
    """Given a list of histories, return the average of the histories"""
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for history in histories:
        acc.append(history["accuracy"])
        val_acc.append(history["val_accuracy"])
        loss.append(history["loss"])
        val_loss.append(history["val_loss"])

    # Average of the histories
    avg_history = {
        "accuracy": np.mean(acc, axis=0),
        "val_accuracy": np.mean(val_acc, axis=0),
        "loss": np.mean(loss, axis=0),
        "val_loss": np.mean(val_loss, axis=0),
    }
    return avg_history


def make_baseline_comparisons(
    epochs: int,
    num_runs: int,
    model_kwargs: dict,
    train_kwargs: dict,
    split_ratio: list[float] = [0.6, 0.2, 0.2],
    save_best: bool = True,
) -> tuple[list[tuple[str, dict]], list[tuple[str, tuple[float, float]]]]:
    """Train model(s) for X num_runs and return the histories and evaluations"""
    model_names = [model['kwargs']['name'] for model in model_kwargs]
    histories_all = {model_name: [] for model_name in model_names}
    evaluations_all = {model_name: [] for model_name in model_names}
    evaluations_best = {model_name: (1, 0) for model_name in model_names}
    for run in range(num_runs):
        # Reshuffle the dataset before each run to ensure each model is trained on the same dataset
        unsort_data()
        datasets = create_datasets(split_ratio=split_ratio)
        # Train each model
        for kw_model, kw_train in zip(model_kwargs, train_kwargs):
            model = make_model(**kw_model)
            name = model.name
            model.compile(
                loss="binary_crossentropy",
                optimizer=kw_train["optimizer"],
                metrics=["accuracy"],
            )
            history, evaluation = train_model(
                model=model,
                datasets=datasets,
                epochs=epochs,
                callbacks=kw_train.get("callbacks", None),
            )
            loss, acc = evaluation
            # TODO Figure out how to retain the dataset of the best model
            # Possible send a seed to numpy?
            if save_best:
                if (loss < evaluations_best[name][0]) and (
                    acc > evaluations_best[name][1]
                ):
                    evaluations_best[name] = (loss, acc)
                    # Save the model
                    model.save(f"./models/toonvision_{name}_run{run}.keras")

            # plot_history(histories, name=model.name)
            histories_all[model.name].append(history)
            evaluations_all[model.name].append(evaluation)

    # Convert dictionaries to tuples of (model_name, histories) and (model_name, evaluations)
    histories = [
        (model_name, histories_all[model_name]) for model_name in model_names
    ]
    evaluations = [
        (model_name, evaluations_all[model_name]) for model_name in model_names
    ]
    return histories, evaluations


# %% Plot wrong predictions
def get_wrong_predictions(model: keras.Model) -> tuple[str, str, float, float]:
    """Retrieve the wrong predictions given a Keras classification model

    Args:
        model_name (str): _description_

    Returns:
        tuple[str, str, float, float]: filename, label, prediction_accuracy, prediction_error
    """
    preds_wrong = []
    for fn in glob(f"{DATA_DIR}/**/cog/*.png", recursive=True):
        label, pred = predict_image(fn, model)
        if label == "Toon":
            preds_wrong.append((fn, label, pred, abs(pred - 0.5)))
    for fn in glob(f"{DATA_DIR}/**/toon/*.png", recursive=True):
        label, pred = predict_image(fn, model)
        if label == "Cog":
            preds_wrong.append((fn, label, pred, abs(pred - 0.5)))
    return preds_wrong
