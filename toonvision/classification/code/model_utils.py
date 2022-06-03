# %% Imports and functions
import keras

# from tensorflow import keras
from keras import layers


def make_model(
    name: str = "", augmentation: keras.Sequential = None, dropout: float = 0.0
) -> keras.Model:
    inputs = keras.Input(shape=(600, 200, 3))
    if augmentation:
        x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(name=name, inputs=inputs, outputs=outputs)
    return model


# %% Make model
model = make_model()

# %%
