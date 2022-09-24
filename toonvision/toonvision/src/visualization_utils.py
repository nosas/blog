import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt


def create_image_grid(filepaths, ncols=4, title: str = "", titles: list[str] = None):
    """
    Create a grid of images given a list of filepaths.
    """
    nrows = int(np.ceil(len(filepaths) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3.25), dpi=100)
    for i, fp in enumerate(filepaths):
        if nrows == 1:
            ax[i].imshow(keras.preprocessing.image.load_img(fp))
            ax[i].axis("off")
            if titles:
                ax[i].set_title(titles[i])
        else:
            ax[int(i / ncols), i % ncols].imshow(keras.preprocessing.image.load_img(fp))
            ax[int(i / ncols), i % ncols].axis("off")
            if titles:
                ax[i].set_title(titles[i])

    # Add the title
    fig.suptitle(title, fontsize=16, y=1)
    fig.tight_layout()
    return fig, ax


def generate_heatmap(
    img_fp: str,
    model: keras.Model,
    layer_name_last_conv: str = "",
    class_id: int = None,
) -> tuple[np.array, np.array]:
    img = tf.keras.preprocessing.image.load_img(img_fp, target_size=(600, 200))
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
