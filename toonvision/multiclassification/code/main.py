# %% Imports
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers

from data_processing import (
    DATA_DIR,
    TRAIN_DIR,
    create_datasets,
    process_images,
    split_data,
    unsort_data,
)
from data_visualization import (
    plot_datasets_all,
    plot_datasets_animals,
    plot_datasets_binary,
    plot_datasets_suits,
    plot_suits_as_bar,
    plot_toons_as_bar,
    plot_xml_data,
)
from model_utils import make_model_padding


LR = 0.001

# %% Convert all images in screenshots directory to data images
# process_images(move_images=True)
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Fri-Jun-10")
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Sat-Jun-11")

# %% Split unsorted images into train, validate, and test sets
# unsort_data()
# ds_train, ds_validate, ds_test = create_datasets(split_ratio=[0.6, 0.2, 0.2])

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

# %% Create a model
model = make_model_padding(name="base")

# %%
from tensorflow.keras.utils import image_dataset_from_directory
from data_visualization import get_obj_details_from_filepath
from os import walk
TRAIN_DIR = DATA_DIR + "/train"


def get_suit_labels():
    filepaths = [fp for fp in walk(TRAIN_DIR + "/cog")][0][2]
    obj_details = [get_obj_details_from_filepath(fp) for fp in filepaths]
    labels = [cog['suit'] for cog in obj_details]
    return labels


ds_labels = get_suit_labels()

ds_train = image_dataset_from_directory(
    TRAIN_DIR + "/cog",
    labels=ds_labels,
    label_mode="int",
    image_size=(600, 200),
    batch_size=32,
)
# %%

""" # TODO

? Can I manually create the datasets instead of using `image_dataset_from_directory`
1. Create `create_suit_datasets` function
2. Have function teardown existing file structure and build suit-based file structure
3. Modify `create_datasets` to create binary (cog vs toon) file structure
"""