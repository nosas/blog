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


LR = 0.001

# %% Convert all images in screenshots directory to data images
# process_images(move_images=True)
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Fri-Jun-10")
# process_images(raw_images_dir=SCREENSHOTS_DIR, move_images=False, filename_filter="Sat-Jun-11")

# %% Split unsorted images into train, validate, and test sets
# unsort_data()
ds_train, ds_validate, ds_test = create_datasets(split_ratio=[0.6, 0.2, 0.2])

# %% Plot bar of suits
plot_suits_as_bar(img_dir=DATA_DIR)

# # %% Plot bar of toons
plot_toons_as_bar(img_dir=DATA_DIR)

# # %% Plot xml data
plot_xml_data()

# # %% Plot all image sizes in unsorted directory
# plt.clf()
# plot_image_sizes(TRAIN_DIR)

# %% Plot the balance of the datasets
plot_datasets_suits()
plot_datasets_animals()
plot_datasets_binary()
plot_datasets_all()
