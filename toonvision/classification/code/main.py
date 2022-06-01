# %% Imports
from data_processing import process_images
from data_visualization import (
    plot_suits_as_bar,
    plot_toons_as_bar,
    plot_xml_data,
)

# %% Convert all images in screenshots directory to data images
# process_images(move_images=False)

# %% Plot bar of suits
plot_suits_as_bar()

# %% Plot bar of toons
plot_toons_as_bar()

# %% Plot xml data
plot_xml_data()

# %%
