# %% Create plot of each unique animal in the Toon dataset
from data_visualization import get_obj_details_from_filepath
from random import shuffle
animals = {}
animal_fns = []
fps = glob(f"{DATA_DIR}/**/toon/*.png", recursive=True)
shuffle(fps)
for fn in fps:
    obj_details = get_obj_details_from_filepath(fn)
    animal = obj_details['animal']
    if animal not in animals:
        animals[animal] = fn

plt.figure(figsize=(15, 15))
for i, animal in enumerate(sorted(animals.keys())):
    plt.subplot(2, 6, i + 1)
    plt.title(animal, fontsize=25)
    plt.imshow(keras.preprocessing.image.load_img(animals[animal], target_size=(600, 200)))
    plt.axis("off")
plt.tight_layout()
plt.show()

# %% Create plot of each unique suit in the Cog dataset
from data_visualization import get_obj_details_from_filepath
from random import shuffle

suits = {}
fps = glob(f"{DATA_DIR}/**/cog/*.png", recursive=True)
shuffle(fps)
for fn in fps:
    obj_details = get_obj_details_from_filepath(fn)
    suit = obj_details['suit']
    if suit not in suits:
        suits[suit] = fn

from data_visualization import SUITS_MAP

plt.figure(figsize=(15, 15))
for i, suit in enumerate(['bb', 'lb', 'cb', 'sb']):
    plt.subplot(3, 5, i + 1)
    plt.title(SUITS_MAP[suit], fontsize=25)
    plt.imshow(keras.preprocessing.image.load_img(suits[suit], target_size=(600, 200)))
    plt.axis("off")
plt.tight_layout()
plt.show()


# %% Draw bounding boxes on the image for each animal
from data_visualization import draw_bounding_boxes

fp = "..//img//raw//screenshots//ttc//ttr-screenshot-Sat-Jun-11-11-11-03-2022-14523.png"
draw_bounding_boxes("..//img//raw//screenshots//ttc//ttr-screenshot-Sat-Jun-11-11-11-03-2022-14523.png")

# %% Plot img_scale in 1 row and 4 columns, use the image size for the labels

imgs = {
    # "very close": "..//img//data//train//toon//toon_cat_33.png",
    "close": "..//img//data//train//toon//toon_cat_33.png",
    "far": "..//img//data//train//toon//toon_cat_17.png",
    "very far": "..//img//data//train//toon//toon_cat_25.png"}

fig = plt.figure(figsize=(8, 6))
plt.subplot(1, len(imgs), 1)
for idx, (distance, img_fp) in enumerate(imgs.items()):
    plt.subplot(1, len(imgs), idx + 1)
    img = keras.preprocessing.image.load_img(img_fp)
    plt.imshow(img, aspect="auto")
    title = f"{distance}\n({img.size[0]}x{img.size[1]})"
    plt.title(title, fontsize=15, color="#adff2f", weight="bold")
    plt.axis("off")
plt.tight_layout()
fig.patch.set_facecolor("#1d1d1d")
plt.savefig("../img/image_size_scale.png", bbox_inches='tight', pad_inches=0)
plt.show()

# %% Visualize the model's layers
import visualkeras as vk
from PIL import ImageFont

model = make_model("visualized")
font = ImageFont.truetype("arial.ttf", 16)
vk.layered_view(model, legend=True, font=font, scale_xy=0.5)

# %% Perform data augmentation on a single image 8 times, and plot them in a grid
import numpy as np

data_augmentation = keras.Sequential(
    [
        # Apply horizontal flipping to 50% of the images
        layers.RandomFlip("horizontal"),
        # Rotate the input image by some factor in range [-20%, 20%] or [-72, 72] in degrees
        layers.RandomRotation(0.075),
        # Zoom in or out by a random factor in range [-30%, 30%]
        layers.RandomZoom(0.2),
    ]
)
# Too harsh of an augmentation
data_augmentation1 = keras.Sequential(
    [
        # Apply horizontal flipping to 50% of the images
        layers.RandomFlip("horizontal"),
        # Rotate the input image by some factor in range [-20%, 20%] or [-72, 72] in degrees
        layers.RandomRotation(0.2),
        # Zoom in or out by a random factor in range [-30%, 30%]
        layers.RandomZoom(0.3),
    ]
)

plt.figure(figsize=(10, 10), dpi=100)
fig, axes = plt.subplots(2, 4, figsize=(10, 10))
for images, _ in ds_train.take(1):
    for i in range(8):
        if i >= 4:
            image_aug = data_augmentation1(images)
        else:
            image_aug = data_augmentation(images)
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(image_aug[0].numpy().astype(np.uint8))
        plt.axis("off")
# fig.tight_layout()
fig.show()

# %% Display the image
img = tf.keras.utils.load_img('../img/data\\test\\toon\\toon_cat_20.png', target_size=(600, 200))
plt.title("toon_cat_20")
plt.axis("off")
plt.imshow(img)