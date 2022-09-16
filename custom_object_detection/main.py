# %%  Imports and function definitions
import numpy as np
import glob
import time
from IPython.display import display

from six import BytesIO

from PIL import Image

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from pascal_voc_writer import Writer

# %% Load model
tf.keras.backend.clear_session()
model = tf.saved_model.load(
    "./tensorflow/workspace/training_demo/exported-models/my_ssd_cogs/saved_model"
)

# %% Load variables
labelmap_path = "./tensorflow/workspace/training_demo/annotations/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(
    labelmap_path, use_display_name=True
)


# %% Functions
def load_image_into_numpy_array(path, resize: tuple[int, int] = None):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path (this can be local or on colossus)

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, "rb").read()
    image = Image.open(BytesIO(img_data))
    if resize:
        image = image.resize(resize)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop("num_detections"))
    output_dict = {
        key: value[0, :num_detections].numpy() for key, value in output_dict.items()
    }
    output_dict["num_detections"] = num_detections

    # detection_classes should be ints.
    output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)

    # Handle models with masks:
    if "detection_masks" in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict["detection_masks"],
            output_dict["detection_boxes"],
            image.shape[0],
            image.shape[1],
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return output_dict


# %% Create function to retrieve detected objects' boxes
def get_highest_scoring_boxes(output_dict: dict, score_threshold: float = 0.5):
    # Output scores are sorted by descending values
    scores = np.array(output_dict["detection_scores"])
    scores = scores[scores > score_threshold]
    num_detections = len(scores)

    if num_detections > 0:
        scores = output_dict["detection_scores"][:num_detections]
        boxes = output_dict["detection_boxes"][:num_detections]
        classes = output_dict["detection_classes"][:num_detections]
        classes_str = np.array(
            [category_index[class_id]["name"] for class_id in classes]
        )

    return scores, boxes, classes, classes_str


def normal_box_to_absolute(image_np: np.ndarray, box: list[float]):
    """Generate absolute box coordinates from relative box coordinates

    Args:
        image_np (np.ndarray): Numpy array containing an image, from `load_image_into_numpy_array`
        box (list[float]): List containing the relative [ymin, xmin, ymax, xmax] values
            Values must be in range [0, 1]

    Returns:
        list[float]: List containing absolute coordinate values
    """
    y, x, _ = image_np.shape
    if box.ndim > 1:  # Input is a list of bounding boxes
        boxes = []
        for b in box:
            ymin, xmin, ymax, xmax = b
            boxes.append(
                np.array([y * ymin, x * xmin, y * ymax, x * xmax]).astype("int")
            )
        return np.array(boxes)
    else:  # Input is a single bounding box
        ymin, xmin, ymax, xmax = box
        return np.array([y * ymin, x * xmin, y * ymax, x * xmax]).astype("int")


# %% Run inference
img_path_test = "./tensorflow/workspace/training_demo/images/label"

start = time.perf_counter()
img_fps = glob.glob(f"{img_path_test}/*.png")
# shuffle(img_fps)
for image_path in img_fps[-1:]:
    image_np = load_image_into_numpy_array(image_path, resize=(1720 * 2, 720 * 2))
    output_dict = run_inference_for_single_image(model, image_np)
    img = vis_util.visualize_boxes_and_labels_on_image_array(
        image=image_np,
        boxes=output_dict["detection_boxes"],
        classes=output_dict["detection_classes"],
        scores=output_dict["detection_scores"],
        category_index=category_index,
        instance_masks=output_dict.get("detection_masks_reframed", None),
        use_normalized_coordinates=True,
        line_thickness=10,
    )
    display(Image.fromarray(img))
    print(image_path)
end = time.perf_counter()
print(f"Took {round(end-start, 3)} seconds")


# %% Get highest scoring boxes
scores, boxes, classes, classes_str = get_highest_scoring_boxes(output_dict)

# %% Convert normalized box coordinates to absolute coordinates
image_np = load_image_into_numpy_array(image_path)
y, x, _ = image_np.shape
absolute_boxes = np.array([normal_box_to_absolute(image_np, box) for box in boxes])

# %% Draw bounding boxes using absolute coordinates
img = vis_util.visualize_boxes_and_labels_on_image_array(
    image=image_np,
    boxes=absolute_boxes,
    classes=classes,
    scores=scores,
    category_index=category_index,
    instance_masks=output_dict.get("detection_masks_reframed", None),
    use_normalized_coordinates=False,
    line_thickness=10,
)
display(Image.fromarray(img))

# %% Create XML files in Pascal-VOC format
# Create pascal VOC writer
writer = Writer(path=image_path, width=image_np.shape[1], height=image_np.shape[0])

# Add objects (class, xmin, ymin, xmax, ymax)
for class_str, box in zip(classes_str, boxes):
    ymin, xmin, ymax, xmax = normal_box_to_absolute(image_np, box)
    writer.addObject(class_str, xmin, ymin, xmax, ymax)

fn_xml = image_path.split("\\")[-1].replace(".png", ".xml")
path_xml = f"tensorflow/workspace/training_demo/images/label/{fn_xml}"
writer.save(annotation_path=path_xml)


# %% Create XML files for a list of test images
img_path_test = "./tensorflow/workspace/training_demo/images/label"

img_fps = glob.glob(f"{img_path_test}/*.png")
# shuffle(img_fps)
for image_path in img_fps:

    # Create downscaled image array to decrease inference times
    image_np_downscale = load_image_into_numpy_array(image_path, resize=(1720, 720))
    output_dict = run_inference_for_single_image(model, image_np_downscale)
    img = vis_util.visualize_boxes_and_labels_on_image_array(
        image=image_np_downscale,
        boxes=output_dict["detection_boxes"],
        classes=output_dict["detection_classes"],
        scores=output_dict["detection_scores"],
        category_index=category_index,
        instance_masks=output_dict.get("detection_masks_reframed", None),
        use_normalized_coordinates=True,
        line_thickness=10,
    )

    # Create original size image array for accurate absolute bounding boxes annotations
    image_np = load_image_into_numpy_array(image_path)
    scores, boxes, classes, classes_str = get_highest_scoring_boxes(output_dict)
    # Create pascal VOC writer
    writer = Writer(path=image_path, width=image_np.shape[1], height=image_np.shape[0])
    # Add objects (class, xmin, ymin, xmax, ymax)
    for class_str, box in zip(classes_str, boxes):
        ymin, xmin, ymax, xmax = normal_box_to_absolute(image_np, box)
        writer.addObject(class_str, xmin, ymin, xmax, ymax)

    fn_xml = image_path.split("\\")[-1].replace(".png", ".xml")
    path_xml = f"{img_path_test}/{fn_xml}"
    writer.save(annotation_path=path_xml)

    print(image_path)
# %%
