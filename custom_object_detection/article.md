# Custom Object Detection

This is for transfer learning object detection
Follow documentation https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/

## Custom training, custom model

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md

---
## Install TensorFlow Object Detection API

Download the the API, Model Garden, protobuf, pycocotools

---
## Test and verify installation
Test installation, download older protobuf version, add model garden to path, pip install a bunch of missing modules

```python
# From within TensorFlow/models/research/
python object_detection/builders/model_builder_tf2_test.py
```

## Create test/train directories

### Partition images and XMLs

Utilize existing `partition_dataset.py` script from TensorFlow documentation
80/20 train/test split

```bash
# from ~/blog/custom_object_detection/tensorflow/workspace/training_demo/images (main)
python ../../../scripts/partition_dataset.py -i ./unsorted/ -o ./ -r 0.2 -x
```

---
## Create label_map

TensorFlow requires a label map, which namely maps each of the used labels to an integer values. This label map is used both by the training and detection processes.

```
# custom_object_detection\tensorflow\workspace\training_demo\annotations\label_map.pbtxt

item {
    id: 1
    name: 'toon'
}

item {
    id: 2
    name: 'cog'
}

```

### Simplify existing labels

Replace `cog_SUIT_NAME` to `cog`, and `toon_ANIMAL` to `toon` in all .xml files

Used VSCode's search and replace, should create a script
Maybe unnecessary, can add a step when creating TensorFlow records

---
## Create TensorFlow records

Now that we have generated our annotations and split our dataset into the desired training and testing subsets, it is time to convert our annotations into the so called TFRecord format.

```bash
# from ~/blog/custom_object_detection/tensorflow/workspace/training_demo/images (main)
# Create train data
python ../../../scripts/generate_tfrecord.py -x ./train -l ../annotations/label_map.pbtxt -o ../annotations/train.record
# Create test data
python ../../../scripts/generate_tfrecord.py -x ./test -l ../annotations/label_map.pbtxt -o ../annotations/test.record
```

---
## Configure training job

To begin with, we need to download the latest pre-trained network for the model we wish to use.
Using TF pre-trained models, specifically `Faster R-CNN ResNet152 V1 1024x1024`
Not creating custom training job (yet)
Download the model and extract to `~\blog\custom_object_detection\tensorflow\workspace\training_demo\pre_trained_models\`

```
training_demo/
├─ ...
├─ pre-trained-models/
│  └─ faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8/
│     ├─ checkpoint/
│     ├─ saved_model/
│     └─ pipeline.config
└─ ...
```

### Configure the training **pipeline**

Create new directory under `training_demo\models`
Modify pipeline.config `blog\custom_object_detection\tensorflow\workspace\training_demo\models\my_ssd\pipeline.config`
Follow steps from here: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline

Alternatively, copy custom pipeline config from TF's object detection sample config directory: `~\blog\custom_object_detection\tensorflow\models\research\object_detection\samples\configs`

---
## Train the model

```bash
# from ~/blog/custom_object_detection/tensorflow/workspace/training_demo
python model_main_tf2.py --model_dir=models/my_ssd_cogs --pipeline_config_path=models/my_ssd_cogs/pipeline.config
```

No module name pycocotools, lvis,
Installed C++ 14.0:       error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

OSError: Checkpoint is expected to be an object-based checkpoint.
  fine_tune_checkpoint: "C:/Users/Sason/Documents/Projects/blog/custom_object_detection/tensorflow/workspace/training_demo/pre_trained_models/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8/checkpoint/ckpt-0.index"
Solution: Modify `fine_tune_checkpoint` in pipeline.config, strip the .index

### Monitor training progress with TensorBoard

<font style="color:red">TODO: Insert image of training</font>

---
## Evaluate the model

---
## Export the model

```bash
# from ~/blog/custom_object_detection/tensorflow/workspace/training_demo
python ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_cogs/pipeline.config --trained_checkpoint_dir ./models/my_ssd_cogs/ --output_directory ./exported-models/my_ssd_cogs
```

New file located under `~/blog/custom_object_detection/tensorflow/workspace/training_demo/exported-models/my_model`


---
## Use model for inference

The model does not generate bounding boxes on the images.
I suspect it's due to the vast difference in image sizes.
There are two avenues for resolving this issue:

1. Train a custom object detector that takes as input an image 1/4 the original size
2. Create a miniature dataset of image sizes relatively near the existing model's input size

Another option is to train using a different architecture, such as SSD or EfficientDet.
I will train another model first before I train a custom object detector.