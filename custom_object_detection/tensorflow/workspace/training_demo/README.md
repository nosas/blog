
Here’s an explanation for the workspace folder structure:

- annotations: This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our dataset images.

- exported-models: This folder will be used to store exported versions of our trained model(s).

- images: This folder contains a copy of all the images in our dataset, as well as the respective *.xml files produced for each one, once labelImg is used to annotate objects.

        images/train: This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model.

        images/test: This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model.

- models: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file *.config, as well as all files generated during the training and evaluation of our model.

- pre-trained-models: This folder will contain the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.

- README.md: This is an optional file which provides some general information regarding the training conditions of our model. It is not used by TensorFlow in any way, but it generally helps when you have a few training folders and/or you are revisiting a trained model after some time.
