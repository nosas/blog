<title>Performance Measures for Classification Problems</title>

# Classification Performance Measures

This article will explain the most common performance measures for classifications problems.
These measures apply to both binary and multi-class classification problems.

We will explain model performance metrics such as confusion matrix, accuracy, precision, recall, F1-score, and ROC curve.
The code in this article utilizes python3.7, tensorflow, and keras.

- [Classification Performance Measures](#classification-performance-measures)
    - [Why are performance measures important?](#why-are-performance-measures-important)
    - [Confusion Matrix](#confusion-matrix)
        - [TP, TN, FP, and FN](#tp-tn-fp-and-fn)
        - [Generate confusion matrix for TensorFlow model](#generate-confusion-matrix-for-tensorflow-model)
    - [Accuracy](#accuracy)
        - [Training accuracy in Keras](#training-accuracy-in-keras)
        - [Accuracy in Keras](#accuracy-in-keras)
    - [Precision](#precision)
        - [Training precision in Keras](#training-precision-in-keras)
    - [Accuracy vs Precision](#accuracy-vs-precision)
    - [Recall](#recall)
    - [F1-score](#f1-score)
    - [ROC Curve](#roc-curve)
    - [Conclusion](#conclusion)
    - [References](#references)

---
## Why are performance measures important?

During training, we monitor how well the model performs on the training data using the loss and accuracy metrics.
While these metrics are useful for monitoring the progress of the model, they are not very useful for evaluating the *performance*, or *quality*, of the model.

For example, imagine we've trained 100 models for the same classification problem, each with a different set of hyperparameters.
How do we know which model is the best?
Do we pick the model with the lowest loss or highest accuracy model?

We could pick the model with the lowest loss, or highest accuracy, but that does not guarantee that the model is the best.
Alternatively, we could pick the model with the least amount of wrong predictions on the test data.
But does that mean that the model is the best?

The loss and accuracy metrics give us a rough idea of the model's performance on the training data, but no indication of the model's general performance.
In order to gain a better understanding of the model's performance, we must use more specific metrics.
The metrics shown later in the article are designed to evaluate the true performance of our classification models.

---
## Confusion Matrix

A confusion matrix is a technique for visualizing a classification model's performance.
As the name suggests, a confusion matrix is a 2-dimensional table.
The confusion matrix is a core part of evaluating a classification model.

The table below shows the confusion matrix for a binary classification problem.
The rows represent the true labels and the columns represent the predicted labels.

<font style="color:red">TODO: Insert binary confusion matrix</font>

We can expand the confusion matrix to include multi-class classification problems.
For example, the table below shows the confusion matrix for a multi-class classification problem.

<font style="color:red">TODO: Insert multiclass confusion matrix</font>

From the confusion matrix, we can calculate the accuracy of the model - the number of correct predictions divided by the total number of predictions.
Furthermore, we can determine the number of true positives, true negatives, false positives, and false negatives.
We'll shorten the names to TP, TN, FP, and FN, respectively.
Using TP, TN, FP, and FN, we can calculate the precision, recall, and F1-score of the model.

### TP, TN, FP, and FN

<font style="color:red">TODO: Write about these</font>

Later in this article, we'll use a confusion matrix to derive the accuracy, precision, recall, and F1-score of our classification models.

### Generate confusion matrix for TensorFlow model

<font style="color:red">TODO: Insert code to generate confusion matrix for TensorFlow model</font>

---
## Accuracy

Accuracy is a metric that measures the percentage of correct predictions across all classes.
In other words, accuracy is **how close the model comes to the correct result**.

For example, imagine the goal is to shoot an arrow and hit the apple.
If we shoot and hit 10 arrows, we would be accurate or have high *accuracy*.
Now imagine a cluster of arrows around the apple - the arrows were *close* to hitting the apple, but had *no guarantee* of hitting the apple.
This remains a case of high accuracy, but with low *precision*.
We'll talk about precision in the next section.

Accuracy is calculated by dividing the number of correct predictions by the total number of predictions.
We calculate accuracy as follows: (TP + TN) / (TP + TN + FP + FN), where TP, TN, FP, and FN are the true positives, true negatives, false positives, and false negatives, respectively.

<font style="color:red">TODO: Insert code snippet to calculate accuracy</font>

### Training accuracy in Keras

During training, we can use Keras' built-in accuracy `Metric` classes: [binary_accuracy][binary_accuracy], [categorical_accuracy][categorical_accuracy], and [sparse_categorical_accuracy][sparse_categorical_accuracy].
For binary classification models, we use the `binary_accuracy` metric.
For multi-class classification models, we use the `[sparse_]categorical_accuracy` metric.

Utilizing the `Metric` classes allows us to use TensorBoard and visualize metrics during training.

<font style="color:red">TODO: Insert code snippet using Keras' built-in accuracy metrics</font>

<font style="color:red">TODO: Image of TensorBoard metrics</font>

### Accuracy in Keras

Alternatively, if we have a pre-trained model, we can use Keras' built-in accuracy methods to calculate the model's prediction accuracy: [binary][binary], [categorical][categorical], [sparse_categorical][sparse_categorical].

Where the `Metric` classes allow us to utilize TensorBoard, the accuracy methods allows us to directly calculate the model's prediction accuracy.

<font style="color:red">TODO: Insert code snippet using Keras' built-in accuracy methods</font>


---
## Precision

Precision is the ratio of correctly predicted *positive* labels to the total number of *positive* labels predicted.
It's calculated as follows: TP / (TP + FP).
In other words, precision is **how consistently the model reaches the correct result**.

<font style="color:red">TODO: Insert code snippet to calculate precision</font>

Imagine the goal is now to shoot an arrow at the apple's center.
We would have high *precision* if there were a cluster of arrows directly at the apple's center.
On the other hand, if we shot a cluster of arrows directly above the apple - the arrows were consistently above the apple, but had no guarantee of hitting the apple's center - we would still have high precision but low *accuracy*.

### Training precision in Keras

During training, we can use Keras' [built-in precision metric](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Precision).

---
## Accuracy vs Precision

Accuracy and precision are closely related.
The distinction between the two is crucial for engineers and scientists.

---
## Recall

---
## F1-score

---
## ROC Curve

---
## Conclusion

---
## References

<!-- Keras built-in training metrics -->
[binary_accuracy]: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy

[categorical_accuracy]: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy

[sparse_categorical_accuracy]: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy

<!-- Keras built-in accuracy methods-->
[binary]: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/binary_accuracy

[categorical]: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/categorical_accuracy

[sparse_categorical]: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/sparse_categorical_accuracy