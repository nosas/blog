<title>Performance Measures for Classification Problems</title>

# Classification Performance Measures

This article will explain the most common performance measures for classifications problems.
These measures apply to both binary and multi-class classification problems.

We will explain model performance metrics such as confusion matrix, accuracy, precision, recall, F1-score, and ROC curve.
The code in this article utilizes python3.7, tensorflow, and keras.

- [Classification Performance Measures](#classification-performance-measures)
    - [Why are performance measures important?](#why-are-performance-measures-important)
    - [Confusion Matrix](#confusion-matrix)
        - [Generate confusion matrix](#generate-confusion-matrix)
    - [Accuracy](#accuracy)
        - [Training accuracy in Keras](#training-accuracy-in-keras)
        - [Accuracy in Keras](#accuracy-in-keras)
    - [Precision](#precision)
        - [Training precision in Keras](#training-precision-in-keras)
    - [Accuracy vs Precision](#accuracy-vs-precision)
    - [Recall](#recall)
    - [When to use Accuracy, Precision, or Recall](#when-to-use-accuracy-precision-or-recall)
    - [F1-score](#f1-score)
    - [ROC Curve](#roc-curve)
    - [Conclusion](#conclusion)
    - [References](#references)

---
## Why are performance measures important?

During training, we monitor how well the model performs on the training data using model loss and accuracy metrics.
While these metrics are useful for monitoring the model's training progress, they are not very useful for evaluating the *performance*, or *quality*, of the model.

For example, imagine we've trained 100 models for the same classification problem, each with a different set of hyperparameters.
How do we know which model is the best?
Do we pick the model with the lowest loss, highest accuracy, or maybe a combination of the two metrics?

We could pick the model with the lowest loss, or highest accuracy, but that does not guarantee that the model is the best.
Alternatively, we could pick the model with the least amount of wrong predictions on the test data.
But does that guarantee we've picked the best model?

The loss and accuracy metrics give us a rough idea of the model's performance on the training data, but no indication of the model's general performance.
In order to gain a better understanding of the model's performance, we must use more specific metrics.
The metrics shown in this article are designed to evaluate the true performance of our classification models.

---
## Confusion Matrix

A confusion matrix is a core computer vision technique for visualizing and evaluating a classification model's performance.
As the name suggests, a confusion matrix is a 2-dimensional table.

From the confusion matrix, we can determine the number of true positives, true negatives, false positives, and false negatives.
We'll shorten the names to TP, TN, FP, and FN, respectively.
Using TP, TN, FP, and FN, we can calculate the model's accuracy, precision, recall, and F1-score.

The table below shows the confusion matrix for a binary classification problem.
The rows represent the true labels and the columns represent the predicted labels.
The diagonal represents correct predictions and all other cells represent incorrect predictions.
Ideally, our confusion matrix should diagonal contain values - no incorrect predictions.

<figure class="center">
    <img src="img/confusion_matrix.png" style="width:100%;"/>
    <figcaption>Confusion matrix for a binary classification problem</figcaption>
</figure>

We can expand the confusion matrix to include multi-class classification problems.
For instance, the table below shows the confusion matrix for a multi-class classification problem with four classes.

The more classes in a multi-class classification problem, the more convoluted the confusion matrix will be.
This should not stop us from using the confusion matrix to evaluate model performance, however.

<figure class="center">
    <img src="img/confusion_matrix_multiclass.png" style="width:100%;"/>
    <figcaption>Confusion matrix for a multiclass classification problem</figcaption>
</figure>

Later in this article, we'll use a confusion matrix to derive the accuracy, precision, recall, and F1-score of our classification models.

### Generate confusion matrix

Given a list of predictions and a list of targets (true labels), we can generate a confusion matrix.
We'll utilize two libraries to display the matrix: `matplotlib` and `sklearn`.

The code block below was used to generate the multiclass confusion matrix in the earlier section.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    predictions: list[str],
    targets: list[str],
    display_labels: list[str],
    title: str = "",
) -> None:
    """Plot the confusion matrix for a list of predictions and targets"""

    # Generate the confusion matrix
    matrix = confusion_matrix(
        y_pred=predictions, y_true=targets, labels=display_labels
        )
    # Generate the figure
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=display_labels
    )
    # Plot the figure, add title, and resize
    display.plot(include_values=True)
    display.ax_.set_title(title)
    fig = display.ax_.get_figure()
    fig.set_figwidth(8)
    fig.set_figheight(8)
    # Show the confusion matrix
    plt.show()
```

The `confusion_matrix()` method from `sklearn.metrics` takes in a list of predictions and a list of targets (true labels).
It outputs a 2-dimensional numpy array that represents the confusion matrix.

```python
>>> confusion_matrix(
        y_pred=preds_str,
        y_true=integer_to_suit(test_labels),
        labels=['bb', 'lb', 'cb', 'sb']
    )
array([[36,  0,  1,  4],
       [ 0, 60,  0,  0],
       [ 4,  2, 35,  0],
       [10,  0,  0, 45]], dtype=int64)
```

It may be important to note that the ordering of the `labels` parameter determines the order of the rows and columns in the confusion matrix.

---
## Accuracy

<figure class="right" style="width:30%;">
    <img src="img/confusion_matrix_accuracy.png" style="width:100%;"/>
    <figcaption>Accuracy is the confusion matrix's diagonal</figcaption>
</figure>

Accuracy is a metric that measures the percentage of correct predictions across all classes.
In other words, accuracy is **how close the model comes to the correct result**.

For example, imagine the goal is to shoot an arrow and hit the apple.
If we shoot and hit 10 arrows, we would be accurate or have high *accuracy*.
Now imagine a cluster of arrows around the apple - the arrows were *close* to hitting the apple, but had *no guarantee* of hitting the apple.
Because the arrows land close to the target this remains a case of high accuracy, but with low *precision*.
We'll talk about precision in the next section.

Using the initial confusion matrix, we can visualize accuracy as the confusion matrix's diagonal.

Accuracy is calculated by dividing the number of correct predictions by the total number of predictions.
We calculate accuracy as follows: `(TP + TN) / (TP + TN + FP + FN)`.
`(TP + TN)` is the number of correct predictions - the diagonal of the confusion matrix.
`(TP + TN + FP + FN)` is the total number of predictions - the sum of all cells in the confusion matrix.

```python
predictions = model.predict(test_images)
# Get the wrong predictions as a True/False array, where True == wrong prediction
wrong_preds = preds != test_labels
# Count the number of wrong predictions (number of True values in the array)
num_wrong_preds = len(np.argwhere(wrong_preds))
# Calculate the accuracy
accuracy = (len(preds) - num_wrong_preds) / len(preds)
```

### Training accuracy in Keras

During training, we can use Keras' built-in accuracy `Metric` classes: [binary_accuracy][binary_accuracy], [categorical_accuracy][categorical_accuracy], and [sparse_categorical_accuracy][sparse_categorical_accuracy].
For binary classification models, we use the `binary_accuracy` metric.
For multi-class classification models, we use the `[sparse_]categorical_accuracy` metric.

Utilizing the `Metric` classes allows us to use TensorBoard and visualize metrics during training.

<font style="color:red">TODO: Insert code snippet using Keras' built-in accuracy metrics</font>

<font style="color:red">TODO: Image of TensorBoard metrics</font>

### Accuracy in Keras

Alternatively, given a pre-trained model, we can use Keras' built-in accuracy methods to calculate the model's prediction accuracy: [binary][binary], [categorical][categorical], [sparse_categorical][sparse_categorical].

Where the `Metric` classes allow us to utilize TensorBoard, the accuracy methods allows us to directly calculate the model's prediction accuracy.

<font style="color:red">TODO: Insert code snippet using Keras' built-in accuracy methods</font>

---
## Precision

Precision is the ratio of correctly predicted *positive* labels to the total number of *positive* labels predicted.
It's calculated as follows: TP / (TP + FP).
In other words, precision is **how consistently the model reaches the correct result**.

<font style="color:red">TODO: Insert code snippet to calculate precision</font>

Imagine the goal is now to shoot an arrow at the apple's center.
Note how the previous goal was more vague - just to hit the apple.
Now we're aiming for a more specific goal - to hit the apple's center.

We would have high *precision* if there were a cluster of arrows directly at the apple's center.
On the other hand, if we shot a cluster of arrows directly above the apple - the arrows were consistently above the apple, but had no guarantee of hitting the apple's center - we would still have high precision due to the consistency, but low precision relative to our desired goal.

### Training precision in Keras

During training, we can use Keras' [built-in precision metric](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Precision).

<font style="color:red">TODO: Insert code snippet using Keras' built-in precision metric</font>

---
## Accuracy vs Precision

Accuracy and precision are closely related.
The two metrics are often used interchangeably in day-to-day work.
However, the distinction between accuracy and precision is crucial for engineers and scientists.

Earlier, we explained how accuracy is how *close* the model is to the correct result whereas precision is how *consistently* the model reaches the correct result.
We can imagine accuracy as how close the arrows land near the apple, and precision as how consistently the arrows land near one another.

One can have high accuracy and low precision - such as when the arrows land everywhere around the apple.
High precision and low accuracy is also possible - such as when the arrows consistently cluster at some point except for the apple.


---
## Recall

---
## When to use Accuracy, Precision, or Recall

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