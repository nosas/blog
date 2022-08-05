<title>Performance Measures for Classification Problems</title>

# Classification Performance Measures

This article will explain the most common performance measures for classifications problems.
These measures apply to both binary and multi-class classification problems.

We will explain model performance metrics such as confusion matrix, accuracy, precision, recall, F1-score, and ROC curve.
The code in this article utilizes python3.7, tensorflow, and keras.

- [Classification Performance Measures](#classification-performance-measures)
    - [Why are performance measures important?](#why-are-performance-measures-important)
    - [Confusion Matrix](#confusion-matrix)
        - [Generate confusion matrix for TensorFlow model](#generate-confusion-matrix-for-tensorflow-model)
    - [Accuracy](#accuracy)
    - [Precision](#precision)
    - [Recall](#recall)
    - [F1-score](#f1-score)
    - [ROC Curve](#roc-curve)
    - [Conclusion](#conclusion)

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

Later in this article, we'll use a confusion matrix to derive the accuracy, precision, recall, and F1-score of our classification models.

### Generate confusion matrix for TensorFlow model

<font style="color:red">TODO: Insert code to generate confusion matrix for TensorFlow model</font>

---
## Accuracy

Accuracy is a metric that measures the percentage of correct predictions across all classes.
It's calculated by dividing the number of correct predictions by the total number of predictions.

We can calculate the accuracy as follows: (TP + TN) / (TP + TN + FP + FN), where TP, TN, FP, and FN are the true positives, true negatives, false positives, and false negatives, respectively.

<font style="color:red">TODO: Insert code snippet to calculate accuracy</font>

During training, we can use Keras' built-in accuracy metrics.
For binary classification models, we user the `binary_accuracy` metric.
For multi-class classification models, we use the `[sparse_]categorical_accuracy` metric.

<font style="color:red">TODO: Insert code snippet using Keras' built-tin accuracy metrics</font>

<!-- ? Given wrong_preds and target_labels, can we use the metric methods by themselves? -->

---
## Precision

---
## Recall

---
## F1-score

---
## ROC Curve

---
## Conclusion