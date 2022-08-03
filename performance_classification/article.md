<title>Performance Measures for Classification Problems</title>

# Classification Performance Measures

This article will explain the most common performance measures for classifications problems.
These measures apply to both binary and multi-class classification problems.

We will explain model performance metrics such as confusion matrix, accuracy, precision, recall, F1-score, and ROC curve.
The code blocks in this article will utilize python3.7, tensorflow, and keras.

- [Classification Performance Measures](#classification-performance-measures)
    - [Why are performance measures important?](#why-are-performance-measures-important)
    - [Confusion Matrix](#confusion-matrix)
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

---
## Accuracy

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