"""Metrics used in the paper: accuracy, macro/weighted F1, per-class F1, AUC."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def evaluate(y_true, y_probs, class_names, title="Test set"):
    """Print a report and return all metrics as a dict (easy to save as JSON).

    Args:
        y_true:  true class indices, shape (N,)
        y_probs: predicted probabilities, shape (N, num_classes)
    """
    y_pred = np.argmax(y_probs, axis=1)
    y_true_onehot = label_binarize(y_true, classes=range(len(class_names)))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0).tolist(),
        "macro_auc": roc_auc_score(y_true_onehot, y_probs, average="macro", multi_class="ovr"),
        "weighted_auc": roc_auc_score(y_true_onehot, y_probs, average="weighted", multi_class="ovr"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0))
    print("Confusion matrix:\n", np.array(metrics["confusion_matrix"]))
    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} "
          f"| Macro AUC: {metrics['macro_auc']:.4f}")
    return metrics
