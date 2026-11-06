"""Leakage-safe validation threshold selection and test evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        raise ValueError("Both positive and negative examples are required")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    scores = np.where(finite, true_positive_rate - false_positive_rate, -np.inf)
    return float(thresholds[int(np.argmax(scores))])


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def evaluate_classes(
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    results: dict[str, object] = {"classes": {}}
    aucs = []
    for index, name in enumerate(class_names):
        threshold = youden_threshold(validation_labels[:, index], validation_probabilities[:, index])
        metrics = binary_metrics(test_labels[:, index], test_probabilities[:, index], threshold)
        results["classes"][name] = metrics
        aucs.append(metrics["roc_auc"])
    results["macro_roc_auc"] = float(np.mean(aucs))
    return results

