import numpy as np

from medsiglip.evaluation import binary_metrics, youden_threshold


def test_youden_threshold_and_binary_metrics_are_finite():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.80, 0.95])
    threshold = youden_threshold(labels, probabilities)
    metrics = binary_metrics(labels, probabilities, threshold)
    assert np.isfinite(threshold)
    assert metrics["roc_auc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
