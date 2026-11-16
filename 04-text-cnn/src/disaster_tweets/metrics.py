"""Shared binary-classification metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(
    y_true: Iterable[int], probabilities: Iterable[float], threshold: float = 0.5
) -> dict[str, float]:
    """Calculate threshold-based metrics and ROC-AUC from positive-class probabilities."""
    y_true_array = np.asarray(list(y_true), dtype=np.int64)
    probabilities_array = np.asarray(list(probabilities), dtype=np.float64)
    if y_true_array.shape != probabilities_array.shape:
        raise ValueError("y_true and probabilities must have the same length.")
    if y_true_array.size == 0:
        raise ValueError("Metrics cannot be calculated for empty inputs.")

    predictions = (probabilities_array >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true_array, predictions)),
        "precision": float(precision_score(y_true_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_true_array, predictions, zero_division=0)),
        "f1": float(f1_score(y_true_array, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, probabilities_array)),
    }
