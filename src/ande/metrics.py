"""Evaluation metrics matching paper Section V-B."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class Metrics:
    accuracy: float
    precision: float
    f1: float
    recall: float
    fpr: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def multiclass_fpr(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Average one-vs-rest false positive rate (paper Eq. 6).

    For class c: FP_c = #(pred=c, true!=c); TN_c = #(pred!=c, true!=c).
    FPR is averaged across classes (macro).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fpr_per_class: list[float] = []
    total = cm.sum()
    for c in range(num_classes):
        fp = cm[:, c].sum() - cm[c, c]
        tn = total - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        denom = fp + tn
        fpr_per_class.append(fp / denom if denom else 0.0)
    return float(np.mean(fpr_per_class))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Metrics:
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        fpr=multiclass_fpr(y_true, y_pred, num_classes),
    )
