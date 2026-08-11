from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


PRIMARY_METRICS = ["At-Risk Recall", "PR-AUC"]
PLANNED_THRESHOLDS = [0.50, 0.30]
PLANNED_ERROR_CATEGORIES = ["False Negatives", "False Positives"]
PLANNED_SLICES = ["low engagement", "medium engagement", "high engagement"]
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    false_negatives: int
    false_positives: int
    illustrative_cost: int


def compute_binary_metrics(
    y_true,
    y_score,
    threshold: float = 0.50,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
    false_positive_cost: int = FALSE_POSITIVE_COST,
) -> BinaryMetrics:
    """Compute the Phase 2 binary metrics without changing the Phase 1 holdout policy.

    This helper is defined in Phase 1 so the metric contract is explicit and reproducible,
    but Phase 1 does not call it on the reserved test split. Held-out evaluation belongs
    to Phase 2.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    y_pred = (y_score_arr >= threshold).astype(int)
    fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))

    return BinaryMetrics(
        threshold=float(threshold),
        precision=float(precision_score(y_true_arr, y_pred, zero_division=0)),
        recall=float(recall_score(y_true_arr, y_pred, zero_division=0)),
        f1=float(f1_score(y_true_arr, y_pred, zero_division=0)),
        pr_auc=float(average_precision_score(y_true_arr, y_score_arr)),
        false_negatives=fn,
        false_positives=fp,
        illustrative_cost=int(false_negative_cost * fn + false_positive_cost * fp),
    )


def phase1_evaluation_plan() -> dict:
    """Return the evaluation/error-analysis plan declared in the Phase 1 report."""
    return {
        "primary_metrics": PRIMARY_METRICS.copy(),
        "thresholds": PLANNED_THRESHOLDS.copy(),
        "error_categories": PLANNED_ERROR_CATEGORIES.copy(),
        "engagement_slices": PLANNED_SLICES.copy(),
        "illustrative_cost": {
            "false_negative": FALSE_NEGATIVE_COST,
            "false_positive": FALSE_POSITIVE_COST,
        },
    }
