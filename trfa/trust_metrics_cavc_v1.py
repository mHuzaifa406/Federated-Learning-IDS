"""Class-Aware Validation Contribution (CAVC), Phase 1 prototype.

CAVC compares a candidate client's local model against the current global
model on the SAME clean server-validation examples. It is a contribution
signal for development experiments, not proof that a client is honest.

For each class k whose validation support is at least `min_class_support`,
we calculate recall for the current global model and the candidate model.
The raw contribution is the unweighted mean class-recall change:

    Q_i = mean_k(Recall(candidate, k) - Recall(global, k))

The bounded display score is:

    CAVC_i = 0.5 + 0.5 * Q_i

Thus 0.5 means no average change, >0.5 means a positive macro-recall
contribution, and <0.5 means a negative macro-recall contribution.

Additional diagnostics include macro-F1 change and the change in the rate at
which non-target attack records are predicted as the target label (Benign by
default). These diagnostics are reported separately; they do not silently
alter the CAVC formula.

Research cautions:
- Class support filtering is essential. A recall estimate based on only a
  handful of examples is unstable.
- Equal class weighting prevents the largest class from dominating the score,
  but does not eliminate validation/client distribution mismatch.
- Repeatedly using the same server-validation set to screen clients can cause
  overfitting to that validation set. Keep an independent final test set.
- A malicious update can improve this validation score or exploit what it
  measures. CAVC must be evaluated alongside other signals and attacks.
"""

from __future__ import annotations

from typing import Mapping, Optional
import math

import numpy as np


CAVC_VERSION = "1.0-prototype"


def _as_label_vector(values, name: str) -> np.ndarray:
    """Validate labels/predictions as a non-empty 1-D integer vector."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric class IDs.")
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{name} must contain finite integer class IDs.")
    return array.astype(np.int64, copy=False)


def _safe_precision_recall_f1(actual: np.ndarray, predicted: np.ndarray, class_id: int):
    """Return one-vs-rest precision, recall and F1 for a class."""
    actual_positive = actual == class_id
    predicted_positive = predicted == class_id

    tp = int(np.count_nonzero(actual_positive & predicted_positive))
    fp = int(np.count_nonzero(~actual_positive & predicted_positive))
    fn = int(np.count_nonzero(actual_positive & ~predicted_positive))
    support = int(np.count_nonzero(actual_positive))

    precision_denominator = tp + fp
    recall_denominator = tp + fn

    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = (2.0 * precision * recall / f1_denominator) if f1_denominator else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": support,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calculate_class_aware_validation_contribution(
    actual_labels,
    global_predictions,
    candidate_predictions,
    *,
    min_class_support: int = 100,
    target_label: int = 0,
    class_names: Optional[Mapping[int, str]] = None,
) -> dict:
    """Calculate the candidate model's class-aware contribution vs global.

    Parameters
    ----------
    actual_labels:
        Ground-truth labels for the clean server-validation examples.
    global_predictions:
        Predictions made by the current global model on those examples.
    candidate_predictions:
        Predictions made by one client's locally trained candidate model on
        exactly the same examples, in exactly the same order.
    min_class_support:
        Minimum number of true examples required for a class to enter the
        macro-recall contribution calculation. Classes below this threshold
        are listed as excluded, not silently assigned zero recall.
    target_label:
        Label that attack records may be incorrectly mapped to. Defaults to
        label 0 (Benign) for the current label-flipping experiments.
    class_names:
        Optional mapping from integer class IDs to readable class names.

    Returns
    -------
    dict with the scalar CAVC score, raw deltas, support information and
    per-class diagnostics. The score is descriptive and must not be treated
    as a calibrated probability of client honesty.
    """
    actual = _as_label_vector(actual_labels, "actual_labels")
    global_pred = _as_label_vector(global_predictions, "global_predictions")
    candidate_pred = _as_label_vector(candidate_predictions, "candidate_predictions")

    if not (actual.size == global_pred.size == candidate_pred.size):
        raise ValueError("Actual labels and both prediction arrays must have equal lengths.")
    if isinstance(min_class_support, bool) or not isinstance(min_class_support, int):
        raise TypeError("min_class_support must be an integer.")
    if min_class_support < 1:
        raise ValueError("min_class_support must be at least 1.")
    if isinstance(target_label, bool) or not isinstance(target_label, (int, np.integer)):
        raise TypeError("target_label must be an integer class ID.")

    class_ids, supports = np.unique(actual, return_counts=True)
    support_by_class = {int(k): int(v) for k, v in zip(class_ids, supports)}
    eligible = [k for k in support_by_class if support_by_class[k] >= min_class_support]
    excluded = [k for k in support_by_class if support_by_class[k] < min_class_support]

    if not eligible:
        raise ValueError(
            "No validation class meets min_class_support="
            f"{min_class_support}. Lower the support threshold or improve validation coverage."
        )

    names = class_names or {}
    per_class = []
    recall_deltas = []
    f1_deltas = []
    precision_deltas = []
    attack_to_target_deltas = []

    for class_id in sorted(eligible):
        global_metrics = _safe_precision_recall_f1(actual, global_pred, class_id)
        candidate_metrics = _safe_precision_recall_f1(actual, candidate_pred, class_id)

        recall_delta = candidate_metrics["recall"] - global_metrics["recall"]
        f1_delta = candidate_metrics["f1"] - global_metrics["f1"]
        precision_delta = candidate_metrics["precision"] - global_metrics["precision"]

        row = {
            "class_id": int(class_id),
            "class_name": str(names.get(int(class_id), int(class_id))),
            "support": int(support_by_class[class_id]),
            "global_precision": global_metrics["precision"],
            "candidate_precision": candidate_metrics["precision"],
            "delta_precision": float(precision_delta),
            "global_recall": global_metrics["recall"],
            "candidate_recall": candidate_metrics["recall"],
            "delta_recall": float(recall_delta),
            "global_f1": global_metrics["f1"],
            "candidate_f1": candidate_metrics["f1"],
            "delta_f1": float(f1_delta),
        }

        # For non-target true classes, measure how often their records are
        # incorrectly classified as the target label (Benign in current runs).
        if class_id != int(target_label):
            class_mask = actual == class_id
            global_to_target = float(np.mean(global_pred[class_mask] == target_label))
            candidate_to_target = float(np.mean(candidate_pred[class_mask] == target_label))
            target_rate_delta = candidate_to_target - global_to_target
            row.update({
                "global_attack_to_target_rate": global_to_target,
                "candidate_attack_to_target_rate": candidate_to_target,
                "delta_attack_to_target_rate": float(target_rate_delta),
            })
            attack_to_target_deltas.append(target_rate_delta)
        else:
            row.update({
                "global_attack_to_target_rate": None,
                "candidate_attack_to_target_rate": None,
                "delta_attack_to_target_rate": None,
            })

        per_class.append(row)
        recall_deltas.append(recall_delta)
        f1_deltas.append(f1_delta)
        precision_deltas.append(precision_delta)

    raw_macro_recall_delta = float(np.mean(recall_deltas))
    cavc_score = float(np.clip(0.5 + 0.5 * raw_macro_recall_delta, 0.0, 1.0))

    return {
        "cavc_version": CAVC_VERSION,
        "min_class_support": int(min_class_support),
        "target_label": int(target_label),
        "validation_records": int(actual.size),
        "eligible_class_ids": [int(x) for x in sorted(eligible)],
        "excluded_low_support_class_ids": [int(x) for x in sorted(excluded)],
        "eligible_class_count": int(len(eligible)),
        "excluded_low_support_class_count": int(len(excluded)),
        "global_macro_recall": float(np.mean([
            _safe_precision_recall_f1(actual, global_pred, k)["recall"] for k in eligible
        ])),
        "candidate_macro_recall": float(np.mean([
            _safe_precision_recall_f1(actual, candidate_pred, k)["recall"] for k in eligible
        ])),
        "raw_macro_recall_delta": raw_macro_recall_delta,
        "cavc_score": cavc_score,
        "global_macro_f1": float(np.mean([
            _safe_precision_recall_f1(actual, global_pred, k)["f1"] for k in eligible
        ])),
        "candidate_macro_f1": float(np.mean([
            _safe_precision_recall_f1(actual, candidate_pred, k)["f1"] for k in eligible
        ])),
        "macro_f1_delta": float(np.mean(f1_deltas)),
        "macro_precision_delta": float(np.mean(precision_deltas)),
        "mean_attack_to_target_rate_delta": (
            float(np.mean(attack_to_target_deltas)) if attack_to_target_deltas else None
        ),
        "per_class": per_class,
        "class_support": {str(k): v for k, v in sorted(support_by_class.items())},
    }
