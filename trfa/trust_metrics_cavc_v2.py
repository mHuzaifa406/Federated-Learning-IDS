"""Class-Aware Validation Contribution (CAVC) V2.

CAVC V2 reports three separate candidate-model contribution measurements on
one unchanged, labelled server-validation set:

1. Attack-detection contribution (higher is better): macro recall change
   over sufficiently supported *attack* classes, excluding the target class
   (Benign by default).
2. Targeted attack-to-target damage (higher is worse): per-attack-class
   increase in the rate at which true attack records are predicted as the
   target label. Both signed mean change and positive-only penalty are
   returned.
3. Benign false-positive impact (higher is worse): change in the rate at
   which true target-class records (Benign by default) are predicted as any
   non-target class.

These are diagnostics, not a probability of honesty and not a final trust
score. They are intentionally not fused into one scalar in V2.

The function uses identical actual labels and prediction positions for the
global and candidate models. Low-support classes are reported but excluded
from the aggregated attack metrics. A caller must not use the final test set
for this calculation or for threshold/model selection.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence
import math

import numpy as np

CAVC_VERSION = "2.0"


def _as_integer_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric class IDs.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    if not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain integer class IDs.")
    return array.astype(np.int64, copy=False)


def _class_metrics(actual: np.ndarray, predicted: np.ndarray, class_id: int) -> dict:
    actual_positive = actual == class_id
    predicted_positive = predicted == class_id
    support = int(np.count_nonzero(actual_positive))
    tp = int(np.count_nonzero(actual_positive & predicted_positive))
    fp = int(np.count_nonzero(~actual_positive & predicted_positive))
    fn = int(np.count_nonzero(actual_positive & ~predicted_positive))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / support if support else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "support": support,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def calculate_cavc_v2(
    actual_labels,
    global_predictions,
    candidate_predictions,
    *,
    min_class_support: int = 100,
    target_label: int = 0,
    class_names: Optional[Mapping[int, str]] = None,
) -> dict:
    """Compare a candidate model against the current global model.

    Parameters
    ----------
    actual_labels:
        Ground-truth labels for the clean server-validation records.
    global_predictions:
        Global model predictions for those records.
    candidate_predictions:
        Candidate client's predictions on the exact same records, in the
        exact same order.
    min_class_support:
        Minimum true-label support required for an attack class to enter
        attack macro-recall and attack-to-target aggregates. The target class
        is not included in the attack aggregates.
    target_label:
        Class used as the target/misdirection label (0 = Benign here).
    class_names:
        Optional mapping from class IDs to readable names.

    Returns
    -------
    dict
        Separate attack-recall, targeted-damage, and benign-FPR components,
        plus per-class diagnostic rows and class-support information.
    """
    actual = _as_integer_vector(actual_labels, "actual_labels")
    global_pred = _as_integer_vector(global_predictions, "global_predictions")
    candidate_pred = _as_integer_vector(candidate_predictions, "candidate_predictions")

    if not (actual.size == global_pred.size == candidate_pred.size):
        raise ValueError("Actual labels and both prediction arrays must have equal lengths.")
    if isinstance(min_class_support, bool) or not isinstance(min_class_support, (int, np.integer)):
        raise TypeError("min_class_support must be an integer.")
    if int(min_class_support) < 1:
        raise ValueError("min_class_support must be at least 1.")
    if isinstance(target_label, bool) or not isinstance(target_label, (int, np.integer)):
        raise TypeError("target_label must be an integer class ID.")

    min_class_support = int(min_class_support)
    target_label = int(target_label)
    names = class_names or {}

    class_ids, supports = np.unique(actual, return_counts=True)
    support_by_class = {int(k): int(v) for k, v in zip(class_ids, supports)}
    if target_label not in support_by_class:
        raise ValueError(f"target_label={target_label} is absent from actual_labels.")

    all_eligible = sorted(k for k, n in support_by_class.items() if n >= min_class_support)
    excluded_low_support = sorted(k for k, n in support_by_class.items() if n < min_class_support)
    eligible_attack = [k for k in all_eligible if k != target_label]
    excluded_attack = [k for k in support_by_class if k != target_label and k not in eligible_attack]

    if not eligible_attack:
        raise ValueError(
            "No attack class meets min_class_support="
            f"{min_class_support}. Lower the threshold or improve validation coverage."
        )

    per_class = []
    attack_recall_deltas = []
    attack_f1_deltas = []
    attack_to_target_deltas = []

    for class_id in sorted(support_by_class):
        gm = _class_metrics(actual, global_pred, class_id)
        cm = _class_metrics(actual, candidate_pred, class_id)
        is_target = class_id == target_label
        eligible_for_attack_aggregate = class_id in eligible_attack

        if not is_target:
            class_mask = actual == class_id
            global_to_target = float(np.mean(global_pred[class_mask] == target_label))
            candidate_to_target = float(np.mean(candidate_pred[class_mask] == target_label))
            to_target_delta = candidate_to_target - global_to_target
        else:
            global_to_target = None
            candidate_to_target = None
            to_target_delta = None

        row = {
            "class_id": int(class_id),
            "class_name": str(names.get(int(class_id), int(class_id))),
            "support": int(support_by_class[class_id]),
            "is_target_class": bool(is_target),
            "eligible_for_attack_aggregate": bool(eligible_for_attack_aggregate),
            "global_precision": gm["precision"],
            "candidate_precision": cm["precision"],
            "delta_precision": cm["precision"] - gm["precision"],
            "global_recall": gm["recall"],
            "candidate_recall": cm["recall"],
            "delta_recall": cm["recall"] - gm["recall"],
            "global_f1": gm["f1"],
            "candidate_f1": cm["f1"],
            "delta_f1": cm["f1"] - gm["f1"],
            "global_attack_to_target_rate": global_to_target,
            "candidate_attack_to_target_rate": candidate_to_target,
            "delta_attack_to_target_rate": to_target_delta,
        }
        per_class.append(row)

        if eligible_for_attack_aggregate:
            attack_recall_deltas.append(cm["recall"] - gm["recall"])
            attack_f1_deltas.append(cm["f1"] - gm["f1"])
            attack_to_target_deltas.append(float(to_target_delta))

    # Component 1: attack-only macro-recall contribution. Benign recall is
    # intentionally excluded so improved Benign recognition cannot mask
    # deteriorating attack detection.
    attack_recall_delta = float(np.mean(attack_recall_deltas))
    attack_detection_score = float(np.clip(0.5 + 0.5 * attack_recall_delta, 0.0, 1.0))
    attack_macro_f1_delta = float(np.mean(attack_f1_deltas))

    # Component 2: report both signed average change and positive-only damage.
    # Positive values mean more true attacks are being predicted as target.
    attack_to_target_delta_mean = float(np.mean(attack_to_target_deltas))
    targeted_damage_penalty = float(np.mean([max(0.0, x) for x in attack_to_target_deltas]))

    # Component 3: target/Benign false-positive rate is the fraction of true
    # target examples predicted as anything other than target.
    target_mask = actual == target_label
    global_target_fpr = float(np.mean(global_pred[target_mask] != target_label))
    candidate_target_fpr = float(np.mean(candidate_pred[target_mask] != target_label))
    target_fpr_delta = candidate_target_fpr - global_target_fpr
    benign_fpr_damage_penalty = float(max(0.0, target_fpr_delta))

    target_global = _class_metrics(actual, global_pred, target_label)
    target_candidate = _class_metrics(actual, candidate_pred, target_label)

    return {
        "cavc_version": CAVC_VERSION,
        "min_class_support": min_class_support,
        "target_label": target_label,
        "validation_records": int(actual.size),
        "class_support": {str(k): int(v) for k, v in sorted(support_by_class.items())},
        "eligible_class_ids": all_eligible,
        "eligible_attack_class_ids": eligible_attack,
        "excluded_low_support_class_ids": excluded_low_support,
        "excluded_low_support_attack_class_ids": excluded_attack,
        "eligible_attack_class_count": len(eligible_attack),
        # Component 1: higher is better; 0.5 is no mean attack-recall change.
        "attack_macro_recall_global": float(np.mean([
            _class_metrics(actual, global_pred, k)["recall"] for k in eligible_attack
        ])),
        "attack_macro_recall_candidate": float(np.mean([
            _class_metrics(actual, candidate_pred, k)["recall"] for k in eligible_attack
        ])),
        "attack_recall_contribution": attack_recall_delta,
        "attack_detection_contribution_score": attack_detection_score,
        "attack_macro_f1_delta": attack_macro_f1_delta,
        # Component 2: higher is worse; positive means more attack->target errors.
        "attack_to_target_rate_delta_mean": attack_to_target_delta_mean,
        "targeted_damage_penalty": targeted_damage_penalty,
        # Component 3: higher is worse; positive means more benign records flagged.
        "global_benign_false_positive_rate": global_target_fpr,
        "candidate_benign_false_positive_rate": candidate_target_fpr,
        "benign_false_positive_rate_delta": float(target_fpr_delta),
        "benign_fpr_damage_penalty": benign_fpr_damage_penalty,
        "global_benign_recall": target_global["recall"],
        "candidate_benign_recall": target_candidate["recall"],
        "benign_recall_delta": target_candidate["recall"] - target_global["recall"],
        "per_class": per_class,
        "interpretation": (
            "Separate contribution diagnostics only. This result is not a calibrated "
            "probability of honesty and must not be used alone for client rejection."
        ),
    }


def rank_clients_within_condition(rows: Sequence[Mapping], *, score_key: str = "attack_detection_contribution_score") -> list[dict]:
    """Rank one row per client within one condition; rank 1 is the lowest score.

    This helper prevents the V1 reporting error where each client's scores
    were ranked only across its own rounds (which forces an average rank of
    (rounds+1)/2 for every client).
    """
    cleaned = [dict(row) for row in rows]
    if not cleaned:
        return []
    if any(score_key not in row for row in cleaned):
        raise KeyError(f"Every row must contain {score_key!r}.")
    client_ids = [row.get("client_id", row.get("Client")) for row in cleaned]
    if any(client_id is None for client_id in client_ids):
        raise ValueError("Every row must contain client_id or Client.")
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("Expected exactly one row per client for ranking.")
    scores = np.asarray([float(row[score_key]) for row in cleaned], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Scores used for ranking must be finite.")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    # Average ranks for ties, 1-based.
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = average_rank
        i = j
    for row, rank in zip(cleaned, ranks):
        row["rank_lowest_first"] = float(rank)
    return cleaned
