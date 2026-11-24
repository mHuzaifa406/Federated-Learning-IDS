"""Synthetic unit tests for the Phase 1 CAVC calculation.

Run from project root:
    python scripts/test_cavc_synthetic.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from trfa.trust_metrics_cavc_v1 import calculate_class_aware_validation_contribution


NAMES = {0: "Benign", 2: "DoS", 4: "PortScan"}


def test_no_change_is_neutral():
    actual = np.array([0] * 10 + [2] * 10 + [4] * 10, dtype=np.int64)
    global_pred = actual.copy()
    candidate_pred = actual.copy()

    result = calculate_class_aware_validation_contribution(
        actual, global_pred, candidate_pred,
        min_class_support=5, class_names=NAMES,
    )

    assert abs(result["raw_macro_recall_delta"]) < 1e-12
    assert abs(result["cavc_score"] - 0.5) < 1e-12


def test_improvement_scores_above_neutral():
    actual = np.array([0] * 10 + [2] * 10 + [4] * 10, dtype=np.int64)
    global_pred = actual.copy()
    global_pred[[0, 10, 20]] = [2, 4, 0]  # one miss in each class
    candidate_pred = actual.copy()        # candidate corrects all three

    result = calculate_class_aware_validation_contribution(
        actual, global_pred, candidate_pred,
        min_class_support=5, class_names=NAMES,
    )

    assert result["raw_macro_recall_delta"] > 0.0
    assert result["cavc_score"] > 0.5


def test_damage_scores_below_neutral():
    actual = np.array([0] * 10 + [2] * 10 + [4] * 10, dtype=np.int64)
    global_pred = actual.copy()
    candidate_pred = actual.copy()
    candidate_pred[10:13] = 0  # degrade DoS recall by 0.30

    result = calculate_class_aware_validation_contribution(
        actual, global_pred, candidate_pred,
        min_class_support=5, class_names=NAMES,
    )

    assert result["raw_macro_recall_delta"] < 0.0
    assert result["cavc_score"] < 0.5
    assert result["mean_attack_to_target_rate_delta"] > 0.0


def test_equal_class_weighting_not_record_weighting():
    # Class 0 has 10 records; class 2 has 100 records. Losing 2/10 records
    # in class 0 changes its recall by -0.20; class 2 is unchanged.
    actual = np.array([0] * 10 + [2] * 100, dtype=np.int64)
    global_pred = actual.copy()
    candidate_pred = actual.copy()
    candidate_pred[:2] = 2

    result = calculate_class_aware_validation_contribution(
        actual, global_pred, candidate_pred,
        min_class_support=5,
    )

    assert abs(result["raw_macro_recall_delta"] - (-0.10)) < 1e-12


def test_low_support_classes_are_excluded_and_reported():
    actual = np.array([0] * 10 + [2] * 10 + [4] * 2, dtype=np.int64)
    predictions = actual.copy()

    result = calculate_class_aware_validation_contribution(
        actual, predictions, predictions,
        min_class_support=5,
    )

    assert result["eligible_class_ids"] == [0, 2]
    assert result["excluded_low_support_class_ids"] == [4]
    assert result["eligible_class_count"] == 2


def test_mismatched_array_lengths_raise():
    actual = np.array([0, 2, 4], dtype=np.int64)
    try:
        calculate_class_aware_validation_contribution(
            actual, actual, actual[:2], min_class_support=1
        )
    except ValueError:
        return
    raise AssertionError("Expected mismatched array lengths to raise ValueError.")


def main():
    tests = [
        test_no_change_is_neutral,
        test_improvement_scores_above_neutral,
        test_damage_scores_below_neutral,
        test_equal_class_weighting_not_record_weighting,
        test_low_support_classes_are_excluded_and_reported,
        test_mismatched_array_lengths_raise,
    ]

    print("CAVC SYNTHETIC UNIT TESTS")
    print("=" * 65)
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("-" * 65)
    print(f"All {len(tests)} synthetic CAVC tests passed.")


if __name__ == "__main__":
    main()
