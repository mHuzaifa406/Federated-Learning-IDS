"""Synthetic unit tests for CAVC V2.

Run from project root:
    python scripts/test_cavc_v2.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from trfa.trust_metrics_cavc_v2 import calculate_cavc_v2, rank_clients_within_condition


def _base_data():
    # 20 benign, 20 DoS, 20 PortScan, 20 Brute Force.
    actual = np.array([0] * 20 + [2] * 20 + [4] * 20 + [6] * 20, dtype=np.int64)
    perfect = actual.copy()
    return actual, perfect


def test_no_change_neutral_attack_contribution():
    actual, pred = _base_data()
    result = calculate_cavc_v2(actual, pred, pred, min_class_support=10)
    assert abs(result["attack_recall_contribution"]) < 1e-12
    assert abs(result["attack_detection_contribution_score"] - 0.5) < 1e-12
    assert abs(result["targeted_damage_penalty"]) < 1e-12
    assert abs(result["benign_fpr_damage_penalty"]) < 1e-12


def test_attack_recall_improvement():
    actual, global_pred = _base_data()
    global_pred[20:24] = 0  # Four DoS records predicted as Benign.
    candidate = actual.copy()
    result = calculate_cavc_v2(actual, global_pred, candidate, min_class_support=10)
    assert result["attack_recall_contribution"] > 0
    assert result["attack_detection_contribution_score"] > 0.5
    assert result["targeted_damage_penalty"] == 0.0


def test_targeted_attack_to_benign_damage_is_separate():
    actual, global_pred = _base_data()
    candidate = actual.copy()
    candidate[20:26] = 0  # 6 DoS examples now classified as Benign.
    result = calculate_cavc_v2(actual, global_pred, candidate, min_class_support=10)
    assert result["attack_recall_contribution"] < 0
    assert result["targeted_damage_penalty"] > 0
    assert result["attack_to_target_rate_delta_mean"] > 0


def test_benign_false_positive_penalty():
    actual, global_pred = _base_data()
    candidate = actual.copy()
    candidate[:5] = 2  # Benign examples now flagged as DoS.
    result = calculate_cavc_v2(actual, global_pred, candidate, min_class_support=10)
    assert result["benign_false_positive_rate_delta"] > 0
    assert result["benign_fpr_damage_penalty"] > 0
    # Benign errors do not enter the attack-only recall component.
    assert abs(result["attack_recall_contribution"]) < 1e-12


def test_minimum_support_excludes_rare_attacks():
    actual = np.array([0] * 20 + [2] * 20 + [4] * 4, dtype=np.int64)
    pred = actual.copy()
    result = calculate_cavc_v2(actual, pred, pred, min_class_support=10)
    assert result["eligible_attack_class_ids"] == [2]
    assert result["excluded_low_support_attack_class_ids"] == [4]
    rare = next(row for row in result["per_class"] if row["class_id"] == 4)
    assert rare["eligible_for_attack_aggregate"] is False
    assert rare["support"] == 4


def test_class_macro_not_sample_weighted():
    # Losing 2/10 examples from class 2 gives -0.2 recall; class 4 is 100/100
    # and unchanged. Equal class macro recall delta = -0.1, regardless of sizes.
    actual = np.array([0] * 10 + [2] * 10 + [4] * 100, dtype=np.int64)
    global_pred = actual.copy()
    candidate = actual.copy()
    candidate[10:12] = 0
    result = calculate_cavc_v2(actual, global_pred, candidate, min_class_support=5)
    assert abs(result["attack_recall_contribution"] - (-0.1)) < 1e-12


def test_mismatched_lengths_raise():
    actual = np.array([0, 2, 4])
    try:
        calculate_cavc_v2(actual, actual, actual[:2], min_class_support=1)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched lengths.")


def test_no_eligible_attack_class_raises():
    actual = np.array([0] * 20 + [2] * 2, dtype=np.int64)
    try:
        calculate_cavc_v2(actual, actual, actual, min_class_support=10)
    except ValueError as exc:
        assert "No attack class" in str(exc)
        return
    raise AssertionError("Expected ValueError when no attack class meets support threshold.")


def test_cross_client_rank_not_within_client_round_rank():
    rows = [
        {"client_id": 1, "attack_detection_contribution_score": 0.4},
        {"client_id": 2, "attack_detection_contribution_score": 0.6},
        {"client_id": 3, "attack_detection_contribution_score": 0.5},
    ]
    ranked = rank_clients_within_condition(rows)
    by_id = {row["client_id"]: row["rank_lowest_first"] for row in ranked}
    assert by_id == {1: 1.0, 2: 3.0, 3: 2.0}


def main():
    tests = [
        test_no_change_neutral_attack_contribution,
        test_attack_recall_improvement,
        test_targeted_attack_to_benign_damage_is_separate,
        test_benign_false_positive_penalty,
        test_minimum_support_excludes_rare_attacks,
        test_class_macro_not_sample_weighted,
        test_mismatched_lengths_raise,
        test_no_eligible_attack_class_raises,
        test_cross_client_rank_not_within_client_round_rank,
    ]
    print("CAVC V2 SYNTHETIC UNIT TESTS")
    print("=" * 72)
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("-" * 72)
    print(f"All {len(tests)} CAVC V2 tests passed.")


if __name__ == "__main__":
    main()
