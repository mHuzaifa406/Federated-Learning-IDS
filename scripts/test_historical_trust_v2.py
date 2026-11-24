"""Unit tests for TRFA Historical Trust V2.

Run from the project root with:
    python scripts\\test_historical_trust_v2.py
"""

from pathlib import Path
import sys
import unittest

# Make the project root importable when this file is executed directly
# from the scripts directory by Python.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trfa.historical_trust_v2 import (
    HistoricalTrustTrackerV2,
    calculate_combined_trust_v2,
    calculate_current_evidence,
)


class HistoricalTrustV2Tests(unittest.TestCase):
    def test_initial_trust_is_neutral(self):
        tracker = HistoricalTrustTrackerV2([1, 2, 3])
        self.assertEqual(tracker.get_scores(), {1: 0.5, 2: 0.5, 3: 0.5})

    def test_current_evidence_excludes_history(self):
        evidence = calculate_current_evidence({1: 0.8}, {1: 0.6})
        self.assertAlmostEqual(evidence[1], 0.7)

    def test_ema_update_uses_current_evidence(self):
        tracker = HistoricalTrustTrackerV2([1], retention=0.9, initial_trust=0.5)
        history = tracker.update({1: 0.7}, round_number=1)
        self.assertAlmostEqual(history[1], 0.52)

    def test_combined_trust_uses_previous_history(self):
        scores = calculate_combined_trust_v2(
            {1: 0.8}, {1: 0.6}, {1: 0.5},
            alpha=1 / 3, beta=1 / 3, gamma=1 / 3,
        )
        self.assertAlmostEqual(scores[1], (0.8 + 0.6 + 0.5) / 3)

    def test_round_updates_must_be_consecutive(self):
        tracker = HistoricalTrustTrackerV2([1])
        tracker.update({1: 0.6}, round_number=1)
        with self.assertRaises(ValueError):
            tracker.update({1: 0.6}, round_number=1)

    def test_all_clients_must_have_evidence(self):
        tracker = HistoricalTrustTrackerV2([1, 2])
        with self.assertRaises(ValueError):
            tracker.update({1: 0.7}, round_number=1)

    def test_unknown_client_is_rejected(self):
        tracker = HistoricalTrustTrackerV2([1, 2])
        with self.assertRaises(ValueError):
            tracker.update({1: 0.7, 2: 0.6, 3: 0.4}, round_number=1)

    def test_invalid_score_is_rejected(self):
        tracker = HistoricalTrustTrackerV2([1])
        with self.assertRaises(ValueError):
            tracker.update({1: 1.2}, round_number=1)

    def test_checkpoint_round_trip(self):
        tracker = HistoricalTrustTrackerV2([1, 2], retention=0.9, initial_trust=0.5)
        tracker.update({1: 0.7, 2: 0.3}, round_number=1)
        state = tracker.state_dict()
        restored = HistoricalTrustTrackerV2([1, 2], retention=0.9, initial_trust=0.5)
        restored.load_state_dict(state)
        self.assertEqual(restored.get_scores(), tracker.get_scores())
        self.assertEqual(restored.last_updated_round, 1)

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            calculate_combined_trust_v2(
                {1: 0.8}, {1: 0.7}, {1: 0.5},
                alpha=0.4, beta=0.4, gamma=0.4,
            )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HistoricalTrustV2Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
