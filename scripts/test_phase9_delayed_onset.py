from __future__ import annotations
import unittest
import numpy as np

from scripts.phase9_cctrt_core import OnlineCCTRTState, attack_active_this_round, poison_eligible_labels


class TestPhase9DelayedOnset(unittest.TestCase):
    def test_poison_fraction_is_based_on_eligible_nonbenign(self):
        y = np.array([0, 1, 2, 0, 3, 4, 0, 5, 6, 0], dtype=np.int64)
        p, changed, eligible = poison_eligible_labels(y, 5, 0.5, 2028, 0)
        self.assertEqual(eligible, 6)
        self.assertEqual(changed, 3)
        self.assertEqual(int(np.count_nonzero(p != y)), 3)


    def test_attack_onset_is_inclusive_and_warmup_is_clean(self):
        self.assertFalse(attack_active_this_round(1, 4))
        self.assertFalse(attack_active_this_round(3, 4))
        self.assertTrue(attack_active_this_round(4, 4))
        self.assertTrue(attack_active_this_round(10, 4))
        self.assertFalse(attack_active_this_round(5, 6))
        self.assertTrue(attack_active_this_round(6, 6))

    def test_poisoning_deterministic(self):
        y = np.array([0, 1, 2, 3, 4, 5, 6, 1, 2, 3], dtype=np.int64)
        a = poison_eligible_labels(y, 7, 0.2, 2028, 0)
        b = poison_eligible_labels(y, 7, 0.2, 2028, 0)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[1:], b[1:])

    def test_round_one_temporal_group_is_neutral(self):
        state = OnlineCCTRTState(list(range(1, 11)))
        sim = {
            cid: {
                "pairwise_direction_score": 0.8 - cid * 0.01,
                "direction_score_to_reference": 0.75 - cid * 0.01,
                "relative_magnitude": 1.0 + cid * 0.01,
                "magnitude_agreement_score": 0.9 - cid * 0.01,
                "reference_distance_agreement_score": 0.85 - cid * 0.01,
            }
            for cid in range(1, 11)
        }
        trust = {cid: 0.5 for cid in range(1, 11)}
        scored = state.score_round(1, sim, trust)
        self.assertTrue(all(abs(v["Temporal_Instability_Group"] - 0.5) < 1e-12 for v in scored.values()))
        self.assertTrue(all(v["Round_Used_Through"] == 1 for v in scored.values()))

    def test_temporal_state_advances_only_consecutive_rounds(self):
        state = OnlineCCTRTState(list(range(1, 11)))
        sim = {
            cid: {
                "pairwise_direction_score": 0.8,
                "direction_score_to_reference": 0.8,
                "relative_magnitude": 1.0,
                "magnitude_agreement_score": 1.0,
                "reference_distance_agreement_score": 1.0,
            }
            for cid in range(1, 11)
        }
        trust = {cid: 0.5 for cid in range(1, 11)}
        state.score_round(1, sim, trust)
        with self.assertRaises(ValueError):
            state.score_round(3, sim, trust)

    def test_future_rounds_cannot_change_already_emitted_scores(self):
        # OnlineCCTRTState emits and returns a round's score immediately; later
        # inputs are not arguments to or dependencies of the earlier call.
        state_a = OnlineCCTRTState(list(range(1, 11)))
        state_b = OnlineCCTRTState(list(range(1, 11)))
        base = {
            cid: {
                "pairwise_direction_score": 0.7 + cid * 0.01,
                "direction_score_to_reference": 0.7 + cid * 0.01,
                "relative_magnitude": 1.0,
                "magnitude_agreement_score": 1.0,
                "reference_distance_agreement_score": 1.0,
            }
            for cid in range(1, 11)
        }
        trust = {cid: 0.5 for cid in range(1, 11)}
        round1_a = state_a.score_round(1, base, trust)
        round1_b = state_b.score_round(1, base, trust)
        self.assertEqual(round1_a, round1_b)
        altered_future = {
            cid: dict(row, relative_magnitude=(10.0 if cid == 5 else 0.1))
            for cid, row in base.items()
        }
        state_a.score_round(2, altered_future, trust)
        # Previously returned round-one scores are immutable outputs.
        self.assertEqual(round1_a, round1_b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
