"""Synthetic unit tests for Trust Fusion V1 score direction and validation."""
import unittest
import pandas as pd

from scripts.run_trust_fusion_v1 import suspicion_percentile, _normalise_condition, _weight_grid


class TrustFusionV1Tests(unittest.TestCase):
    def test_low_value_is_most_suspicious_when_lower_is_bad(self):
        values = pd.Series([0.2, 0.5, 0.9])
        scores = suspicion_percentile(values, higher_is_suspicious=False)
        self.assertGreater(scores.iloc[0], scores.iloc[1])
        self.assertGreater(scores.iloc[1], scores.iloc[2])

    def test_high_value_is_most_suspicious_when_higher_is_bad(self):
        values = pd.Series([0.2, 0.5, 0.9])
        scores = suspicion_percentile(values, higher_is_suspicious=True)
        self.assertLess(scores.iloc[0], scores.iloc[1])
        self.assertLess(scores.iloc[1], scores.iloc[2])

    def test_ties_receive_equal_percentiles(self):
        scores = suspicion_percentile(pd.Series([1.0, 1.0, 1.0]), higher_is_suspicious=True)
        self.assertTrue((scores == scores.iloc[0]).all())

    def test_condition_normalization(self):
        self.assertEqual(_normalise_condition("poison_client07"), "poison-client07")
        self.assertEqual(_normalise_condition("clean"), "clean")

    def test_weight_grid_sums_to_one(self):
        grid = _weight_grid(0.1)
        self.assertEqual(len(grid), 66)
        for row in grid:
            self.assertAlmostEqual(row["similarity_weight"] + row["historical_weight"] + row["cavc_weight"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
