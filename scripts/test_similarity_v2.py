from __future__ import annotations
import unittest
import torch
from trfa.trust_similarity_v2 import (
    calculate_reference_update_v2,
    calculate_update_similarity_v2,
)

class SimilarityV2Tests(unittest.TestCase):
    def test_reference_is_coordinate_median(self):
        updates = {1: torch.tensor([1., 2.]), 2: torch.tensor([2., 4.]), 3: torch.tensor([100., -5.])}
        ref = calculate_reference_update_v2(updates)
        self.assertTrue(torch.allclose(ref, torch.tensor([2., 2.], dtype=torch.float64)))

    def test_scores_are_bounded(self):
        updates = {1: torch.tensor([1., 2.]), 2: torch.tensor([1.1, 2.1]), 3: torch.tensor([0.9, 1.9])}
        result = calculate_update_similarity_v2(updates)
        for row in result.values():
            for key in ("direction_score_to_reference", "pairwise_direction_score", "magnitude_agreement_score", "reference_distance_agreement_score"):
                self.assertGreaterEqual(row[key], 0.0)
                self.assertLessEqual(row[key], 1.0)

    def test_large_magnitude_outlier_is_visible(self):
        updates = {1: torch.tensor([1., 0.]), 2: torch.tensor([1.1, 0.]), 3: torch.tensor([0.9, 0.]), 4: torch.tensor([20., 0.])}
        result = calculate_update_similarity_v2(updates)
        self.assertGreater(result[4]["relative_magnitude"], 10.0)
        self.assertLess(result[4]["magnitude_agreement_score"], 0.2)

    def test_reversed_direction_is_visible(self):
        updates = {1: torch.tensor([1., 0.]), 2: torch.tensor([1.1, 0.]), 3: torch.tensor([0.9, 0.]), 4: torch.tensor([-1., 0.])}
        result = calculate_update_similarity_v2(updates)
        self.assertLess(result[4]["direction_score_to_reference"], 0.1)

    def test_identical_updates_have_full_magnitude_agreement(self):
        updates = {1: torch.tensor([1., 2.]), 2: torch.tensor([1., 2.]), 3: torch.tensor([1., 2.])}
        result = calculate_update_similarity_v2(updates)
        for row in result.values():
            self.assertAlmostEqual(row["magnitude_agreement_score"], 1.0)
            self.assertAlmostEqual(row["reference_distance_agreement_score"], 1.0)

    def test_zero_updates_are_handled(self):
        updates = {1: torch.zeros(3), 2: torch.zeros(3), 3: torch.zeros(3)}
        result = calculate_update_similarity_v2(updates)
        for row in result.values():
            self.assertAlmostEqual(row["direction_score_to_reference"], 1.0)
            self.assertAlmostEqual(row["magnitude_agreement_score"], 1.0)

    def test_dimension_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            calculate_update_similarity_v2({1: torch.ones(2), 2: torch.ones(3)})

    def test_nonfinite_update_rejected(self):
        with self.assertRaises(ValueError):
            calculate_update_similarity_v2({1: torch.tensor([1., float('nan')]), 2: torch.ones(2)})

    def test_single_client_rejected(self):
        with self.assertRaises(ValueError):
            calculate_update_similarity_v2({1: torch.ones(2)})

if __name__ == "__main__":
    unittest.main(verbosity=2)
