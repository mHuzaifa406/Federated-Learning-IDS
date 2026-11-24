import unittest
import numpy as np
from scripts.phase5_attack_utils import poison_eligible_labels


class TestPhase5AttackUtility(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0] * 20 + [1] * 30 + [2] * 25 + [4] * 25, dtype=np.int64)

    def test_fraction_is_applied_to_eligible_records(self):
        poisoned, count, eligible = poison_eligible_labels(self.labels, 5, 0.20, 2027)
        self.assertEqual(eligible, 80)
        self.assertEqual(count, 16)
        self.assertEqual(np.count_nonzero(poisoned != self.labels), 16)

    def test_existing_target_labels_remain_target(self):
        poisoned, _, _ = poison_eligible_labels(self.labels, 7, 0.05, 2027)
        self.assertTrue(np.all(poisoned[:20] == 0))

    def test_deterministic_for_same_seed_and_client(self):
        a, _, _ = poison_eligible_labels(self.labels, 9, 0.20, 2027)
        b, _, _ = poison_eligible_labels(self.labels, 9, 0.20, 2027)
        np.testing.assert_array_equal(a, b)

    def test_invalid_fraction_rejected(self):
        for value in (0, -0.1, 1, 1.5, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    poison_eligible_labels(self.labels, 1, value, 2027)

    def test_zero_change_rejected(self):
        with self.assertRaises(ValueError):
            poison_eligible_labels(np.array([0, 0, 0]), 1, 0.05, 2027)


if __name__ == "__main__":
    unittest.main()
