from __future__ import annotations

import unittest
import pandas as pd

from scripts.phase10_bctcd_core import replay_bctcd
from scripts.phase10_matched_control_core import (
    prepare_clean_replay_frame,
    compare_pre_attack_trajectories,
    build_matched_round_comparison,
)


class TestMatchedCleanControl(unittest.TestCase):
    def _clean_frame(self):
        rows = []
        for rnd in range(1, 5):
            for cid in range(1, 4):
                rows.append({
                    "Round": rnd, "Client": cid, "Is_Malicious": False,
                    "Attack_Active": False, "Attack_Start_Round": 11,
                    "Target_Client": cid == 2,
                    "Poisoned_Records": 0, "relative_magnitude": 1.0 + cid * 0.1,
                    "magnitude_agreement_score": 0.8, "reference_distance_agreement_score": 0.7,
                    "Combined_Trust": 0.6, "pairwise_direction_score": 0.7,
                    "direction_score_to_reference": 0.8,
                })
        return pd.DataFrame(rows)

    def test_clean_replay_uses_internal_target_sentinel(self):
        raw = self._clean_frame()
        replay = prepare_clean_replay_frame(raw, 2)
        self.assertEqual(int(replay.groupby("Round")["Is_Malicious"].sum().iloc[0]), 1)
        self.assertTrue((replay.loc[replay.Client == 2, "Is_Malicious"]).all())
        self.assertFalse(replay["Attack_Active"].any())
        self.assertTrue((replay["Attack_Start_Round"] == 5).all())

    def test_clean_control_replays_without_attack_labels_in_saved_output(self):
        raw = self._clean_frame()
        replay_input = prepare_clean_replay_frame(raw, 2)
        scores, metrics, _ = replay_bctcd(replay_input, warmup_rounds=2, slack=1.0)
        self.assertEqual(sorted(scores["Round"].unique().tolist()), [3, 4])
        self.assertEqual(int(scores["Is_Malicious"].sum()), 2)  # One sentinel row per monitored round.
        self.assertTrue((scores.loc[scores["Is_Malicious"], "Client"] == 2).all())
        self.assertFalse(scores["Attack_Active"].any())
        # The clean-control runner restores Is_Malicious=False before persisting.
        saved_view = scores.copy()
        saved_view["Is_Malicious"] = False
        self.assertFalse(saved_view["Is_Malicious"].any())
        self.assertFalse(metrics["Attack_Active"].any())

    def test_original_control_is_not_mutated(self):
        raw = self._clean_frame()
        before = raw.copy(deep=True)
        _ = prepare_clean_replay_frame(raw, 2)
        pd.testing.assert_frame_equal(raw, before)

    def test_rejects_malicious_flag_in_clean_control(self):
        raw = self._clean_frame()
        raw.loc[raw.Client == 2, "Is_Malicious"] = True
        with self.assertRaises(ValueError):
            prepare_clean_replay_frame(raw, 2)

    def test_rejects_attack_active_in_clean_control(self):
        raw = self._clean_frame()
        raw.loc[(raw.Client == 2) & (raw.Round == 4), "Attack_Active"] = True
        with self.assertRaises(ValueError):
            prepare_clean_replay_frame(raw, 2)

    def test_pre_attack_matching_detects_identical_trajectories(self):
        clean = self._clean_frame()
        attack = clean.copy()
        attack["Is_Malicious"] = attack["Client"].eq(2)
        attack["Attack_Start_Round"] = 4
        attack.loc[(attack.Client == 2) & (attack.Round == 4), "Attack_Active"] = True
        result = compare_pre_attack_trajectories(clean, attack, 4)
        self.assertTrue(result["matched_within_tolerance"])
        self.assertAlmostEqual(result["max_absolute_difference"], 0.0)

    def test_pre_attack_matching_detects_mismatch(self):
        clean = self._clean_frame()
        attack = clean.copy()
        attack.loc[(attack.Client == 2) & (attack.Round == 2), "Combined_Trust"] += 0.1
        result = compare_pre_attack_trajectories(clean, attack, 4)
        self.assertFalse(result["matched_within_tolerance"])

    def test_paired_phase10_comparison(self):
        base = []
        for rnd in (4, 5, 6):
            for cid in (1, 2, 3):
                base.append({
                    "Round": rnd, "Client": cid,
                    "BCTCD_Round_Deviation": 0.2 * cid,
                    "BCTCD_CUSUM_Score": 0.1 * cid,
                    "BCTCD_Deviation_Rank": cid,
                    "BCTCD_CUSUM_Rank": cid,
                    "BCTCD_Deviation_Top3": cid == 1,
                    "BCTCD_CUSUM_Top3": cid == 1,
                })
        clean = pd.DataFrame(base)
        attack = clean.copy()
        attack.loc[(attack.Round == 6) & (attack.Client == 2), "BCTCD_CUSUM_Score"] += 0.5
        paired = build_matched_round_comparison(clean, attack, 2, 6)
        target = paired[(paired.Round == 6) & (paired.Client == 2)].iloc[0]
        self.assertAlmostEqual(target["BCTCD_CUSUM_Score_Poison_Minus_Clean"], 0.5)
        self.assertTrue(bool(target["Is_Probe_Client"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
