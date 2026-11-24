import unittest
import numpy as np
import pandas as pd

from scripts.phase10_bctcd_core import (
    BASE_FEATURES, RAW_TO_FEATURE, fit_client_baselines,
    prepare_features, replay_bctcd, score_online_round,
)


def make_frame(rounds=6, attack_start=4, target=5):
    rows = []
    for rnd in range(1, rounds + 1):
        for cid in range(1, 11):
            attacked = cid == target and rnd >= attack_start
            shift = 0.45 if attacked else 0.0
            rows.append({
                "Round": rnd, "Client": cid, "Is_Malicious": cid == target,
                "Attack_Active": attacked, "Attack_Start_Round": attack_start,
                "relative_magnitude": float(np.exp(0.1 * cid + shift * (1 if attacked else 0))),
                "magnitude_agreement_score": 0.55 + 0.01 * cid + shift * (1 if attacked else 0),
                "reference_distance_agreement_score": 0.60 + 0.005 * cid + shift * (1 if attacked else 0),
                "Combined_Trust": 0.50 + 0.01 * cid + shift * (1 if attacked else 0),
                "pairwise_direction_score": 0.70 - 0.005 * cid - shift * (1 if attacked else 0),
                "direction_score_to_reference": 0.72 - 0.004 * cid - shift * (1 if attacked else 0),
            })
    return pd.DataFrame(rows)


class TestBCTCD(unittest.TestCase):
    def test_features_log_relative_magnitude(self):
        frame = make_frame()
        prepared = prepare_features(frame)
        self.assertTrue(np.allclose(prepared["Log_Relative_Magnitude"], np.log(frame["relative_magnitude"])))

    def test_baselines_use_only_warmup(self):
        frame = make_frame()
        stats_a, _ = fit_client_baselines(prepare_features(frame), 3)
        changed = frame.copy()
        changed.loc[changed["Round"] > 3, "Combined_Trust"] += 100
        stats_b, _ = fit_client_baselines(prepare_features(changed), 3)
        self.assertEqual(stats_a[(5, "Combined_Trust")], stats_b[(5, "Combined_Trust")])

    def test_scale_floor_prevents_zero_scale(self):
        frame = make_frame()
        frame.loc[:, "Combined_Trust"] = 0.5
        stats, _ = fit_client_baselines(prepare_features(frame), 3)
        self.assertGreaterEqual(stats[(1, "Combined_Trust")].effective_scale, 0.05)

    def test_attacked_client_deviation_rises(self):
        frame = make_frame()
        scores, metrics, _ = replay_bctcd(frame, warmup_rounds=3, slack=0.5)
        r4 = scores[scores["Round"] == 4].set_index("Client")
        self.assertGreater(r4.loc[5, "BCTCD_Round_Deviation"], r4.loc[1, "BCTCD_Round_Deviation"])

    def test_cusum_nonnegative_and_accumulates(self):
        frame = make_frame()
        scores, _, _ = replay_bctcd(frame, warmup_rounds=3, slack=0.0)
        target = scores[scores["Client"] == 5].sort_values("Round")
        self.assertTrue((target["BCTCD_CUSUM_Score"] >= 0).all())
        self.assertGreater(target.iloc[-1]["BCTCD_CUSUM_Score"], 0)

    def test_common_mode_shift_is_removed_by_cohort_centering(self):
        frame = prepare_features(make_frame(rounds=6, attack_start=6))
        baselines, _ = fit_client_baselines(frame, 3)
        round4 = frame[frame["Round"] == 4].copy()
        round4["Combined_Trust"] += 0.20
        scored, _ = score_online_round(
            round4, baselines, {i: 0.0 for i in range(1, 11)}, slack=1.0
        )
        self.assertTrue(np.allclose(scored["Combined_Trust_Cohort_Centered_Abs_Z"], 0.0, atol=1e-8))

    def test_round_score_independent_of_future_rows(self):
        frame = prepare_features(make_frame())
        baseline, _ = fit_client_baselines(frame, 3)
        round4 = frame[frame["Round"] == 4]
        first, _ = score_online_round(round4, baseline, {i: 0.0 for i in range(1, 11)}, slack=0.5)
        modified_future = frame.copy()
        modified_future.loc[modified_future["Round"] > 4, list(BASE_FEATURES)] = 999
        second, _ = score_online_round(modified_future[modified_future["Round"] == 4], baseline,
                                       {i: 0.0 for i in range(1, 11)}, slack=0.5)
        self.assertTrue(np.allclose(first["BCTCD_CUSUM_Score"], second["BCTCD_CUSUM_Score"]))

    def test_rejects_bad_attack_flags(self):
        frame = make_frame()
        frame.loc[(frame.Client == 5) & (frame.Round == 4), "Attack_Active"] = False
        with self.assertRaises(ValueError):
            replay_bctcd(frame, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
