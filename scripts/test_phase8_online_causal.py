import unittest
import numpy as np
import pandas as pd

from scripts.run_phase8_online_causal import causal_round_scores, percentile


def toy_frames():
    rows_s, rows_h = [], []
    for r in range(1, 5):
        for c in range(1, 11):
            # Deterministic client-specific profiles; client 10 changes from round 2.
            rel = 1.0 + 0.02 * c
            if c == 10 and r >= 2:
                rel *= (1.0 + 0.25 * (r - 1))
            rows_s.append({
                'Round': r, 'Client': c,
                'pairwise_direction_score': 0.95 - 0.01 * c - (0.08 if c == 10 and r >= 2 else 0),
                'direction_score_to_reference': 0.96 - 0.01 * c - (0.07 if c == 10 and r >= 2 else 0),
                'relative_magnitude': rel,
                'magnitude_agreement_score': 1.0 / max(rel, 1e-9),
                'reference_distance_agreement_score': 0.9 - 0.005 * c - (0.04 if c == 10 and r >= 2 else 0),
            })
            rows_h.append({'Round': r, 'Client': c, 'Combined_Trust': 0.8 - 0.005 * c - (0.04 if c == 10 and r >= 2 else 0)})
    return pd.DataFrame(rows_s), pd.DataFrame(rows_h)


class TestPhase8Causal(unittest.TestCase):
    def test_percentile_low_direction_is_suspicious(self):
        s = pd.Series([0.1, 0.5, 0.9])
        p = percentile(s, False)
        self.assertGreater(p.iloc[0], p.iloc[2])

    def test_round_one_temporal_score_is_neutral(self):
        sim, hist = toy_frames()
        out = causal_round_scores(sim, hist, upto_round=1)
        self.assertTrue(np.allclose(out['Temporal_Instability_Group'], 0.5))

    def test_no_future_information_leakage(self):
        sim, hist = toy_frames()
        base = causal_round_scores(sim, hist, upto_round=3)
        sim2, hist2 = sim.copy(), hist.copy()
        sim2.loc[sim2['Round'] == 4, 'relative_magnitude'] *= 100.0
        sim2.loc[sim2['Round'] == 4, 'pairwise_direction_score'] = 0.0
        hist2.loc[hist2['Round'] == 4, 'Combined_Trust'] = 0.0
        changed_future = causal_round_scores(sim2, hist2, upto_round=3)
        cols = ['Direction_Anomaly_Group', 'Temporal_Instability_Group', 'CCTRT_Causal_Score']
        pd.testing.assert_frame_equal(base[cols], changed_future[cols])

    def test_output_has_only_rounds_requested(self):
        sim, hist = toy_frames()
        out = causal_round_scores(sim, hist, upto_round=2)
        self.assertEqual(sorted(out['Round'].unique().tolist()), [1, 2])
        self.assertEqual(len(out), 20)

    def test_client_local_temporal_change(self):
        sim, hist = toy_frames()
        out = causal_round_scores(sim, hist, upto_round=3)
        round3 = out[out['Round'] == 3].set_index('Client')
        self.assertGreater(round3.loc[10, 'Temporal_Instability_Group'], round3.loc[1, 'Temporal_Instability_Group'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
