import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def percentile(values, high_is_suspicious):
    return values.rank(method='average', ascending=high_is_suspicious, pct=True)


def mean_abs_step(frame, col, transform='identity'):
    tmp = frame[['Client', 'Round', col]].copy().sort_values(['Client', 'Round'])
    x = tmp[col].astype(float)
    if transform == 'log':
        x = np.log(np.clip(x, 1e-12, None))
    tmp['_x'] = x
    tmp['_step'] = tmp.groupby('Client')['_x'].diff().abs()
    return tmp.groupby('Client')['_step'].mean()


class TestCCTRTV1(unittest.TestCase):
    def test_percentile_lower_is_suspicious(self):
        s = pd.Series([0.1, 0.2, 0.9])
        p = percentile(s, False)
        self.assertGreater(p.iloc[0], p.iloc[2])

    def test_percentile_higher_is_suspicious(self):
        s = pd.Series([0.1, 0.2, 0.9])
        p = percentile(s, True)
        self.assertGreater(p.iloc[2], p.iloc[0])

    def test_temporal_step_is_client_local(self):
        df = pd.DataFrame({'Client':[1,1,1,2,2,2], 'Round':[1,2,3,1,2,3], 'x':[1,2,4,10,10,10]})
        v = mean_abs_step(df, 'x')
        self.assertAlmostEqual(v.loc[1], 1.5)
        self.assertAlmostEqual(v.loc[2], 0.0)

    def test_log_temporal_step_handles_positive_ratios(self):
        df = pd.DataFrame({'Client':[1,1,1,2,2,2], 'Round':[1,2,3,1,2,3], 'ratio':[1,2,4,1,1,1]})
        v = mean_abs_step(df, 'ratio', 'log')
        self.assertGreater(v.loc[1], 0.0)
        self.assertAlmostEqual(v.loc[2], 0.0)

    def test_auc_one_attacker_rank_order(self):
        y = np.array([0,0,1,0])
        score = np.array([0.1,0.2,0.9,0.3])
        self.assertAlmostEqual(roc_auc_score(y, score), 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
