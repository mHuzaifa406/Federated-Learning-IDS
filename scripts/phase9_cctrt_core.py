"""Pure utility functions for Phase 9 delayed-onset CCTRT experiments."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def poison_eligible_labels(original_y: np.ndarray, client_id: int, poison_fraction: float,
                           poison_seed: int, target_label: int = 0) -> tuple[np.ndarray, int, int]:
    """Deterministically flip floor(fraction * non-target labels) to target."""
    fraction = float(poison_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("poison_fraction must be finite and strictly between 0 and 1.")
    y = np.asarray(original_y, dtype=np.int64)
    poisoned = np.array(y, dtype=np.int64, copy=True)
    eligible = np.flatnonzero(poisoned != int(target_label))
    count = int(math.floor(len(eligible) * fraction))
    if count < 1:
        raise ValueError(f"Poison fraction produces zero records; eligible count={len(eligible)}.")
    rng = np.random.default_rng(int(poison_seed) + int(client_id))
    selected = rng.choice(eligible, size=count, replace=False)
    poisoned[selected] = int(target_label)
    actual = int(np.count_nonzero(poisoned != y))
    if actual != count:
        raise RuntimeError(f"Poison-count verification failed: expected {count}, changed {actual}.")
    return poisoned, actual, int(len(eligible))



def attack_active_this_round(round_number: int, attack_start_round: int) -> bool:
    """Return whether poisoning is active in this round (inclusive onset)."""
    if not isinstance(round_number, int) or not isinstance(attack_start_round, int):
        raise TypeError("Round numbers must be integers.")
    if round_number < 1 or attack_start_round not in (4, 6):
        raise ValueError("round_number must be positive and attack_start_round must be 4 or 6.")
    return round_number >= attack_start_round

def _percentile(values: list[float], high_is_suspicious: bool) -> list[float]:
    series = pd.Series(values, dtype=float)
    if series.isna().any() or not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("CCTRT received a missing or non-finite value.")
    return series.rank(method="average", ascending=high_is_suspicious, pct=True).tolist()


class OnlineCCTRTState:
    """Maintains causal temporal steps; update exactly once after each round."""
    TEMPORAL_COLUMNS = (
        "Log_Relative_Magnitude",
        "Magnitude_Agreement",
        "Reference_Distance_Agreement",
        "Combined_Trust",
    )

    def __init__(self, client_ids: list[int]):
        self.client_ids = list(client_ids)
        self.previous: dict[int, dict[str, float]] = {}
        self.step_sums = {cid: {col: 0.0 for col in self.TEMPORAL_COLUMNS} for cid in self.client_ids}
        self.step_counts = {cid: 0 for cid in self.client_ids}
        self.last_round = 0

    def score_round(self, round_number: int, similarity_rows: dict[int, dict[str, float]],
                    combined_trust: dict[int, float]) -> dict[int, dict[str, float]]:
        if round_number != self.last_round + 1:
            raise ValueError(f"CCTRT expected round {self.last_round + 1}; received {round_number}.")
        if set(similarity_rows) != set(self.client_ids) or set(combined_trust) != set(self.client_ids):
            raise ValueError("CCTRT client IDs are incomplete or inconsistent.")

        current: dict[int, dict[str, float]] = {}
        for cid in self.client_ids:
            sim = similarity_rows[cid]
            ratio = float(sim["relative_magnitude"])
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"Client {cid} has invalid relative_magnitude={ratio}.")
            now = {
                "Log_Relative_Magnitude": float(math.log(max(ratio, 1e-12))),
                "Magnitude_Agreement": float(sim["magnitude_agreement_score"]),
                "Reference_Distance_Agreement": float(sim["reference_distance_agreement_score"]),
                "Combined_Trust": float(combined_trust[cid]),
            }
            if any(not math.isfinite(v) for v in now.values()):
                raise ValueError(f"Client {cid} has non-finite CCTRT input.")
            current[cid] = now

        if round_number > 1:
            for cid in self.client_ids:
                for col in self.TEMPORAL_COLUMNS:
                    self.step_sums[cid][col] += abs(current[cid][col] - self.previous[cid][col])
                self.step_counts[cid] += 1

        pairwise = [float(similarity_rows[cid]["pairwise_direction_score"]) for cid in self.client_ids]
        reference = [float(similarity_rows[cid]["direction_score_to_reference"]) for cid in self.client_ids]
        pairwise_anomaly = _percentile(pairwise, high_is_suspicious=False)
        reference_anomaly = _percentile(reference, high_is_suspicious=False)

        if round_number == 1:
            temporal_group = [0.5] * len(self.client_ids)
            temporal_features = {
                col: ([0.0] * len(self.client_ids), [0.5] * len(self.client_ids))
                for col in self.TEMPORAL_COLUMNS
            }
        else:
            temporal_features = {}
            for col in self.TEMPORAL_COLUMNS:
                means = [self.step_sums[cid][col] / max(self.step_counts[cid], 1) for cid in self.client_ids]
                temporal_features[col] = (means, _percentile(means, high_is_suspicious=True))
            temporal_group = [
                float(np.mean([temporal_features[col][1][idx] for col in self.TEMPORAL_COLUMNS]))
                for idx in range(len(self.client_ids))
            ]

        result: dict[int, dict[str, float]] = {}
        for idx, cid in enumerate(self.client_ids):
            direction_group = 0.5 * (pairwise_anomaly[idx] + reference_anomaly[idx])
            record: dict[str, Any] = {
                "Direction_Pairwise_Anomaly": float(pairwise_anomaly[idx]),
                "Direction_Reference_Anomaly": float(reference_anomaly[idx]),
                "Direction_Anomaly_Group": float(direction_group),
                "Temporal_Instability_Group": float(temporal_group[idx]),
                "CCTRT_Causal_Score": float(0.5 * direction_group + 0.5 * temporal_group[idx]),
                "Round_Used_Through": int(round_number),
            }
            for col in self.TEMPORAL_COLUMNS:
                mean_step, rank_values = temporal_features[col]
                record[f"Cumulative_Mean_Abs_Step_{col}"] = float(mean_step[idx])
                record[f"Temporal_{col}_Percentile"] = float(rank_values[idx])
            result[cid] = record

        # Advance history only after emitting scores for this round.
        self.previous = current
        self.last_round = round_number
        return result
