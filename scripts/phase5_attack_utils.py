"""Shared deterministic label-flipping utility for Phase 5 diagnostics."""
from __future__ import annotations
import numpy as np


def poison_eligible_labels(original_y, client_id: int, poison_fraction: float,
                           poison_seed: int, target_label: int = 0):
    """Flip floor(fraction * eligible_non_target_records) to target_label.

    Returns (new_labels, changed_count, eligible_count). Only labels different
    from target_label are eligible; existing target labels are never selected.
    Selection is deterministic for a fixed poison_seed and client_id.
    """
    fraction = float(poison_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("poison_fraction must be finite and strictly between 0 and 1.")
    y = np.asarray(original_y, dtype=np.int64)
    poisoned = np.array(y, dtype=np.int64, copy=True)
    eligible = np.flatnonzero(poisoned != int(target_label))
    count = int(np.floor(len(eligible) * fraction))
    if count < 1:
        raise ValueError(f"Poison fraction produces zero records; eligible count={len(eligible)}.")
    rng = np.random.default_rng(int(poison_seed) + int(client_id))
    selected = rng.choice(eligible, size=count, replace=False)
    poisoned[selected] = int(target_label)
    actual = int(np.count_nonzero(poisoned != y))
    if actual != count:
        raise RuntimeError(f"Poison-count verification failed: requested {count}, changed {actual}.")
    return poisoned, actual, int(len(eligible))
