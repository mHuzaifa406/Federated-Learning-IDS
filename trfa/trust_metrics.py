"""
TRFA Trust Metrics — Version 2 (incremental development)
========================================================

This module implements:
1. Update similarity against a coordinate-wise median reference.
2. Temporal training consistency using each client's own update history.
3. Historical trust and combined trust scoring.
4. Threshold-based client screening.

Important research note
-----------------------
Temporal consistency measures whether a client's update direction and
magnitude remain reasonably stable across communication rounds. It is a
reliability signal, NOT proof that a client is benign. A persistent malicious
client can also behave consistently, so this signal must be combined with
other evidence and evaluated empirically.

Backward compatibility
----------------------
The old scalar-validation-loss function is retained as
`calculate_validation_improvement_score` and as a backward-compatible
`calculate_training_consistency` wrapper. New experiments should use
`TrainingConsistencyTracker` instead. Existing historical experiment scripts
will therefore continue to import the old function until they are deliberately
migrated.

This module does not train models or modify datasets.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F


EPSILON = 1e-12
TRUST_METRICS_VERSION = "2.0-incremental"


# ============================================================
# 1. CONVERT MODEL STATES INTO UPDATE VECTORS
# ============================================================

def flatten_model_update(local_state, global_state):
    """Return the flattened update Delta_i = local_model_i - global_model.

    The returned vector is a CPU float32 tensor. Model-state keys, tensor
    shapes, floating-point types, and finite values are checked.
    """
    if set(local_state.keys()) != set(global_state.keys()):
        raise ValueError("Local and global model keys do not match.")

    update_parts = []

    for key in global_state:
        local_tensor = local_state[key].detach().cpu()
        global_tensor = global_state[key].detach().cpu()

        if local_tensor.shape != global_tensor.shape:
            raise ValueError(f"Tensor shape mismatch for parameter: {key}")

        if not torch.is_floating_point(local_tensor):
            raise TypeError(f"Non-floating model parameter: {key}")

        if not torch.is_floating_point(global_tensor):
            raise TypeError(f"Non-floating global parameter: {key}")

        difference = local_tensor.float() - global_tensor.float()

        if not torch.isfinite(difference).all().item():
            raise ValueError(f"Non-finite update values for parameter: {key}")

        update_parts.append(difference.reshape(-1))

    if not update_parts:
        raise ValueError("No model parameters were supplied.")

    return torch.cat(update_parts)


# ============================================================
# 2. UPDATE SIMILARITY
# ============================================================

def _validated_update_vectors(client_updates):
    if not isinstance(client_updates, Mapping) or not client_updates:
        raise ValueError("client_updates must be a non-empty mapping.")

    vectors = {}
    expected_length = None

    for client_id, update in client_updates.items():
        if not isinstance(update, torch.Tensor):
            raise TypeError(f"Update for client {client_id} must be a tensor.")

        vector = update.detach().cpu().float().reshape(-1)

        if vector.numel() == 0:
            raise ValueError(f"Update for client {client_id} is empty.")

        if not torch.isfinite(vector).all().item():
            raise ValueError(f"Update for client {client_id} contains non-finite values.")

        if expected_length is None:
            expected_length = vector.numel()
        elif vector.numel() != expected_length:
            raise ValueError("Client update vectors have different lengths.")

        vectors[client_id] = vector

    return vectors


def calculate_reference_update(client_updates):
    """Calculate the coordinate-wise median reference update."""
    vectors = _validated_update_vectors(client_updates)
    stacked = torch.stack(list(vectors.values()), dim=0)
    return torch.quantile(stacked, 0.5, dim=0)


def calculate_update_similarity(client_updates, reference_update=None):
    """Calculate cosine similarity to the coordinate-wise median update.

    Cosine similarity is mapped from [-1, 1] to [0, 1]. Zero-vector cases
    receive neutral score 0.5. This score measures directional agreement;
    it does NOT detect unusually large update magnitude by itself.
    """
    vectors = _validated_update_vectors(client_updates)

    if reference_update is None:
        reference_update = calculate_reference_update(vectors)

    if not isinstance(reference_update, torch.Tensor):
        raise TypeError("reference_update must be a tensor.")

    reference = reference_update.detach().cpu().float().reshape(-1)

    if not torch.isfinite(reference).all().item():
        raise ValueError("reference_update contains non-finite values.")

    if any(vector.numel() != reference.numel() for vector in vectors.values()):
        raise ValueError("Update length differs from reference update length.")

    reference_norm = torch.linalg.vector_norm(reference)
    scores = {}

    for client_id, vector in vectors.items():
        vector_norm = torch.linalg.vector_norm(vector)

        if vector_norm.item() <= EPSILON or reference_norm.item() <= EPSILON:
            scores[client_id] = 0.5
            continue

        cosine = F.cosine_similarity(
            vector.unsqueeze(0),
            reference.unsqueeze(0),
            dim=1,
            eps=EPSILON,
        ).item()

        scores[client_id] = float(np.clip((1.0 + cosine) / 2.0, 0.0, 1.0))

    return scores


# ============================================================
# 3. TEMPORAL TRAINING CONSISTENCY — NEW VERSION
# ============================================================

class TrainingConsistencyTracker:
    """Track the temporal stability of each client's model updates.

    For rounds after the first observation, the score combines:
      - Directional consistency: mapped cosine similarity between the
        current update and that client's previous-round update.
      - Magnitude consistency: exp(-abs(log(current_norm / previous_norm))).

    Both components are in [0, 1]. The default combination gives them equal
    weight. A first observation receives neutral score 0.5 because there is no
    history against which to judge consistency.

    The tracker is deliberately stateful. Call `score()` before `update()` in
    each communication round. This prevents the current update from being
    compared with itself.

    This is a candidate operational definition for the proposed methodology;
    its usefulness against poisoning and non-IID heterogeneity must be tested.
    """

    def __init__(
        self,
        client_ids: Optional[Iterable] = None,
        direction_weight: float = 0.5,
        magnitude_weight: float = 0.5,
        initial_score: float = 0.5,
    ):
        weights = [float(direction_weight), float(magnitude_weight)]

        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("Consistency weights must be finite and non-negative.")

        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("Consistency weights must sum to 1.")

        if not math.isfinite(initial_score) or not 0.0 <= initial_score <= 1.0:
            raise ValueError("initial_score must be between 0 and 1.")

        self.direction_weight = weights[0]
        self.magnitude_weight = weights[1]
        self.initial_score = float(initial_score)
        self._previous_updates: Dict[object, torch.Tensor] = {}

        if client_ids is not None:
            # Keep the configured client list explicit; clients are still
            # initialized without an update history and receive neutral scores.
            self._client_ids = set(client_ids)
        else:
            self._client_ids = None

    @staticmethod
    def _direction_score(current: torch.Tensor, previous: torch.Tensor) -> float:
        current_norm = torch.linalg.vector_norm(current)
        previous_norm = torch.linalg.vector_norm(previous)

        if current_norm.item() <= EPSILON or previous_norm.item() <= EPSILON:
            return 0.5

        cosine = F.cosine_similarity(
            current.unsqueeze(0), previous.unsqueeze(0), dim=1, eps=EPSILON
        ).item()
        return float(np.clip((1.0 + cosine) / 2.0, 0.0, 1.0))

    @staticmethod
    def _magnitude_score(current: torch.Tensor, previous: torch.Tensor) -> float:
        current_norm = float(torch.linalg.vector_norm(current).item())
        previous_norm = float(torch.linalg.vector_norm(previous).item())

        if current_norm <= EPSILON and previous_norm <= EPSILON:
            return 0.5
        if current_norm <= EPSILON or previous_norm <= EPSILON:
            return 0.0

        log_ratio = abs(math.log(current_norm / previous_norm))
        return float(np.clip(math.exp(-log_ratio), 0.0, 1.0))

    def score(self, client_updates: Mapping) -> Dict:
        """Score current updates against each client's previous-round update."""
        vectors = _validated_update_vectors(client_updates)

        if self._client_ids is not None:
            unknown = set(vectors) - self._client_ids
            if unknown:
                raise KeyError(f"Unknown client IDs supplied: {sorted(unknown, key=str)}")

        scores = {}

        for client_id, current in vectors.items():
            previous = self._previous_updates.get(client_id)

            if previous is None:
                scores[client_id] = self.initial_score
                continue

            if previous.numel() != current.numel():
                raise ValueError(
                    f"Update size changed across rounds for client {client_id}."
                )

            direction = self._direction_score(current, previous)
            magnitude = self._magnitude_score(current, previous)

            combined = (
                self.direction_weight * direction
                + self.magnitude_weight * magnitude
            )
            scores[client_id] = float(np.clip(combined, 0.0, 1.0))

        return scores

    def update(self, client_updates: Mapping) -> None:
        """Store this round's updates after their consistency was scored."""
        vectors = _validated_update_vectors(client_updates)

        if self._client_ids is not None:
            unknown = set(vectors) - self._client_ids
            if unknown:
                raise KeyError(f"Unknown client IDs supplied: {sorted(unknown, key=str)}")

        for client_id, vector in vectors.items():
            previous = self._previous_updates.get(client_id)
            if previous is not None and previous.numel() != vector.numel():
                raise ValueError(
                    f"Update size changed across rounds for client {client_id}."
                )
            self._previous_updates[client_id] = vector.clone()

    def state_dict(self) -> dict:
        """Return a serializable tracker state for checkpointing."""
        return {
            "direction_weight": self.direction_weight,
            "magnitude_weight": self.magnitude_weight,
            "initial_score": self.initial_score,
            "previous_updates": {
                client_id: update.clone()
                for client_id, update in self._previous_updates.items()
            },
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore a tracker state saved by `state_dict()`."""
        required = {
            "direction_weight", "magnitude_weight", "initial_score", "previous_updates"
        }
        missing = required - set(state)
        if missing:
            raise ValueError(f"Tracker state is missing keys: {sorted(missing)}")

        direction_weight = float(state["direction_weight"])
        magnitude_weight = float(state["magnitude_weight"])
        weights = [direction_weight, magnitude_weight]

        if any(not math.isfinite(w) or w < 0.0 for w in weights):
            raise ValueError("Invalid consistency weights in tracker state.")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("Consistency weights in tracker state must sum to 1.")

        initial_score = float(state["initial_score"])
        if not math.isfinite(initial_score) or not 0.0 <= initial_score <= 1.0:
            raise ValueError("Invalid initial_score in tracker state.")

        restored = (
            _validated_update_vectors(state["previous_updates"])
            if state["previous_updates"]
            else {}
        )

        if self._client_ids is not None:
            unknown = set(restored) - self._client_ids
            if unknown:
                raise KeyError(
                    f"Tracker state contains unknown clients: {sorted(unknown, key=str)}"
                )

        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.initial_score = initial_score
        self._previous_updates = {
            key: value.clone() for key, value in restored.items()
        }


# ============================================================
# 4. LEGACY VALIDATION-IMPROVEMENT SCORE — RETAINED FOR OLD RUNS
# ============================================================

def calculate_validation_improvement_score(
    baseline_validation_loss,
    candidate_validation_losses,
):
    """Legacy score based on one-round global validation-loss improvement.

    Retained so existing historical scripts can be reproduced. New TRFA runs
    should use TrainingConsistencyTracker because this function is not a
    temporal consistency measure and can penalize honest non-IID clients.
    """
    baseline = float(baseline_validation_loss)

    if not math.isfinite(baseline):
        raise ValueError("Baseline validation loss must be finite.")
    if not candidate_validation_losses:
        raise ValueError("No candidate validation losses were supplied.")

    changes = {}
    for client_id, candidate_loss in candidate_validation_losses.items():
        candidate = float(candidate_loss)
        if not math.isfinite(candidate):
            raise ValueError(f"Non-finite validation loss for client {client_id}.")
        changes[client_id] = baseline - candidate

    absolute_changes = np.asarray(list(map(abs, changes.values())), dtype=np.float64)
    nonzero_changes = absolute_changes[absolute_changes > EPSILON]

    if len(nonzero_changes) == 0:
        return {client_id: 0.5 for client_id in changes}

    scale = max(float(np.median(nonzero_changes)), EPSILON)
    scores = {}

    for client_id, improvement in changes.items():
        scaled = float(np.clip(improvement / scale, -60.0, 60.0))
        scores[client_id] = float(1.0 / (1.0 + math.exp(-scaled)))

    return scores


def calculate_training_consistency(baseline_validation_loss, candidate_validation_losses):
    """Backward-compatible alias for the legacy validation-improvement score.

    Deprecated for new experiments. Use TrainingConsistencyTracker instead.
    """
    return calculate_validation_improvement_score(
        baseline_validation_loss,
        candidate_validation_losses,
    )


# ============================================================
# 5. HISTORICAL TRUST — EXISTING FORMULA RETAINED FOR THIS STEP
# ============================================================

class HistoricalTrustTracker:
    """Maintain one historical trust value per client using an EMA.

    NOTE: This version retains the existing `update(current_trust_scores)`
    behavior for backward compatibility. The feedback-loop issue will be
    addressed in a separate controlled revision, after consistency is tested.
    """

    def __init__(
        self,
        client_ids: Iterable,
        retention: float = 0.90,
        initial_trust: float = 0.50,
    ):
        if not 0.0 <= retention < 1.0:
            raise ValueError("retention must be in [0, 1).")
        if not 0.0 <= initial_trust <= 1.0:
            raise ValueError("initial_trust must be in [0, 1].")

        self.retention = float(retention)
        self.initial_trust = float(initial_trust)
        self._history = {client_id: self.initial_trust for client_id in client_ids}

    def get_scores(self, client_ids: Optional[Iterable] = None):
        if client_ids is None:
            return dict(self._history)
        return {
            client_id: self._history.get(client_id, self.initial_trust)
            for client_id in client_ids
        }

    def update(self, current_trust_scores):
        for client_id, current_score in current_trust_scores.items():
            score = float(current_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Invalid trust score for client {client_id}.")

            previous = self._history.get(client_id, self.initial_trust)
            self._history[client_id] = (
                self.retention * previous + (1.0 - self.retention) * score
            )


# ============================================================
# 6. COMBINED TRUST SCORE
# ============================================================

def calculate_combined_trust(
    similarity_scores,
    consistency_scores,
    historical_scores,
    alpha=1.0 / 3.0,
    beta=1.0 / 3.0,
    gamma=1.0 / 3.0,
):
    """Combine the three trust signals: T_i = alpha*S_i + beta*C_i + gamma*H_i."""
    weights = [float(alpha), float(beta), float(gamma)]

    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("Trust weights must be finite and non-negative.")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("Trust weights must sum to 1.")

    client_ids = set(similarity_scores)
    if set(consistency_scores) != client_ids or set(historical_scores) != client_ids:
        raise ValueError("Client IDs differ between trust signals.")

    combined_scores = {}
    for client_id in client_ids:
        values = [
            float(similarity_scores[client_id]),
            float(consistency_scores[client_id]),
            float(historical_scores[client_id]),
        ]

        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Invalid trust component for client {client_id}.")

        combined = alpha * values[0] + beta * values[1] + gamma * values[2]
        combined_scores[client_id] = float(np.clip(combined, 0.0, 1.0))

    return combined_scores


# ============================================================
# 7. TRUST-BASED CLIENT SCREENING
# ============================================================

def screen_clients(trust_scores, threshold=0.50):
    """Accept clients with trust >= threshold; return accepted and excluded IDs."""
    if not math.isfinite(float(threshold)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    accepted = []
    excluded = []

    for client_id, score in trust_scores.items():
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid combined trust score for client {client_id}.")

        if score >= threshold:
            accepted.append(client_id)
        else:
            excluded.append(client_id)

    return accepted, excluded


# ============================================================
# 8. SYNTHETIC TESTS
# ============================================================

def run_smoke_tests():
    print("=" * 68)
    print("TRFA TRUST METRICS V2 SMOKE TEST")
    print("=" * 68)

    # Existing similarity behavior.
    updates_round_1 = {
        1: torch.tensor([0.10, 0.20, 0.30]),
        2: torch.tensor([0.11, 0.19, 0.31]),
        3: torch.tensor([-0.10, -0.20, -0.30]),
    }
    similarity = calculate_update_similarity(updates_round_1)
    assert set(similarity) == {1, 2, 3}
    assert all(0.0 <= score <= 1.0 for score in similarity.values())

    # New temporal consistency: first observation is neutral.
    consistency_tracker = TrainingConsistencyTracker(client_ids=[1, 2, 3])
    first_scores = consistency_tracker.score(updates_round_1)
    assert all(abs(score - 0.5) < 1e-12 for score in first_scores.values())
    consistency_tracker.update(updates_round_1)

    # Same direction and same magnitude should score higher than a reversal.
    updates_round_2 = {
        1: torch.tensor([0.20, 0.40, 0.60]),       # same direction, 2x norm
        2: torch.tensor([0.11, 0.19, 0.31]),       # same direction, same norm
        3: torch.tensor([0.10, 0.20, 0.30]),        # reversed direction
    }
    second_scores = consistency_tracker.score(updates_round_2)
    assert second_scores[2] > second_scores[1]
    assert second_scores[2] > second_scores[3]
    consistency_tracker.update(updates_round_2)

    # Backward compatibility for historical scripts.
    legacy = calculate_training_consistency(
        1.0, {1: 0.90, 2: 1.00, 3: 1.10}
    )
    assert legacy[1] > 0.5 and legacy[2] == 0.5 and legacy[3] < 0.5

    history = HistoricalTrustTracker([1, 2, 3], retention=0.90, initial_trust=0.50)
    combined = calculate_combined_trust(
        similarity,
        second_scores,
        history.get_scores(),
    )
    accepted, excluded = screen_clients(combined, threshold=0.50)
    assert set(accepted).isdisjoint(excluded)
    assert set(accepted + excluded) == {1, 2, 3}

    # Verify save/restore of temporal tracker state.
    restored = TrainingConsistencyTracker(client_ids=[1, 2, 3])
    restored.load_state_dict(consistency_tracker.state_dict())
    restored_scores = restored.score(updates_round_2)
    original_scores = consistency_tracker.score(updates_round_2)
    for client_id in original_scores:
        assert abs(restored_scores[client_id] - original_scores[client_id]) < 1e-12

    print("Round-1 consistency (neutral):", first_scores)
    print("Round-2 temporal consistency:", second_scores)
    print("Legacy validation score (compatibility):", legacy)
    print("Smoke tests passed.")


if __name__ == "__main__":
    run_smoke_tests()
