"""Historical trust V2 for TRFA.

This module separates current-round evidence from the historical trust value.
It intentionally does NOT update history from the combined trust score.

Definitions
-----------
Current-round evidence (from validated non-historical signals):
    E_i^r = (alpha*S_i^r + beta*C_i^r) / (alpha + beta)

Historical update, performed after the round's trust decision:
    H_i^r = rho*H_i^(r-1) + (1-rho)*E_i^r

Combined trust for the current round:
    T_i^r = alpha*S_i^r + beta*C_i^r + gamma*H_i^(r-1)

All inputs must be finite and in [0, 1]. The three final trust weights must
be non-negative and sum to one. This module is an isolated implementation;
it does not change the existing trust_metrics.py or any experiment scripts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Hashable
import math


DEFAULT_ALPHA = 1.0 / 3.0
DEFAULT_BETA = 1.0 / 3.0
DEFAULT_GAMMA = 1.0 / 3.0
DEFAULT_RETENTION = 0.90
DEFAULT_INITIAL_TRUST = 0.50


def _validate_score_map(scores: Mapping, name: str) -> dict:
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError(f"{name} must be a non-empty mapping.")
    result = {}
    for client_id, raw_score in scores.items():
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name}[{client_id!r}] must be numeric.") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{name}[{client_id!r}] must be finite and in [0, 1].")
        result[client_id] = score
    return result


def _validate_weights(alpha: float, beta: float, gamma: float | None = None) -> tuple[float, ...]:
    values = [float(alpha), float(beta)]
    if gamma is not None:
        values.append(float(gamma))
    if any(not math.isfinite(x) or x < 0.0 for x in values):
        raise ValueError("Trust weights must be finite and non-negative.")
    if gamma is None:
        if sum(values) <= 0.0:
            raise ValueError("At least one current-evidence weight must be positive.")
    elif not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("alpha + beta + gamma must equal 1.")
    return tuple(values)


def calculate_current_evidence(
    similarity_scores: Mapping,
    consistency_scores: Mapping,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> dict:
    """Blend only current-round signals for the historical EMA update.

    The current evidence weights are normalized by alpha + beta. Historical
    trust is deliberately excluded, preventing the old recursive feedback.
    """
    alpha, beta = _validate_weights(alpha, beta)
    similarity = _validate_score_map(similarity_scores, "similarity_scores")
    consistency = _validate_score_map(consistency_scores, "consistency_scores")
    if set(similarity) != set(consistency):
        raise ValueError("Client IDs differ between current-round signals.")

    denominator = alpha + beta
    return {
        client_id: (alpha * similarity[client_id] + beta * consistency[client_id]) / denominator
        for client_id in similarity
    }


def calculate_combined_trust_v2(
    similarity_scores: Mapping,
    consistency_scores: Mapping,
    previous_historical_scores: Mapping,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> dict:
    """Calculate current-round trust using history from the prior round."""
    alpha, beta, gamma = _validate_weights(alpha, beta, gamma)
    similarity = _validate_score_map(similarity_scores, "similarity_scores")
    consistency = _validate_score_map(consistency_scores, "consistency_scores")
    historical = _validate_score_map(previous_historical_scores, "previous_historical_scores")
    client_ids = set(similarity)
    if set(consistency) != client_ids or set(historical) != client_ids:
        raise ValueError("Client IDs differ between trust components.")

    return {
        client_id: (
            alpha * similarity[client_id]
            + beta * consistency[client_id]
            + gamma * historical[client_id]
        )
        for client_id in similarity
    }


class HistoricalTrustTrackerV2:
    """Maintains historical trust using independent current-round evidence.

    Call order per communication round:
      1. Read H^(r-1) with get_scores().
      2. Calculate the current-round combined trust with
         calculate_combined_trust_v2().
      3. Apply screening and aggregation.
      4. Calculate E^r from current signals only.
      5. Call update(E^r, round_number=r) exactly once.

    Do not pass combined trust scores to update().
    """

    VERSION = "2.0"

    def __init__(
        self,
        client_ids: Iterable[Hashable],
        retention: float = DEFAULT_RETENTION,
        initial_trust: float = DEFAULT_INITIAL_TRUST,
    ) -> None:
        ids = list(client_ids)
        if not ids:
            raise ValueError("client_ids must contain at least one client.")
        if len(ids) != len(set(ids)):
            raise ValueError("client_ids must be unique.")
        if not math.isfinite(float(retention)) or not 0.0 <= float(retention) < 1.0:
            raise ValueError("retention must be in [0, 1).")
        if not math.isfinite(float(initial_trust)) or not 0.0 <= float(initial_trust) <= 1.0:
            raise ValueError("initial_trust must be in [0, 1].")

        self.client_ids = tuple(ids)
        self.retention = float(retention)
        self.initial_trust = float(initial_trust)
        self._history = {client_id: self.initial_trust for client_id in self.client_ids}
        self.last_updated_round = 0
        self._update_log: list[dict] = []

    def get_scores(self, client_ids: Iterable[Hashable] | None = None) -> dict:
        requested = self.client_ids if client_ids is None else tuple(client_ids)
        unknown = set(requested) - set(self._history)
        if unknown:
            raise KeyError(f"Unknown client IDs requested: {sorted(map(str, unknown))}")
        return {client_id: self._history[client_id] for client_id in requested}

    def update(self, current_evidence_scores: Mapping, *, round_number: int | None = None) -> dict:
        """Update history from current-round evidence, never from combined trust."""
        evidence = _validate_score_map(current_evidence_scores, "current_evidence_scores")
        if set(evidence) != set(self._history):
            missing = set(self._history) - set(evidence)
            extra = set(evidence) - set(self._history)
            raise ValueError(
                f"Evidence client IDs must exactly match tracker clients; "
                f"missing={sorted(map(str, missing))}, extra={sorted(map(str, extra))}."
            )

        next_round = self.last_updated_round + 1 if round_number is None else round_number
        if isinstance(next_round, bool) or not isinstance(next_round, int) or next_round < 1:
            raise ValueError("round_number must be a positive integer.")
        if next_round != self.last_updated_round + 1:
            raise ValueError(
                f"Expected round {self.last_updated_round + 1}, received {next_round}. "
                "History must be updated exactly once per consecutive round."
            )

        # Validate everything before mutating tracker state (atomic update).
        new_history = {
            client_id: (
                self.retention * self._history[client_id]
                + (1.0 - self.retention) * evidence[client_id]
            )
            for client_id in self.client_ids
        }
        self._history = new_history
        self.last_updated_round = next_round
        self._update_log.append({"round": next_round, "evidence": dict(evidence), "history": dict(new_history)})
        return dict(self._history)

    def state_dict(self) -> dict:
        """Return a serializable checkpoint of tracker state."""
        return {
            "version": self.VERSION,
            "client_ids": list(self.client_ids),
            "retention": self.retention,
            "initial_trust": self.initial_trust,
            "last_updated_round": self.last_updated_round,
            "history": dict(self._history),
            "update_log": list(self._update_log),
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore tracker state after validating configuration and client IDs."""
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping.")
        if state.get("version") != self.VERSION:
            raise ValueError(f"Unsupported historical tracker version: {state.get('version')!r}")
        if tuple(state.get("client_ids", ())) != self.client_ids:
            raise ValueError("Checkpoint client IDs do not match tracker client IDs/order.")
        if not math.isclose(float(state.get("retention")), self.retention, abs_tol=1e-12):
            raise ValueError("Checkpoint retention does not match tracker configuration.")
        if not math.isclose(float(state.get("initial_trust")), self.initial_trust, abs_tol=1e-12):
            raise ValueError("Checkpoint initial_trust does not match tracker configuration.")
        history = _validate_score_map(state.get("history", {}), "checkpoint.history")
        if set(history) != set(self._history):
            raise ValueError("Checkpoint history client IDs do not match tracker clients.")
        last_round = state.get("last_updated_round")
        if isinstance(last_round, bool) or not isinstance(last_round, int) or last_round < 0:
            raise ValueError("Checkpoint last_updated_round must be a non-negative integer.")
        self._history = history
        self.last_updated_round = last_round
        self._update_log = list(state.get("update_log", []))

    @property
    def update_log(self) -> list[dict]:
        return list(self._update_log)
