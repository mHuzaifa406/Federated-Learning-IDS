"""TRFA Update Similarity V2.

This module reports complementary diagnostics for each client update:
- cosine similarity to a coordinate-wise median reference;
- median pairwise cosine similarity to the other clients;
- update L2 norm and ratio to the round median norm;
- magnitude agreement with the round median norm;
- distance to the coordinate-wise median reference and relative distance.

These values are diagnostics, not calibrated trust probabilities. Do not use
any one value as an automatic malicious-client decision without validation on
clean non-IID and poisoned experiments.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

EPSILON = 1e-12


def _to_vector(update: torch.Tensor, client_id: Any) -> torch.Tensor:
    if not isinstance(update, torch.Tensor):
        raise TypeError(f"Update for client {client_id} must be a torch.Tensor.")
    vector = update.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if vector.numel() == 0:
        raise ValueError(f"Update for client {client_id} is empty.")
    if not torch.isfinite(vector).all().item():
        raise ValueError(f"Update for client {client_id} contains non-finite values.")
    return vector


def calculate_reference_update_v2(client_updates: Mapping[Any, torch.Tensor]) -> torch.Tensor:
    """Return the coordinate-wise median of all flattened client updates."""
    if not isinstance(client_updates, Mapping) or len(client_updates) < 2:
        raise ValueError("At least two client updates are required.")
    vectors = [_to_vector(update, cid) for cid, update in client_updates.items()]
    dimension = vectors[0].numel()
    if any(vector.numel() != dimension for vector in vectors):
        raise ValueError("Client update vectors have different dimensions.")
    return torch.quantile(torch.stack(vectors, dim=0), 0.5, dim=0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = torch.linalg.vector_norm(a).item()
    nb = torch.linalg.vector_norm(b).item()
    if na <= EPSILON and nb <= EPSILON:
        return 1.0
    if na <= EPSILON or nb <= EPSILON:
        return 0.0
    value = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1, eps=EPSILON).item()
    return float(np.clip(value, -1.0, 1.0))


def _scaled_cosine(value: float) -> float:
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


def _relative_agreement(value: float, reference: float) -> tuple[float, float]:
    """Return (value/reference ratio, symmetric [0,1] agreement score)."""
    if value <= EPSILON and reference <= EPSILON:
        return 1.0, 1.0
    if value <= EPSILON or reference <= EPSILON:
        return (0.0 if reference > EPSILON else min(value / EPSILON, 1e12)), 0.0
    ratio = value / reference
    score = math.exp(-abs(math.log(ratio)))
    return float(ratio), float(np.clip(score, 0.0, 1.0))


def calculate_update_similarity_v2(
    client_updates: Mapping[Any, torch.Tensor],
    reference_update: Optional[torch.Tensor] = None,
) -> Dict[Any, Dict[str, float]]:
    """Calculate per-client update similarity and magnitude diagnostics.

    No final combined trust score is returned intentionally. The diagnostic
    dimensions should first be checked for false penalties on honest non-IID
    clients and for separation under multiple poisoning settings.
    """
    if not isinstance(client_updates, Mapping) or len(client_updates) < 2:
        raise ValueError("At least two client updates are required.")

    vectors = {cid: _to_vector(update, cid) for cid, update in client_updates.items()}
    dimensions = {vector.numel() for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError("Client update vectors have different dimensions.")

    if reference_update is None:
        reference = calculate_reference_update_v2(vectors)
    else:
        reference = _to_vector(reference_update, "reference")
        if reference.numel() != next(iter(dimensions)):
            raise ValueError("Reference update dimension does not match client updates.")

    norms = {cid: float(torch.linalg.vector_norm(vec).item()) for cid, vec in vectors.items()}
    median_norm = float(np.median(list(norms.values())))

    distances = {
        cid: float(torch.linalg.vector_norm(vec - reference).item())
        for cid, vec in vectors.items()
    }
    median_distance = float(np.median(list(distances.values())))

    results: Dict[Any, Dict[str, float]] = {}
    ids = list(vectors)
    for cid, vector in vectors.items():
        cosine_to_reference = _cosine(vector, reference)
        pairwise_cosines = [
            _cosine(vector, vectors[other_id])
            for other_id in ids
            if other_id != cid
        ]
        median_pairwise_cosine = float(np.median(pairwise_cosines))
        magnitude_ratio, magnitude_agreement = _relative_agreement(norms[cid], median_norm)
        distance_ratio, distance_agreement = _relative_agreement(distances[cid], median_distance)

        results[cid] = {
            "cosine_to_median_reference": float(cosine_to_reference),
            "direction_score_to_reference": _scaled_cosine(cosine_to_reference),
            "median_pairwise_cosine": median_pairwise_cosine,
            "pairwise_direction_score": _scaled_cosine(median_pairwise_cosine),
            "update_l2_norm": norms[cid],
            "round_median_update_l2_norm": median_norm,
            "relative_magnitude": magnitude_ratio,
            "magnitude_agreement_score": magnitude_agreement,
            "distance_to_median_reference": distances[cid],
            "round_median_reference_distance": median_distance,
            "relative_reference_distance": distance_ratio,
            "reference_distance_agreement_score": distance_agreement,
        }
    return results
