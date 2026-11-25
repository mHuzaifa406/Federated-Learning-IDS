"""Clean-reference calibration for the Phase 9 CCTRT detector.

The calibrated statistic is the round/client-specific change in
Cumulative_Mean_Abs_Step_Combined_Trust. It is scored against distributions
learned exclusively from independent all-honest Phase 9 control runs.

This is a detector score, not a probability of maliciousness.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

FEATURE = "Cumulative_Mean_Abs_Step_Combined_Trust"
FORMAT_VERSION = 1


def _read_clean_run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    metadata_path = run_dir / "phase9_metadata.json"
    scores_path = run_dir / "phase9_client_round_scores.csv"
    if not metadata_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError(
            f"{run_dir} must contain phase9_metadata.json and "
            "phase9_client_round_scores.csv"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "all-honest" not in str(metadata.get("phase", "")).lower():
        raise ValueError(f"{run_dir} is not labelled as an all-honest control.")
    if metadata.get("malicious_client") is not None:
        raise ValueError(f"{run_dir} declares a malicious client; calibration must be clean.")
    if metadata.get("attack_active") is not False:
        raise ValueError(f"{run_dir} metadata does not confirm attack_active=false.")
    if metadata.get("screening_enabled") is not False:
        raise ValueError(f"{run_dir} must be an unscreened all-honest calibration run.")
    if metadata.get("all_clients_aggregated") is not True:
        raise ValueError(f"{run_dir} must aggregate all clients during calibration.")

    frame = pd.read_csv(scores_path)
    required = {"Round", "Client", FEATURE, "Attack_Active"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{scores_path} is missing columns: {sorted(missing)}")
    if frame["Attack_Active"].astype(str).str.lower().isin({"true", "1"}).any():
        raise ValueError(f"{scores_path} contains attack-active rows.")
    values = pd.to_numeric(frame[FEATURE], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{scores_path} contains missing or non-finite {FEATURE} values.")

    return metadata, frame


def build_calibration(
    clean_run_dirs: Iterable[str | Path],
    *,
    quantile: float = 0.975,
) -> dict[str, Any]:
    """Build a per-client/per-round clean reference from >=3 distinct seeds."""
    dirs = [Path(p).expanduser().resolve() for p in clean_run_dirs]
    if len(dirs) < 3:
        raise ValueError("At least three independent all-honest calibration runs are required.")
    if len(set(dirs)) != len(dirs):
        raise ValueError("Duplicate calibration directories were supplied.")
    if not math.isfinite(float(quantile)) or not 0.90 <= float(quantile) < 1.0:
        raise ValueError("quantile must be in [0.90, 1.0).")

    loaded = [_read_clean_run(p) for p in dirs]
    metadata = [item[0] for item in loaded]
    frames = [item[1] for item in loaded]

    seeds = [int(m["training_seed"]) for m in metadata]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Calibration training seeds must be distinct; found {seeds}.")

    partition_seeds = {int(m["partition_seed"]) for m in metadata}
    alphas = {round(float(m["dirichlet_alpha"]), 10) for m in metadata}
    round_counts = {int(m["rounds"]) for m in metadata}
    client_counts = {int(m["clients"]) for m in metadata}
    if len(partition_seeds) != 1 or len(alphas) != 1 or len(round_counts) != 1 or len(client_counts) != 1:
        raise ValueError("All clean calibration runs must use the same partition seed, alpha, rounds and clients.")

    rounds = next(iter(round_counts))
    clients = next(iter(client_counts))
    expected = {(r, c) for r in range(1, rounds + 1) for c in range(1, clients + 1)}
    matrices: list[dict[tuple[int, int], float]] = []

    for run_dir, frame in zip(dirs, frames):
        frame = frame.copy()
        frame["Round"] = pd.to_numeric(frame["Round"], errors="raise").astype(int)
        frame["Client"] = pd.to_numeric(frame["Client"], errors="raise").astype(int)
        if frame.duplicated(["Round", "Client"]).any():
            raise ValueError(f"{run_dir} has duplicate Round/Client rows.")
        keyed = {
            (int(row.Round), int(row.Client)): float(getattr(row, FEATURE))
            for row in frame[["Round", "Client", FEATURE]].itertuples(index=False)
        }
        if set(keyed) != expected:
            raise ValueError(f"{run_dir} does not contain exactly {rounds} rounds x {clients} clients.")
        matrices.append(keyed)

    profile: dict[str, dict[str, float | int]] = {}
    residuals: list[float] = []
    # Leave-one-run-out residuals prevent a clean run from calibrating
    # against a reference that includes its own observation.
    for key in sorted(expected):
        sample = np.asarray([m[key] for m in matrices], dtype=float)
        center = float(np.median(sample))
        profile[f"{key[0]}:{key[1]}"] = {"center": center, "n": int(len(sample))}
        for run_index, observation in enumerate(sample):
            reference_sample = np.delete(sample, run_index)
            residuals.append(float(observation - np.median(reference_sample)))

    residual_array = np.asarray(residuals, dtype=float)
    residual_median = float(np.median(residual_array))
    robust_scale = float(1.4826 * np.median(np.abs(residual_array - residual_median)))
    if robust_scale <= 0.0:
        robust_scale = float(np.std(residual_array, ddof=1))
    if not math.isfinite(robust_scale) or robust_scale <= 0.0:
        raise ValueError(
            "Clean calibration runs show no measurable variation in the selected signal; "
            "a defensible normalization scale cannot be estimated."
        )

    standardized_clean_residuals = residual_array / robust_scale
    threshold_z = float(np.quantile(standardized_clean_residuals, float(quantile)))

    return {
        "format_version": FORMAT_VERSION,
        "method": "Clean-reference calibrated CCTRT",
        "feature": FEATURE,
        "direction": "higher_is_suspicious",
        "score_definition": "(observed - clean_reference_median) / pooled_clean_residual_scale",
        "threshold_z": threshold_z,
        "quantile": float(quantile),
        "pooled_clean_residual_scale": robust_scale,
        "partition_seed": next(iter(partition_seeds)),
        "dirichlet_alpha": next(iter(alphas)),
        "rounds": rounds,
        "clients": clients,
        "training_seeds": seeds,
        "calibration_run_count": len(dirs),
        "calibration_directories": [str(p) for p in dirs],
        "profile": profile,
        "limitations": [
            "Thresholds are estimated from clean calibration trajectories only.",
            "This calibration is specific to the recorded partition, feature pipeline and training configuration.",
            "The anomaly score is not a calibrated probability of maliciousness.",
            "Independent clean-control and attack runs are still required to estimate generalization and false-positive rates.",
        ],
    }


def save_calibration(calibration: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    return path


def load_calibration(
    calibration_path: str | Path,
    *,
    expected_partition_seed: int | None = None,
    expected_alpha: float | None = None,
    expected_rounds: int | None = None,
    expected_clients: int | None = None,
) -> dict[str, Any]:
    path = Path(calibration_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("format_version") != FORMAT_VERSION or model.get("feature") != FEATURE:
        raise ValueError("Unsupported or incompatible CCTRT calibration file.")
    if expected_partition_seed is not None and int(model["partition_seed"]) != int(expected_partition_seed):
        raise ValueError("Calibration partition seed does not match the evaluation partition.")
    if expected_alpha is not None and not math.isclose(float(model["dirichlet_alpha"]), float(expected_alpha), abs_tol=1e-9):
        raise ValueError("Calibration Dirichlet alpha does not match the evaluation partition.")
    if expected_rounds is not None and int(model["rounds"]) != int(expected_rounds):
        raise ValueError("Calibration round count does not match the evaluation run.")
    if expected_clients is not None and int(model["clients"]) != int(expected_clients):
        raise ValueError("Calibration client count does not match the evaluation run.")
    if not math.isfinite(float(model["pooled_clean_residual_scale"])) or float(model["pooled_clean_residual_scale"]) <= 0:
        raise ValueError("Calibration scale must be positive and finite.")
    return model


def score_calibrated_round(
    round_features: dict[int, dict[str, float]],
    round_number: int,
    calibration: dict[str, Any],
) -> dict[int, dict[str, float | bool]]:
    """Calculate per-client standardized deviations for one completed round."""
    clients = int(calibration["clients"])
    if set(round_features) != set(range(1, clients + 1)):
        raise ValueError("Round features must contain every calibrated client exactly once.")
    if not 1 <= int(round_number) <= int(calibration["rounds"]):
        raise ValueError(f"Round {round_number} is outside the calibrated range.")

    feature = str(calibration["feature"])
    scale = float(calibration["pooled_clean_residual_scale"])
    threshold = float(calibration["threshold_z"])
    result: dict[int, dict[str, float | bool]] = {}
    for client_id in range(1, clients + 1):
        key = f"{int(round_number)}:{client_id}"
        if key not in calibration["profile"]:
            raise ValueError(f"Calibration profile is missing {key}.")
        observed = float(round_features[client_id][feature])
        center = float(calibration["profile"][key]["center"])
        if not math.isfinite(observed):
            raise ValueError(f"Client {client_id} has a non-finite calibrated feature.")
        z_score = (observed - center) / scale
        result[client_id] = {
            "Observed_Feature": observed,
            "Clean_Reference_Median": center,
            "Calibrated_Anomaly_Z": float(z_score),
            "Calibrated_Alert": bool(z_score > threshold),
            "Calibration_Threshold_Z": threshold,
        }
    return result


def apply_calibration_to_frame(
    frame: pd.DataFrame,
    calibration: dict[str, Any],
    *,
    persistence: int = 2,
) -> pd.DataFrame:
    """Score an existing Phase 9 file and apply consecutive-round persistence."""
    if isinstance(persistence, bool) or int(persistence) < 1:
        raise ValueError("persistence must be a positive integer.")
    required = {"Round", "Client", calibration["feature"]}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Score CSV is missing columns: {sorted(missing)}")

    result = frame.copy()
    result["Round"] = pd.to_numeric(result["Round"], errors="raise").astype(int)
    result["Client"] = pd.to_numeric(result["Client"], errors="raise").astype(int)
    result["Calibrated_Anomaly_Z"] = np.nan
    result["Calibrated_Alert"] = False
    result["Consecutive_Alert_Rounds"] = 0
    result["Rejected_By_Calibrated_TRFA"] = False

    streaks = {cid: 0 for cid in range(1, int(calibration["clients"]) + 1)}
    for round_number, group in result.groupby("Round", sort=True):
        features = {
            int(row.Client): {calibration["feature"]: float(getattr(row, calibration["feature"]))}
            for row in group[["Client", calibration["feature"]]].itertuples(index=False)
        }
        scored = score_calibrated_round(features, int(round_number), calibration)
        for idx in group.index:
            cid = int(result.at[idx, "Client"])
            item = scored[cid]
            alert = bool(item["Calibrated_Alert"])
            streaks[cid] = streaks[cid] + 1 if alert else 0
            result.at[idx, "Calibrated_Anomaly_Z"] = float(item["Calibrated_Anomaly_Z"])
            result.at[idx, "Calibrated_Alert"] = alert
            result.at[idx, "Consecutive_Alert_Rounds"] = streaks[cid]
            result.at[idx, "Rejected_By_Calibrated_TRFA"] = streaks[cid] >= int(persistence)

    return result.sort_values(["Round", "Client"]).reset_index(drop=True)
