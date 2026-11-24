"""TRFA Trust Fusion V1: offline, non-screening diagnostic.

This script combines client-level summaries from CAVC V2, Historical Trust V2,
and Similarity V2. It never changes training, aggregation, screening, or model
checkpoints. Higher fusion scores mean "more unusual / more suspicious" only
for exploratory ranking; they are NOT calibrated probabilities of attack.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MALICIOUS_CLIENT_ID = 7
EXPECTED_CLIENTS = set(range(1, 11))
CONDITIONS = ("clean", "poison-client07")
DEFAULT_WEIGHTS = {"similarity": 1.0 / 3.0, "historical": 1.0 / 3.0, "cavc": 1.0 / 3.0}


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _read_csv(path: Path, required: Iterable[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file was not found: {path}\n"
            "Check the input folder or pass the correct CSV path to the command."
        )
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} CSV is missing required columns: {missing}\nFound: {list(frame.columns)}")
    if frame.empty:
        raise ValueError(f"{label} CSV is empty: {path}")
    return frame



def _bool_values(values: pd.Series, label: str) -> pd.Series:
    """Parse CSV boolean columns safely (including textual True/False)."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        unique = set(pd.to_numeric(values, errors="raise").dropna().unique().tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"{label} must contain only 0/1 or True/False values.")
        return values.astype(int).astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    parsed = values.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        bad = sorted(values[parsed.isna()].astype(str).unique().tolist())
        raise ValueError(f"{label} contains unrecognised boolean values: {bad}")
    return parsed.astype(bool)


def _normalise_condition(value: object) -> str:
    token = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if token in {"clean", "benign"}:
        return "clean"
    if token in {"poison-client07", "poisoned-client07", "poison-client-07", "poisoned"}:
        return "poison-client07"
    raise ValueError(f"Unrecognised CAVC condition '{value}'. Expected clean or poison_client07.")


def _validate_client_ids(frame: pd.DataFrame, column: str, label: str) -> None:
    ids = set(pd.to_numeric(frame[column], errors="raise").astype(int).unique())
    if ids != EXPECTED_CLIENTS:
        raise ValueError(f"{label} must contain clients 1–10 exactly. Found: {sorted(ids)}")


def _aggregate_round_file(path: Path, condition: str, kind: str) -> pd.DataFrame:
    if kind == "historical":
        required = ["Round", "Client", "Is_Malicious", "Combined_Trust", "Update_Similarity", "Temporal_Consistency", "Historical_Trust_After"]
    else:
        required = ["Round", "Client", "Is_Malicious", "pairwise_direction_score", "direction_score_to_reference", "magnitude_agreement_score", "relative_magnitude", "reference_distance_agreement_score", "relative_reference_distance"]
    frame = _read_csv(path, required, f"{condition} {kind}")
    frame["Client"] = pd.to_numeric(frame["Client"], errors="raise").astype(int)
    frame["Round"] = pd.to_numeric(frame["Round"], errors="raise").astype(int)
    _validate_client_ids(frame, "Client", f"{condition} {kind}")
    if frame.duplicated(["Round", "Client"]).any():
        raise ValueError(f"{condition} {kind} has duplicate (Round, Client) records.")
    rounds = sorted(frame["Round"].unique().tolist())
    if rounds != list(range(1, 11)):
        raise ValueError(f"{condition} {kind} must contain rounds 1–10. Found: {rounds}")
    counts = frame.groupby("Client")["Round"].nunique()
    if not (counts == 10).all():
        raise ValueError(f"{condition} {kind} does not have 10 rounds for every client.")

    malicious_by_client = frame.groupby("Client")["Is_Malicious"].nunique()
    if (malicious_by_client > 1).any():
        raise ValueError(f"{condition} {kind} has inconsistent Is_Malicious values across rounds.")
    frame["Is_Malicious"] = _bool_values(frame["Is_Malicious"], f"{condition} {kind} Is_Malicious")
    flags = frame.groupby("Client")["Is_Malicious"].first().astype(bool)
    expected_malicious = {MALICIOUS_CLIENT_ID} if condition == "poison-client07" else set()
    observed_malicious = set(flags[flags].index.astype(int).tolist())
    if observed_malicious != expected_malicious:
        raise ValueError(
            f"{condition} {kind} malicious-client labels do not match the expected condition. "
            f"Expected {sorted(expected_malicious)}, observed {sorted(observed_malicious)}."
        )

    numeric_columns = [c for c in required if c not in {"Round", "Client", "Is_Malicious"}]
    for col in numeric_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{condition} {kind} contains non-finite values in {col}.")

    means = frame.groupby("Client", as_index=False)[numeric_columns].mean()
    means["Is_Malicious"] = means["Client"].eq(MALICIOUS_CLIENT_ID) if condition == "poison-client07" else False
    means["Condition"] = condition
    means["Rounds_Aggregated"] = 10
    return means


def _load_cavc(path: Path) -> pd.DataFrame:
    required = ["condition", "client_id", "is_malicious", "attack_detection_contribution_score", "targeted_damage_penalty", "benign_fpr_damage_penalty"]
    frame = _read_csv(path, required, "CAVC V2 client summary")
    frame["Condition"] = frame["condition"].map(_normalise_condition)
    frame["is_malicious"] = _bool_values(frame["is_malicious"], "CAVC is_malicious")
    frame["Client"] = pd.to_numeric(frame["client_id"], errors="raise").astype(int)
    _validate_client_ids(frame[frame["Condition"] == "clean"], "Client", "CAVC clean")
    _validate_client_ids(frame[frame["Condition"] == "poison-client07"], "Client", "CAVC poisoned")
    if "min_class_support" in frame.columns:
        supports = sorted(pd.to_numeric(frame["min_class_support"], errors="raise").astype(int).unique())
        if supports != [100]:
            raise ValueError(
                "This V1 implementation expects the CAVC V2 client summary at min_class_support=100. "
                f"Found support values: {supports}. Use the matching *_min_support_100.csv file."
            )
    if frame.duplicated(["Condition", "Client"]).any():
        raise ValueError("CAVC V2 summary must have exactly one row per client and condition.")
    for condition in CONDITIONS:
        subset = frame[frame["Condition"] == condition]
        expected = {MALICIOUS_CLIENT_ID} if condition == "poison-client07" else set()
        observed = set(subset.loc[subset["is_malicious"], "Client"].astype(int).tolist())
        if observed != expected:
            raise ValueError(f"CAVC {condition} malicious-client labels mismatch: expected {sorted(expected)}, observed {sorted(observed)}")
    for col in required[3:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f"CAVC V2 contains non-finite values in {col}.")
    return frame[["Condition", "Client", "attack_detection_contribution_score", "targeted_damage_penalty", "benign_fpr_damage_penalty"]].copy()


def suspicion_percentile(values: pd.Series, *, higher_is_suspicious: bool) -> pd.Series:
    """Cross-sectional percentile in [0,1]; 1 is most suspicious."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Cannot rank missing or non-finite diagnostic values.")
    # With ascending=True, the largest gets percentile 1.0 (high-is-suspicious).
    # With ascending=False, the smallest gets percentile 1.0 (low-is-suspicious).
    return numeric.rank(method="average", ascending=higher_is_suspicious, pct=True)


def _add_component_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    # Similarity V2: low agreement / direction is unusual; ratios far from 1 are unusual.
    similarity_specs = {
        "sim_pairwise_direction_suspicion": ("pairwise_direction_score", False),
        "sim_reference_direction_suspicion": ("direction_score_to_reference", False),
        "sim_magnitude_suspicion": ("magnitude_agreement_score", False),
        "sim_reference_distance_suspicion": ("reference_distance_agreement_score", False),
    }
    for destination, (source, direction) in similarity_specs.items():
        out[destination] = out.groupby("Condition")[source].transform(
            lambda s: suspicion_percentile(s, higher_is_suspicious=direction)
        )
    out["sim_relative_magnitude_deviation"] = np.abs(np.log(np.clip(out["relative_magnitude"].astype(float), 1e-12, None)))
    out["sim_relative_magnitude_suspicion"] = out.groupby("Condition")["sim_relative_magnitude_deviation"].transform(
        lambda s: suspicion_percentile(s, higher_is_suspicious=True)
    )
    out["sim_relative_reference_distance_deviation"] = np.abs(np.log(np.clip(out["relative_reference_distance"].astype(float), 1e-12, None)))
    out["sim_relative_reference_distance_suspicion"] = out.groupby("Condition")["sim_relative_reference_distance_deviation"].transform(
        lambda s: suspicion_percentile(s, higher_is_suspicious=True)
    )
    sim_cols = [c for c in out.columns if c.startswith("sim_") and c.endswith("_suspicion")]
    out["Similarity_Group_Score"] = out[sim_cols].mean(axis=1)

    # Historical V2 Combined_Trust already includes similarity, consistency and prior history.
    # Do not count its sub-signals a second time in this group.
    out["Historical_Group_Score"] = out.groupby("Condition")["Combined_Trust"].transform(
        lambda s: suspicion_percentile(s, higher_is_suspicious=False)
    )

    # CAVC V2 components stay separate at source; percentile scaling only makes their
    # directions comparable for this exploratory ranking. It does not make them honesty labels.
    out["cavc_attack_recall_concern"] = out.groupby("Condition")["attack_detection_contribution_score"].transform(
        lambda s: suspicion_percentile(s, higher_is_suspicious=False)
    )
    out["cavc_targeted_damage_concern"] = out.groupby("Condition")["targeted_damage_penalty"].transform(
        lambda s: suspicion_percentile(s, higher_is_suspicious=True)
    )
    out["cavc_benign_fpr_concern"] = out.groupby("Condition")["benign_fpr_damage_penalty"].transform(
        lambda s: suspicion_percentile(s, higher_is_suspicious=True)
    )
    out["CAVC_Group_Score"] = out[[
        "cavc_attack_recall_concern", "cavc_targeted_damage_concern", "cavc_benign_fpr_concern"
    ]].mean(axis=1)
    out["Equal_Weight_Fusion_Score"] = (
        out["Similarity_Group_Score"] + out["Historical_Group_Score"] + out["CAVC_Group_Score"]
    ) / 3.0
    return out


def _rank_poisoned_condition(frame: pd.DataFrame, score_column: str) -> dict:
    poison = frame[frame["Condition"] == "poison-client07"].copy()
    poison = poison.sort_values([score_column, "Client"], ascending=[False, True]).reset_index(drop=True)
    malicious_rows = poison[poison["Is_Malicious"].astype(bool)]
    if len(malicious_rows) != 1:
        raise ValueError("Expected exactly one malicious client in poisoned condition.")
    client_row = malicious_rows.iloc[0]
    honest = poison[~poison["Is_Malicious"].astype(bool)]
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        y = poison["Is_Malicious"].astype(int).to_numpy()
        scores = poison[score_column].to_numpy(dtype=float)
        auc = float(roc_auc_score(y, scores))
        ap = float(average_precision_score(y, scores))
    except Exception:
        auc = float("nan")
        ap = float("nan")
    return {
        "score": float(client_row[score_column]),
        "malicious_client_rank_most_suspicious_first": int(poison.index[poison["Client"] == MALICIOUS_CLIENT_ID][0]) + 1,
        "malicious_minus_honest_mean_score": float(client_row[score_column] - honest[score_column].mean()),
        "roc_auc_10_clients_one_attacker": auc,
        "average_precision_10_clients_one_attacker": ap,
    }


def _weight_grid(step: float = 0.1) -> list[dict[str, float]]:
    ticks = int(round(1.0 / step))
    rows: list[dict[str, float]] = []
    for s in range(ticks + 1):
        for h in range(ticks + 1 - s):
            c = ticks - s - h
            rows.append({"similarity_weight": s / ticks, "historical_weight": h / ticks, "cavc_weight": c / ticks})
    return rows


def run_fusion(args: argparse.Namespace) -> Path:
    root = Path(args.project_root).resolve()
    cavc_path = _resolve(root, args.cavc_csv)
    hist_clean_path = _resolve(root, args.historical_clean)
    hist_poison_path = _resolve(root, args.historical_poison)
    sim_clean_path = _resolve(root, args.similarity_clean)
    sim_poison_path = _resolve(root, args.similarity_poison)

    cavc = _load_cavc(cavc_path)
    all_frames = []
    for condition, hist_path, sim_path in [
        ("clean", hist_clean_path, sim_clean_path),
        ("poison-client07", hist_poison_path, sim_poison_path),
    ]:
        hist = _aggregate_round_file(hist_path, condition, "historical")
        sim = _aggregate_round_file(sim_path, condition, "similarity")
        hist = hist.drop(columns=["Condition", "Rounds_Aggregated"])
        sim = sim.drop(columns=["Condition", "Rounds_Aggregated"])
        merged = hist.merge(sim, on=["Client", "Is_Malicious"], how="inner", validate="one_to_one", suffixes=("_hist", "_sim"))
        if set(merged["Client"].astype(int)) != EXPECTED_CLIENTS:
            raise ValueError(f"Client IDs do not align across Historical and Similarity V2 for {condition}.")
        merged["Condition"] = condition
        all_frames.append(merged)
    dynamic = pd.concat(all_frames, ignore_index=True)
    combined = dynamic.merge(cavc, on=["Condition", "Client"], how="inner", validate="one_to_one")
    if len(combined) != 20:
        raise ValueError(f"Expected 20 client-condition records after merging; got {len(combined)}.")
    combined = _add_component_scores(combined)
    combined["Fusion_Rank_Most_Suspicious_First"] = combined.groupby("Condition")["Equal_Weight_Fusion_Score"].rank(
        method="average", ascending=False
    )

    # Paired change analysis: compare each client's relative suspicion percentile in poison vs clean.
    score_cols = ["Similarity_Group_Score", "Historical_Group_Score", "CAVC_Group_Score", "Equal_Weight_Fusion_Score"]
    clean = combined[combined["Condition"] == "clean"][["Client", *score_cols]].set_index("Client")
    poisoned = combined[combined["Condition"] == "poison-client07"][["Client", *score_cols]].set_index("Client")
    delta = (poisoned - clean).reset_index()
    delta = delta.rename(columns={c: f"Poison_Minus_Clean_{c}" for c in score_cols})
    delta["Change_Rank_Most_Increased_First"] = delta["Poison_Minus_Clean_Equal_Weight_Fusion_Score"].rank(method="average", ascending=False)
    combined = combined.merge(delta, on="Client", how="left", validate="many_to_one")

    default = DEFAULT_WEIGHTS
    combined["Weighted_Fusion_Score"] = (
        default["similarity"] * combined["Similarity_Group_Score"]
        + default["historical"] * combined["Historical_Group_Score"]
        + default["cavc"] * combined["CAVC_Group_Score"]
    )

    grid_rows = []
    for weights in _weight_grid(0.1):
        score_name = "_grid_score"
        temp = combined.copy()
        temp[score_name] = (
            weights["similarity_weight"] * temp["Similarity_Group_Score"]
            + weights["historical_weight"] * temp["Historical_Group_Score"]
            + weights["cavc_weight"] * temp["CAVC_Group_Score"]
        )
        metrics = _rank_poisoned_condition(temp.rename(columns={score_name: "_score"}), "_score")
        poison_scores = temp[temp["Condition"] == "poison-client07"].set_index("Client")[score_name]
        clean_scores = temp[temp["Condition"] == "clean"].set_index("Client")[score_name]
        change = poison_scores - clean_scores
        grid_rows.append({
            **weights,
            "client07_rank_most_suspicious_first": metrics["malicious_client_rank_most_suspicious_first"],
            "client07_minus_honest_mean": metrics["malicious_minus_honest_mean_score"],
            "poisoned_condition_roc_auc": metrics["roc_auc_10_clients_one_attacker"],
            "poisoned_condition_average_precision": metrics["average_precision_10_clients_one_attacker"],
            "client07_paired_change": float(change.loc[MALICIOUS_CLIENT_ID]),
            "client07_paired_change_rank": int(change.rank(method="average", ascending=False).loc[MALICIOUS_CLIENT_ID]),
        })
    grid = pd.DataFrame(grid_rows).sort_values(
        ["client07_rank_most_suspicious_first", "poisoned_condition_roc_auc", "client07_minus_honest_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    output_dir = root / "results" / f"trust_fusion_v1_diagnostic_{args.output_tag}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains results: {output_dir}\n"
            "Use a new --output-tag. Existing results will not be overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "trust_fusion_v1_client_scores.csv", index=False)
    delta.to_csv(output_dir / "trust_fusion_v1_paired_client_changes.csv", index=False)
    grid.to_csv(output_dir / "trust_fusion_v1_weight_sweep.csv", index=False)

    default_metrics = _rank_poisoned_condition(combined.rename(columns={"Equal_Weight_Fusion_Score": "_score"}), "_score")
    poison07 = combined[(combined["Condition"] == "poison-client07") & (combined["Client"] == MALICIOUS_CLIENT_ID)].iloc[0]
    clean07 = combined[(combined["Condition"] == "clean") & (combined["Client"] == MALICIOUS_CLIENT_ID)].iloc[0]
    best = grid.iloc[0].to_dict()
    report = [
        "TRFA TRUST FUSION V1 — OFFLINE DIAGNOSTIC",
        "===========================================",
        "",
        "Purpose: exploratory combination of existing CAVC V2, Historical Trust V2 and Similarity V2 diagnostics.",
        "No model training, aggregation, client screening or independent-test evaluation was performed.",
        "Higher values indicate greater diagnostic unusualness, not a calibrated attack probability.",
        "",
        "Equal-weight fusion (one-third per group):",
        f"  Client 07 poisoned-condition score: {default_metrics['score']:.6f}",
        f"  Client 07 rank (1 = most suspicious): {default_metrics['malicious_client_rank_most_suspicious_first']}/10",
        f"  Client 07 minus honest-client mean: {default_metrics['malicious_minus_honest_mean_score']:.6f}",
        f"  ROC-AUC across 10 clients (one attacker): {default_metrics['roc_auc_10_clients_one_attacker']:.6f}",
        f"  Average precision across 10 clients (one attacker): {default_metrics['average_precision_10_clients_one_attacker']:.6f}",
        "",
        "Client 07 paired change in equal-weight suspicion score:",
        f"  Clean: {float(clean07['Equal_Weight_Fusion_Score']):.6f}",
        f"  Poisoned: {float(poison07['Equal_Weight_Fusion_Score']):.6f}",
        f"  Poisoned minus clean: {float(poison07['Equal_Weight_Fusion_Score'] - clean07['Equal_Weight_Fusion_Score']):.6f}",
        "",
        "Top-ranked weight combination in this same-scenario exploratory 0.1-step grid (not a selected setting):",
        f"  Similarity={best['similarity_weight']:.1f}; Historical={best['historical_weight']:.1f}; CAVC={best['cavc_weight']:.1f}",
        f"  Client 07 rank: {int(best['client07_rank_most_suspicious_first'])}/10; ROC-AUC={best['poisoned_condition_roc_auc']:.6f}",
        "",
        "INTERPRETATION LIMITS",
        "- Only one malicious client, one poisoning configuration and one partition/training seed are represented.",
        "- Similarity and Historical Trust are averaged over 10 rounds; CAVC is a separate frozen-checkpoint candidate diagnostic.",
        "- Percentile scaling is relative to the 10 clients in each condition; scores are not probabilities and depend on this cohort.",
        "- The weight sweep is exploratory and evaluated on the same scenario used to inspect it; do not call its best weights optimal.",
        "- The 10-client ROC-AUC is descriptive and has only one positive client; it is not evidence of generalization.",
        "- Independent test data was not used. No screening or aggregation decision was made.",
        "",
        "Next validation required: repeat with additional malicious clients, attack strengths, independent seeds and a separately held-out development configuration before selecting weights or thresholds.",
    ]
    (output_dir / "trust_fusion_v1_summary.txt").write_text("\n".join(report), encoding="utf-8")

    metadata = {
        "phase": "Trust Fusion V1",
        "purpose": "offline exploratory diagnostic only",
        "conditions": list(CONDITIONS),
        "clients": 10,
        "rounds_averaged_for_historical_and_similarity": 10,
        "cavc_min_class_support": 100,
        "malicious_client_id": MALICIOUS_CLIENT_ID,
        "default_weights": default,
        "weight_sweep_step": 0.1,
        "input_files": {
            "cavc_v2_client_summary_min_support_100": str(cavc_path),
            "historical_clean": str(hist_clean_path),
            "historical_poisoned": str(hist_poison_path),
            "similarity_clean": str(sim_clean_path),
            "similarity_poisoned": str(sim_poison_path),
        },
        "limitations": [
            "CAVC frozen-checkpoint candidate diagnostics are not round-matched with the two 10-round federated diagnostics.",
            "One attacker, one attack configuration and one seed; no generalization claim.",
            "Percentile features are cohort-relative, not calibrated probabilities.",
            "Weight sweep is exploratory and reuses this same scenario; weights are not selected for deployment.",
            "No screening, aggregation change, model training or independent-test evaluation.",
        ],
    }
    (output_dir / "trust_fusion_v1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTRUST FUSION V1 DIAGNOSTIC COMPLETED")
    print("Output directory:", output_dir)
    print("\nEqual-weight fusion, poisoned condition:")
    print(f"Client 07 rank: {default_metrics['malicious_client_rank_most_suspicious_first']}/10 (1 = most suspicious)")
    print(f"Client 07 score minus honest mean: {default_metrics['malicious_minus_honest_mean_score']:.6f}")
    print(f"Descriptive ROC-AUC (10 clients, one attacker): {default_metrics['roc_auc_10_clients_one_attacker']:.6f}")
    print("\nTop 5 exploratory weight combinations:")
    print(grid.head(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nScreening enabled: NO")
    print("Aggregation modified: NO")
    print("Independent test used: NO")
    print("Existing results overwritten: NO")
    return output_dir


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run offline TRFA Trust Fusion V1 diagnostic.")
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--cavc-csv", default=r"results\scenario_cavc_v2_paired_client07_seed43\cavc_v2_client_summary_min_support_100.csv")
    parser.add_argument("--historical-clean", default=r"results\historical_trust_v2_clean_noniid_seed43_matched_clean_control_01\historical_v2_client_round_trust.csv")
    parser.add_argument("--historical-poison", default=r"results\historical_trust_v2_poison_client07_noniid_seed43\historical_v2_client_round_trust.csv")
    parser.add_argument("--similarity-clean", default=r"results\similarity_v2_clean_noniid_seed43\similarity_v2_client_round_scores.csv")
    parser.add_argument("--similarity-poison", default=r"results\similarity_v2_poison_client07_noniid_seed43\similarity_v2_client_round_scores.csv")
    parser.add_argument("--output-tag", default="seed43_noniid_client07_v1")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_fusion(parse_args())
    except Exception as exc:
        print(f"\nTRUST FUSION V1 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
