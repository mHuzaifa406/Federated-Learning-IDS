"""Phase 5 offline fusion diagnostic for a configurable poisoned client.

This script reads existing diagnostic outputs only. It does not train models,
change aggregation, screen clients, or access the independent test partition.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CLIENTS = set(range(1, 11))


def _resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _read_csv(path: Path, required: Iterable[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} CSV missing columns {missing}; found {list(df.columns)}")
    if df.empty:
        raise ValueError(f"{label} CSV is empty: {path}")
    return df


def _to_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        vals = set(pd.to_numeric(series, errors="raise").dropna().unique().tolist())
        if not vals.issubset({0, 1}):
            raise ValueError(f"{label} must contain boolean values.")
        return series.astype(int).astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        raise ValueError(f"{label} contains invalid boolean entries.")
    return out.astype(bool)


def _normalise_condition(value: object, malicious_client: int) -> str:
    token = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if token in {"clean", "benign"}:
        return "clean"
    match = re.fullmatch(r"(?:poison|poisoned)(?:-client)?-?(\d{1,2})?", token)
    if match:
        found = int(match.group(1)) if match.group(1) else malicious_client
        if found != malicious_client:
            raise ValueError(f"Condition refers to Client {found}, expected Client {malicious_client}.")
        return f"poison-client{malicious_client:02d}"
    raise ValueError(f"Unrecognised experiment condition: {value}")


def _validate_ids(df: pd.DataFrame, col: str, label: str) -> None:
    ids = set(pd.to_numeric(df[col], errors="raise").astype(int).unique())
    if ids != CLIENTS:
        raise ValueError(f"{label} must include clients 1–10 exactly; found {sorted(ids)}")


def _load_round_data(path: Path, kind: str, condition: str, malicious_client: int) -> pd.DataFrame:
    if kind == "historical":
        required = ["Round", "Client", "Is_Malicious", "Combined_Trust", "Update_Similarity",
                    "Temporal_Consistency", "Historical_Trust_After"]
    else:
        required = ["Round", "Client", "Is_Malicious", "pairwise_direction_score",
                    "direction_score_to_reference", "magnitude_agreement_score", "relative_magnitude",
                    "reference_distance_agreement_score", "relative_reference_distance"]
    df = _read_csv(path, required, f"{condition} {kind}")
    df["Round"] = pd.to_numeric(df["Round"], errors="raise").astype(int)
    df["Client"] = pd.to_numeric(df["Client"], errors="raise").astype(int)
    _validate_ids(df, "Client", f"{condition} {kind}")
    if sorted(df["Round"].unique().tolist()) != list(range(1, 11)):
        raise ValueError(f"{condition} {kind}: expected rounds 1–10.")
    if df.duplicated(["Round", "Client"]).any():
        raise ValueError(f"{condition} {kind}: duplicate (round, client) rows.")
    if not (df.groupby("Client")["Round"].nunique() == 10).all():
        raise ValueError(f"{condition} {kind}: every client must have 10 round rows.")
    df["Is_Malicious"] = _to_bool(df["Is_Malicious"], f"{condition} {kind} Is_Malicious")
    flags = set(df.loc[df["Is_Malicious"], "Client"].unique().tolist())
    expected = {malicious_client} if condition != "clean" else set()
    if flags != expected:
        raise ValueError(f"{condition} {kind}: expected malicious flags {sorted(expected)}, found {sorted(flags)}")
    num_cols = [c for c in required if c not in {"Round", "Client", "Is_Malicious"}]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if not np.isfinite(df[c].to_numpy(dtype=float)).all():
            raise ValueError(f"{condition} {kind}: non-finite values in {c}")
    means = df.groupby("Client", as_index=False)[num_cols].mean()
    means["Is_Malicious"] = means["Client"].eq(malicious_client) if condition != "clean" else False
    means["Condition"] = condition
    return means


def _load_cavc(path: Path, poison_condition: str, malicious_client: int) -> pd.DataFrame:
    req = ["condition", "client_id", "is_malicious", "attack_detection_contribution_score",
           "targeted_damage_penalty", "benign_fpr_damage_penalty"]
    df = _read_csv(path, req, "CAVC V2 summary")
    df["Condition"] = df["condition"].map(lambda x: _normalise_condition(x, malicious_client))
    df["Client"] = pd.to_numeric(df["client_id"], errors="raise").astype(int)
    df["is_malicious"] = _to_bool(df["is_malicious"], "CAVC is_malicious")
    if "min_class_support" in df.columns:
        supports = sorted(pd.to_numeric(df["min_class_support"], errors="raise").astype(int).unique())
        if supports != [100]:
            raise ValueError(f"CAVC input must be min_class_support=100, found {supports}")
    if df.duplicated(["Condition", "Client"]).any():
        raise ValueError("CAVC summary must have one row per client and condition.")
    for condition in ("clean", poison_condition):
        subset = df[df["Condition"] == condition]
        _validate_ids(subset, "Client", f"CAVC {condition}")
        flags = set(subset.loc[subset["is_malicious"], "Client"].tolist())
        expected = {malicious_client} if condition == poison_condition else set()
        if flags != expected:
            raise ValueError(f"CAVC {condition}: expected malicious flags {sorted(expected)}, found {sorted(flags)}")
    num = ["attack_detection_contribution_score", "targeted_damage_penalty", "benign_fpr_damage_penalty"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if not np.isfinite(df[c].to_numpy(dtype=float)).all():
            raise ValueError(f"CAVC has non-finite values in {c}")
    return df[["Condition", "Client", *num]].copy()


def _pct(values: pd.Series, higher_is_suspicious: bool) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if x.isna().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Cannot rank missing/non-finite diagnostic values.")
    return x.rank(method="average", ascending=higher_is_suspicious, pct=True)


def _add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sim_specs = {
        "sim_pairwise_direction": ("pairwise_direction_score", False),
        "sim_reference_direction": ("direction_score_to_reference", False),
        "sim_magnitude_agreement": ("magnitude_agreement_score", False),
        "sim_reference_distance": ("reference_distance_agreement_score", False),
    }
    sim_cols = []
    for target, (source, high_bad) in sim_specs.items():
        out[target] = out.groupby("Condition")[source].transform(lambda s: _pct(s, high_bad))
        sim_cols.append(target)
    out["sim_relative_magnitude_deviation"] = np.abs(np.log(np.clip(out["relative_magnitude"].astype(float), 1e-12, None)))
    out["sim_relative_magnitude"] = out.groupby("Condition")["sim_relative_magnitude_deviation"].transform(lambda s: _pct(s, True))
    out["sim_reference_distance_deviation"] = np.abs(np.log(np.clip(out["relative_reference_distance"].astype(float), 1e-12, None)))
    out["sim_reference_distance_ratio"] = out.groupby("Condition")["sim_reference_distance_deviation"].transform(lambda s: _pct(s, True))
    sim_cols += ["sim_relative_magnitude", "sim_reference_distance_ratio"]
    out["Similarity_Group_Score"] = out[sim_cols].mean(axis=1)

    # Historical V2 Combined_Trust already contains similarity/consistency/history.
    out["Historical_Group_Score"] = out.groupby("Condition")["Combined_Trust"].transform(lambda s: _pct(s, False))
    out["cavc_attack_recall_concern"] = out.groupby("Condition")["attack_detection_contribution_score"].transform(lambda s: _pct(s, False))
    out["cavc_targeted_damage_concern"] = out.groupby("Condition")["targeted_damage_penalty"].transform(lambda s: _pct(s, True))
    out["cavc_benign_fpr_concern"] = out.groupby("Condition")["benign_fpr_damage_penalty"].transform(lambda s: _pct(s, True))
    out["CAVC_Group_Score"] = out[["cavc_attack_recall_concern", "cavc_targeted_damage_concern", "cavc_benign_fpr_concern"]].mean(axis=1)
    out["Equal_Weight_Fusion_Score"] = out[["Similarity_Group_Score", "Historical_Group_Score", "CAVC_Group_Score"]].mean(axis=1)
    return out


def _weight_grid(step: float = 0.1) -> list[dict[str, float]]:
    ticks = int(round(1 / step))
    return [{"similarity_weight": s / ticks, "historical_weight": h / ticks, "cavc_weight": (ticks - s - h) / ticks}
            for s in range(ticks + 1) for h in range(ticks + 1 - s)]


def _metrics(poison_df: pd.DataFrame, score_col: str, malicious_client: int) -> dict:
    ordered = poison_df.sort_values([score_col, "Client"], ascending=[False, True]).reset_index(drop=True)
    target = ordered[ordered["Client"] == malicious_client]
    if len(target) != 1:
        raise ValueError("Expected exactly one target client row in poisoned condition.")
    honest = ordered[ordered["Client"] != malicious_client]
    labels = ordered["Client"].eq(malicious_client).astype(int).to_numpy()
    scores = ordered[score_col].to_numpy(dtype=float)
    return {
        "malicious_client_rank_most_suspicious_first": int(ordered.index[ordered["Client"] == malicious_client][0]) + 1,
        "malicious_minus_honest_mean_score": float(target.iloc[0][score_col] - honest[score_col].mean()),
        "roc_auc_10_clients_one_attacker": float(roc_auc_score(labels, scores)),
        "average_precision_10_clients_one_attacker": float(average_precision_score(labels, scores)),
        "malicious_score": float(target.iloc[0][score_col]),
    }


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Offline Phase 5 fusion diagnostic for one client/attack setting.")
    p.add_argument("--project-root", default=str(project_root))
    p.add_argument("--malicious-client", required=True, type=int, choices=range(1, 11))
    p.add_argument("--training-seed", type=int, default=43)
    p.add_argument("--poison-fraction", required=True, type=float)
    p.add_argument("--poison-seed", type=int, default=2027)
    p.add_argument("--cavc-csv", required=True)
    p.add_argument("--historical-clean", required=True)
    p.add_argument("--historical-poison", required=True)
    p.add_argument("--similarity-clean", required=True)
    p.add_argument("--similarity-poison", required=True)
    p.add_argument("--output-tag", required=True)
    return p.parse_args()


def run(args) -> Path:
    if not 0.0 < args.poison_fraction < 1.0:
        raise ValueError("--poison-fraction must be between 0 and 1.")
    root = Path(args.project_root).resolve()
    cid = args.malicious_client
    poison_condition = f"poison-client{cid:02d}"
    paths = {
        "cavc": _resolve(root, args.cavc_csv),
        "hist_clean": _resolve(root, args.historical_clean),
        "hist_poison": _resolve(root, args.historical_poison),
        "sim_clean": _resolve(root, args.similarity_clean),
        "sim_poison": _resolve(root, args.similarity_poison),
    }
    cavc = _load_cavc(paths["cavc"], poison_condition, cid)
    parts = []
    for condition, hkey, skey in [("clean", "hist_clean", "sim_clean"), (poison_condition, "hist_poison", "sim_poison")]:
        hist = _load_round_data(paths[hkey], "historical", condition, cid)
        sim = _load_round_data(paths[skey], "similarity", condition, cid)
        hist = hist.drop(columns=["Condition"])
        sim = sim.drop(columns=["Condition"])
        merged = hist.merge(sim, on=["Client", "Is_Malicious"], how="inner", validate="one_to_one", suffixes=("_hist", "_sim"))
        if set(merged["Client"].astype(int)) != CLIENTS:
            raise ValueError(f"Client IDs did not align for {condition}.")
        merged["Condition"] = condition
        parts.append(merged)
    combined = pd.concat(parts, ignore_index=True).merge(cavc, on=["Condition", "Client"], how="inner", validate="one_to_one")
    if len(combined) != 20:
        raise ValueError(f"Expected 20 client-condition rows after fusion; received {len(combined)}.")
    combined = _add_scores(combined)
    combined["Fusion_Rank_Most_Suspicious_First"] = combined.groupby("Condition")["Equal_Weight_Fusion_Score"].rank(method="average", ascending=False)

    score_cols = ["Similarity_Group_Score", "Historical_Group_Score", "CAVC_Group_Score", "Equal_Weight_Fusion_Score"]
    clean = combined[combined.Condition == "clean"].set_index("Client")
    poison = combined[combined.Condition == poison_condition].set_index("Client")
    delta = (poison[score_cols] - clean[score_cols]).reset_index().rename(columns={c: f"Poison_Minus_Clean_{c}" for c in score_cols})
    delta["Change_Rank_Most_Increased_First"] = delta["Poison_Minus_Clean_Equal_Weight_Fusion_Score"].rank(method="average", ascending=False)
    combined = combined.merge(delta, on="Client", how="left", validate="many_to_one")

    grid_rows = []
    poison_rows = combined[combined.Condition == poison_condition].copy()
    for w in _weight_grid(0.1):
        score = (w["similarity_weight"] * poison_rows["Similarity_Group_Score"]
                 + w["historical_weight"] * poison_rows["Historical_Group_Score"]
                 + w["cavc_weight"] * poison_rows["CAVC_Group_Score"])
        temp = poison_rows.assign(_score=score)
        m = _metrics(temp, "_score", cid)
        clean_scores = (w["similarity_weight"] * clean["Similarity_Group_Score"]
                        + w["historical_weight"] * clean["Historical_Group_Score"]
                        + w["cavc_weight"] * clean["CAVC_Group_Score"])
        paired_change = score.reset_index(drop=True) - clean_scores.reindex(poison_rows["Client"]).to_numpy()
        change_rank = pd.Series(paired_change, index=poison_rows["Client"]).rank(ascending=False, method="average").loc[cid]
        grid_rows.append({**w, **m, "target_paired_change": float(paired_change[list(poison_rows["Client"]).index(cid)]),
                          "target_paired_change_rank": float(change_rank)})
    grid = pd.DataFrame(grid_rows).sort_values(["malicious_client_rank_most_suspicious_first", "roc_auc_10_clients_one_attacker", "malicious_minus_honest_mean_score"],
                                                ascending=[True, False, False]).reset_index(drop=True)

    baseline_metrics = {}
    for col in ("Similarity_Group_Score", "Historical_Group_Score", "CAVC_Group_Score", "Equal_Weight_Fusion_Score"):
        baseline_metrics[col] = _metrics(poison_rows.assign(_score=poison_rows[col]), "_score", cid)
    equal_metrics = baseline_metrics["Equal_Weight_Fusion_Score"]
    target_clean = float(clean.loc[cid, "Equal_Weight_Fusion_Score"])
    target_poison = float(poison.loc[cid, "Equal_Weight_Fusion_Score"])
    tag = "".join(ch for ch in args.output_tag if ch.isalnum() or ch in "-_")
    if not tag:
        raise ValueError("--output-tag must contain letters, numbers, '-' or '_'.")
    output_dir = root / "results" / f"phase5_fusion_c{cid:02d}_p{args.poison_fraction:.4f}_s{args.training_seed}_{tag}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {output_dir}; choose another --output-tag.")
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "phase5_fusion_client_scores.csv", index=False)
    delta.to_csv(output_dir / "phase5_fusion_paired_changes.csv", index=False)
    grid.to_csv(output_dir / "phase5_fusion_weight_sweep_exploratory.csv", index=False)

    lines = [
        "TRFA PHASE 5 — MULTI-SCENARIO FUSION DIAGNOSTIC",
        "================================================",
        f"Malicious client: {cid}", f"Poison fraction of eligible non-Benign records: {args.poison_fraction:.6%}",
        f"Training seed: {args.training_seed}", f"Poison selection seed: {args.poison_seed}",
        "No training, screening, aggregation change, or independent-test evaluation was performed by this fusion script.",
        "Higher scores mean greater cohort-relative diagnostic unusualness; they are not probabilities.", "",
        "Component comparison in poisoned condition:",
        *[f"  {name}: rank {m['malicious_client_rank_most_suspicious_first']}/10; ROC-AUC {m['roc_auc_10_clients_one_attacker']:.6f}; AP {m['average_precision_10_clients_one_attacker']:.6f}"
          for name, m in baseline_metrics.items()],
        "",
        "Equal-weight fusion in poisoned condition:",
        f"  Target client rank (1 = most suspicious): {equal_metrics['malicious_client_rank_most_suspicious_first']}/10",
        f"  Target minus honest mean: {equal_metrics['malicious_minus_honest_mean_score']:.6f}",
        f"  Descriptive ROC-AUC (10 clients, one attacker): {equal_metrics['roc_auc_10_clients_one_attacker']:.6f}",
        f"  Descriptive average precision: {equal_metrics['average_precision_10_clients_one_attacker']:.6f}",
        f"  Clean target score: {target_clean:.6f}", f"  Poisoned target score: {target_poison:.6f}",
        f"  Poisoned minus clean score: {target_poison-target_clean:.6f}", "",
        "Top exploratory weights on this same scenario (not a selected/validated setting):",
        grid.head(5).to_string(index=False), "",
        "Interpretation limits:",
        "- CAVC is a frozen-checkpoint local-candidate diagnostic; the other two components are 10-round FL diagnostics.",
        "- Percentile normalization is relative to these 10 clients and each condition.",
        "- The 10-client ROC-AUC has one positive client and is descriptive only.",
        "- The weight sweep reuses this same scenario and must not be used to claim optimal weights.",
        "- This scenario alone cannot establish cross-client, cross-strength, or cross-seed generalization.",
        "- Screening remained disabled in the upstream component experiments; independent test data was not used."
    ]
    (output_dir / "phase5_fusion_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    meta = {
        "phase": "5", "malicious_client": cid, "poison_fraction_of_eligible_nonbenign": args.poison_fraction,
        "training_seed": args.training_seed, "poison_seed": args.poison_seed,
        "inputs": {k: str(v) for k, v in paths.items()}, "clients": 10, "rounds_averaged": 10,
        "cavc_min_class_support": 100, "screening_enabled": False, "independent_test_used": False,
        "limitations": ["CAVC and full-FL diagnostics are not the same experimental unit.",
                         "Cohort percentiles are not calibrated probabilities.",
                         "Single scenario, one attacker, one seed; descriptive metrics only.",
                         "Weight sweep is exploratory and in-sample."]
    }
    (output_dir / "phase5_fusion_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nPHASE 5 FUSION COMPLETED")
    print("Output directory:", output_dir)
    print("Target client rank:", equal_metrics["malicious_client_rank_most_suspicious_first"], "/10")
    print("Descriptive ROC-AUC:", f"{equal_metrics['roc_auc_10_clients_one_attacker']:.6f}")
    print("Clean-to-poison target score change:", f"{target_poison-target_clean:+.6f}")
    print("Screening enabled: NO; aggregation modified: NO; independent test used: NO")
    return output_dir


if __name__ == "__main__":
    run(parse_args())
