"""Run Phase 10 BCTCD V1 on one existing Phase 9 delayed-onset experiment."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase10_bctcd_core import replay_bctcd


def parse_args():
    p = argparse.ArgumentParser(description="Baseline-calibrated causal change-detection replay.")
    p.add_argument("--input-dir", required=True, help="Existing Phase 9 experiment output directory, relative to project root or absolute.")
    p.add_argument("--warmup-rounds", type=int, default=3)
    p.add_argument("--cusum-slack", type=float, default=1.0)
    p.add_argument("--z-cap", type=float, default=6.0)
    p.add_argument("--output-tag", required=True, help="Unique output folder suffix.")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    input_dir = input_dir.resolve()
    input_csv = input_dir / "phase9_client_round_scores.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing Phase 9 input: {input_csv}")
    if not args.output_tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.output_tag):
        raise ValueError("--output-tag must contain only letters, numbers, hyphens or underscores.")

    raw = pd.read_csv(input_csv)
    scores, metrics, baselines = replay_bctcd(raw, args.warmup_rounds, args.cusum_slack, args.z_cap)
    output_dir = PROJECT_ROOT / "results" / f"phase10_bctcd_{args.output_tag}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains results: {output_dir}. Use another --output-tag.")
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "phase10_bctcd_client_round_scores.csv", index=False)
    metrics.to_csv(output_dir / "phase10_bctcd_round_metrics.csv", index=False)
    baselines.to_csv(output_dir / "phase10_bctcd_baseline_parameters.csv", index=False)

    active = metrics[metrics["Attack_Active"]]
    preattack_monitor = metrics[~metrics["Attack_Active"]]
    summary = {
        "phase": "Phase 10 BCTCD V1",
        "method": "Client-specific robust baseline + per-round deviation + one-sided CUSUM",
        "input_experiment": str(input_dir),
        "warmup_rounds": int(args.warmup_rounds),
        "score_monitoring_starts_round": int(args.warmup_rounds + 1),
        "cusum_slack": float(args.cusum_slack),
        "z_cap": float(args.z_cap),
        "target_client": int(raw.loc[raw["Is_Malicious"].astype(bool), "Client"].iloc[0]),
        "attack_start_round": int(raw["Attack_Start_Round"].iloc[0]),
        "active_rounds": int(len(active)),
        "mean_active_deviation_auc": float(active["Deviation_ROC_AUC"].mean()),
        "mean_active_cusum_auc": float(active["CUSUM_ROC_AUC"].mean()),
        "active_deviation_top3_rate": float(active["Target_Deviation_Top3"].mean()),
        "active_cusum_top3_rate": float(active["Target_CUSUM_Top3"].mean()),
        "pre_attack_monitor_rounds": int(len(preattack_monitor)),
        "pre_attack_deviation_top3_rounds": int(preattack_monitor["Target_Deviation_Top3"].sum()),
        "pre_attack_cusum_top3_rounds": int(preattack_monitor["Target_CUSUM_Top3"].sum()),
        "screening_enabled": False,
        "aggregation_modified": False,
        "independent_test_used": False,
        "limitations": [
            "Offline causal replay of an existing Phase 9 run; no new model training performed.",
            "Baseline is estimated from the first warmup_rounds observations and frozen afterward.",
            "Feature scales include predeclared floors because the warm-up has only a few rounds.",
            "Scores are client-specific deviations and cohort rankings, not calibrated attack probabilities.",
            "One delayed-onset scenario is not sufficient for generalization; matched clean runs are still required.",
        ],
    }
    (output_dir / "phase10_bctcd_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "PHASE 10 — BASELINE-CALIBRATED TEMPORAL CHANGE DETECTION (BCTCD V1)",
        "===================================================================",
        f"Input experiment: {input_dir}",
        f"Target client: {summary['target_client']}",
        f"Attack start round: {summary['attack_start_round']}",
        f"Warm-up rounds: {summary['warmup_rounds']}",
        f"CUSUM slack: {summary['cusum_slack']}",
        "",
        f"Mean active-round deviation ROC-AUC: {summary['mean_active_deviation_auc']:.6f}",
        f"Mean active-round CUSUM ROC-AUC: {summary['mean_active_cusum_auc']:.6f}",
        f"Deviation top-3 rate after onset: {summary['active_deviation_top3_rate']:.6f}",
        f"CUSUM top-3 rate after onset: {summary['active_cusum_top3_rate']:.6f}",
        f"Pre-attack monitored rounds: {summary['pre_attack_monitor_rounds']}",
        f"Pre-attack deviation top-3 rounds: {summary['pre_attack_deviation_top3_rounds']}",
        f"Pre-attack CUSUM top-3 rounds: {summary['pre_attack_cusum_top3_rounds']}",
        "",
        "No training was performed. No client was excluded. Independent test was not used.",
        "This is a diagnostic replay, not validated live attack detection.",
    ]
    (output_dir / "phase10_bctcd_summary.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nPHASE 10 BCTCD V1 REPLAY COMPLETED")
    print("Input:", input_dir)
    print("Output directory:", output_dir)
    print("Warm-up rounds:", args.warmup_rounds)
    print("Mean active-round deviation ROC-AUC:", f"{summary['mean_active_deviation_auc']:.6f}")
    print("Mean active-round CUSUM ROC-AUC:", f"{summary['mean_active_cusum_auc']:.6f}")
    print("Deviation top-3 rate:", f"{summary['active_deviation_top3_rate']:.6f}")
    print("CUSUM top-3 rate:", f"{summary['active_cusum_top3_rate']:.6f}")
    print("Pre-attack monitored top-3 rounds: deviation", summary["pre_attack_deviation_top3_rounds"], "/", summary["pre_attack_monitor_rounds"],
          "CUSUM", summary["pre_attack_cusum_top3_rounds"], "/", summary["pre_attack_monitor_rounds"])
    print("Screening enabled: NO; aggregation modified: NO; independent test used: NO")

if __name__ == "__main__":
    main()
