"""Run BCTCD on an all-honest matched FL control; no model training occurs here."""
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
from scripts.phase10_matched_control_core import prepare_clean_replay_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Replay BCTCD on a matched all-honest control run.")
    parser.add_argument("--input-dir", required=True, help="Phase 9 clean-control results folder.")
    parser.add_argument("--target-client", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--cusum-slack", type=float, default=1.0)
    parser.add_argument("--z-cap", type=float, default=6.0)
    parser.add_argument("--output-tag", required=True)
    return parser.parse_args()


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def run(args) -> Path:
    if args.warmup_rounds < 2:
        raise ValueError("--warmup-rounds must be at least 2.")
    if not args.output_tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.output_tag):
        raise ValueError("--output-tag must contain only letters, numbers, hyphens and underscores.")
    input_dir = _resolve(args.input_dir)
    input_csv = input_dir / "phase9_client_round_scores.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing clean Phase 9 client-round file: {input_csv}")
    raw = pd.read_csv(input_csv)
    replay_frame = prepare_clean_replay_frame(raw, args.target_client)
    scores, metrics, baselines = replay_bctcd(
        replay_frame, warmup_rounds=args.warmup_rounds,
        slack=args.cusum_slack, z_cap=args.z_cap,
    )
    # The replay core uses an internal positive label to track one probe client.
    # Restore the true experimental status before saving/reporting the control.
    scores["Is_Malicious"] = False
    scores["Target_Client"] = scores["Client"].eq(int(args.target_client))
    scores["Attack_Active"] = False
    scores["Control_Condition"] = "all_honest"
    metrics["Attack_Active"] = False
    metrics["Control_Condition"] = "all_honest"

    output_dir = PROJECT_ROOT / "results" / f"phase10_bctcd_clean_{args.output_tag}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output folder already contains results: {output_dir}. Choose a new --output-tag.")
    output_dir.mkdir(parents=True, exist_ok=True)

    scores.to_csv(output_dir / "phase10_bctcd_clean_client_round_scores.csv", index=False)
    metrics.to_csv(output_dir / "phase10_bctcd_clean_round_metrics.csv", index=False)
    baselines.to_csv(output_dir / "phase10_bctcd_clean_baseline_parameters.csv", index=False)

    monitored_rounds = sorted(scores["Round"].unique().tolist())
    alert_summary = scores.groupby("Client", as_index=False).agg(
        Monitored_Rounds=("Round", "nunique"),
        Deviation_Top3_Rounds=("BCTCD_Deviation_Top3", "sum"),
        CUSUM_Top3_Rounds=("BCTCD_CUSUM_Top3", "sum"),
        Deviation_Top1_Rounds=("BCTCD_Deviation_Rank", lambda s: int((s == 1).sum())),
        CUSUM_Top1_Rounds=("BCTCD_CUSUM_Rank", lambda s: int((s == 1).sum())),
        Mean_Deviation=("BCTCD_Round_Deviation", "mean"),
        Mean_CUSUM=("BCTCD_CUSUM_Score", "mean"),
        Max_CUSUM=("BCTCD_CUSUM_Score", "max"),
    )
    alert_summary["Deviation_Top3_Rate"] = alert_summary["Deviation_Top3_Rounds"] / alert_summary["Monitored_Rounds"]
    alert_summary["CUSUM_Top3_Rate"] = alert_summary["CUSUM_Top3_Rounds"] / alert_summary["Monitored_Rounds"]
    alert_summary["Is_Probe_Client"] = alert_summary["Client"].eq(int(args.target_client))
    alert_summary.to_csv(output_dir / "phase10_bctcd_clean_client_alert_summary.csv", index=False)

    probe = alert_summary[alert_summary["Is_Probe_Client"]].iloc[0]
    metadata = {
        "phase": "Phase 10 matched all-honest BCTCD control",
        "input_phase9_directory": str(input_dir),
        "target_probe_client": int(args.target_client),
        "all_clients_honest": True,
        "attack_active": False,
        "warmup_rounds": int(args.warmup_rounds),
        "monitored_rounds": monitored_rounds,
        "cusum_slack": float(args.cusum_slack),
        "z_cap": float(args.z_cap),
        "probe_client_deviation_top3_rounds": int(probe["Deviation_Top3_Rounds"]),
        "probe_client_cusum_top3_rounds": int(probe["CUSUM_Top3_Rounds"]),
        "probe_client_deviation_top3_rate": float(probe["Deviation_Top3_Rate"]),
        "probe_client_cusum_top3_rate": float(probe["CUSUM_Top3_Rate"]),
        "screening_enabled": False,
        "aggregation_modified": False,
        "independent_test_used": False,
        "interpretation": "Top-3 is a cohort ranking diagnostic, not a calibrated false-positive rate or rejection threshold.",
        "files": [
            "phase10_bctcd_clean_client_round_scores.csv",
            "phase10_bctcd_clean_round_metrics.csv",
            "phase10_bctcd_clean_baseline_parameters.csv",
            "phase10_bctcd_clean_client_alert_summary.csv",
        ],
    }
    (output_dir / "phase10_bctcd_clean_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = [
        "PHASE 10 — MATCHED ALL-HONEST BCTCD CONTROL",
        "=============================================",
        f"Probe client (honest throughout): {args.target_client}",
        f"Input Phase 9 directory: {input_dir}",
        f"Baseline warm-up rounds: 1-{args.warmup_rounds}",
        f"Monitored rounds: {monitored_rounds}",
        "Attack active: NEVER",
        f"Probe deviation top-3: {int(probe['Deviation_Top3_Rounds'])}/{int(probe['Monitored_Rounds'])}",
        f"Probe CUSUM top-3: {int(probe['CUSUM_Top3_Rounds'])}/{int(probe['Monitored_Rounds'])}",
        f"Probe mean deviation: {float(probe['Mean_Deviation']):.6f}",
        f"Probe mean CUSUM: {float(probe['Mean_CUSUM']):.6f}",
        "",
        "All clients remained honest and were aggregated.",
        "Top-3 rates are cohort-relative ranking indicators, not calibrated false-positive rates.",
        "Screening disabled; aggregation unchanged; independent test not used.",
    ]
    (output_dir / "phase10_bctcd_clean_summary.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nPHASE 10 MATCHED CLEAN-CONTROL REPLAY COMPLETED")
    print("Output directory:", output_dir)
    print("Probe client (honest throughout):", args.target_client)
    print("Monitored rounds:", monitored_rounds)
    print("Probe deviation top-3:", f"{int(probe['Deviation_Top3_Rounds'])}/{int(probe['Monitored_Rounds'])}")
    print("Probe CUSUM top-3:", f"{int(probe['CUSUM_Top3_Rounds'])}/{int(probe['Monitored_Rounds'])}")
    print("Screening enabled: NO; aggregation modified: NO; independent test used: NO")
    return output_dir


if __name__ == "__main__":
    run(parse_args())
