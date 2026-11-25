"""Apply a clean-reference CCTRT calibration to one Phase 9 score CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trfa.cctrt_calibration import load_calibration, apply_calibration_to_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--score-csv", required=True)
    parser.add_argument("--metadata", default=None, help="Defaults to phase9_metadata.json beside score CSV.")
    parser.add_argument("--target-client", type=int, default=None, help="Set only for an attack run; inferred from metadata when available.")
    parser.add_argument("--attack-start-round", type=int, default=None, help="Inferred from metadata when available.")
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    score_path = Path(args.score_csv).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else score_path.parent / "phase9_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    target = args.target_client if args.target_client is not None else metadata.get("malicious_client")
    onset = args.attack_start_round if args.attack_start_round is not None else metadata.get("attack_start_round")

    calibration = load_calibration(
        args.calibration,
        expected_partition_seed=(
            int(metadata["partition_seed"]) if metadata.get("partition_seed") is not None else None
        ),
        expected_alpha=(
            float(metadata["dirichlet_alpha"]) if metadata.get("dirichlet_alpha") is not None else None
        ),
        expected_rounds=(
            int(metadata["rounds"]) if metadata.get("rounds") is not None else None
        ),
        expected_clients=(
            int(metadata["clients"]) if metadata.get("clients") is not None else None
        ),
    )
    evaluation_seed = metadata.get("training_seed")
    if evaluation_seed is not None and int(evaluation_seed) in [
        int(seed) for seed in calibration["training_seeds"]
    ]:
        raise ValueError(
            "Evaluation training seed appears in calibration seeds; "
            "choose an independent held-out evaluation run."
        )
    frame = pd.read_csv(score_path)
    scored = apply_calibration_to_frame(frame, calibration, persistence=args.persistence)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "calibrated_client_round_scores.csv"
    scored.to_csv(scored_path, index=False)

    total_rows = len(scored)
    false_alert_rows = 0
    true_alerts = 0
    active_opportunities = 0
    detection_rounds: list[int] = []
    aucs: list[float] = []
    aps: list[float] = []

    if target is None:
        false_alert_rows = int(scored["Rejected_By_Calibrated_TRFA"].sum())
        clean_fpr = false_alert_rows / total_rows if total_rows else 0.0
        summary = {
            "evaluation_type": "all_honest_control",
            "target_client": None,
            "calibration_training_seeds": calibration["training_seeds"],
            "calibration_threshold_z": calibration["threshold_z"],
            "persistence_rounds": args.persistence,
            "client_round_opportunities": total_rows,
            "false_excluded_client_rounds": false_alert_rows,
            "false_exclusion_rate": clean_fpr,
            "false_excluded_clients": sorted(scored.loc[scored["Rejected_By_Calibrated_TRFA"], "Client"].unique().astype(int).tolist()),
        }
    else:
        target = int(target)
        onset = int(onset) if onset is not None else 1
        for rnd, group in scored.groupby("Round", sort=True):
            if int(rnd) < onset:
                continue
            active_opportunities += 1
            y_true = group["Client"].eq(target).astype(int).to_numpy()
            y_score = group["Calibrated_Anomaly_Z"].astype(float).to_numpy()
            if len(np.unique(y_true)) == 2:
                aucs.append(float(roc_auc_score(y_true, y_score)))
                aps.append(float(average_precision_score(y_true, y_score)))
            row = group[group["Client"] == target]
            if not row.empty and bool(row.iloc[0]["Rejected_By_Calibrated_TRFA"]):
                true_alerts += 1
                detection_rounds.append(int(rnd))
        honest = scored[scored["Client"] != target]
        false_alert_rows = int(honest["Rejected_By_Calibrated_TRFA"].sum())
        honest_opportunities = len(honest)
        clean_fpr = false_alert_rows / honest_opportunities if honest_opportunities else 0.0
        pre = scored[(scored["Client"] == target) & (scored["Round"] < onset)]
        pre_false_alerts = int(pre["Rejected_By_Calibrated_TRFA"].sum())
        summary = {
            "evaluation_type": "delayed_onset_attack",
            "target_client": target,
            "attack_start_round": onset,
            "calibration_training_seeds": calibration["training_seeds"],
            "calibration_threshold_z": calibration["threshold_z"],
            "persistence_rounds": args.persistence,
            "active_rounds": active_opportunities,
            "target_detected_active_rounds": true_alerts,
            "target_detection_rate": true_alerts / active_opportunities if active_opportunities else None,
            "first_detection_round": min(detection_rounds) if detection_rounds else None,
            "detection_delay_rounds": (min(detection_rounds) - onset + 1) if detection_rounds else None,
            "honest_client_round_opportunities": honest_opportunities,
            "false_excluded_honest_client_rounds": false_alert_rows,
            "honest_false_exclusion_rate": clean_fpr,
            "target_false_alerts_before_onset": pre_false_alerts,
            "mean_active_round_auc": float(np.mean(aucs)) if aucs else None,
            "mean_active_round_average_precision": float(np.mean(aps)) if aps else None,
        }

    summary_path = output_dir / "calibrated_detection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Per-client-round output:", scored_path)
    print("Summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
