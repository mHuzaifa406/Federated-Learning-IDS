"""Phase 9: binary non-IID FL run with delayed-onset label flipping and online CCTRT.

The selected client behaves honestly before --attack-start-round. From that round
onward, the configured share of its eligible non-Benign local labels is flipped
to Benign in every communication round. Every round's CCTRT score is computed
immediately using only current and earlier-round observations.

By default the runner is diagnostic and aggregates all clients. Supplying
--calibration enables clean-reference calibrated screening and accepted-only
aggregation. The independent test set is never used by this runner.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score

import scripts.federated_engine as engine
from models.ids_model import IntrusionDetectionMLP
from trfa.trust_metrics import (
    flatten_model_update,
    calculate_reference_update,
    calculate_update_similarity,
    TrainingConsistencyTracker,
)
from trfa.trust_similarity_v2 import calculate_update_similarity_v2
from trfa.historical_trust_v2 import (
    HistoricalTrustTrackerV2,
    calculate_current_evidence,
    calculate_combined_trust_v2,
)
from trfa.aggregation import aggregate_accepted_models
from scripts.phase9_cctrt_core import OnlineCCTRTState, attack_active_this_round, poison_eligible_labels
from trfa.cctrt_calibration import load_calibration, score_calibrated_round

CLIENT_IDS = list(range(1, 11))
TARGET_LABEL = 0
DIRICHLET_ALPHA = 0.5
ROUNDS = 10
TRUST_ALPHA = TRUST_BETA = TRUST_GAMMA = 1.0 / 3.0
HISTORY_RETENTION = 0.90
INITIAL_HISTORICAL_TRUST = 0.50


def to_binary_labels(original_labels: np.ndarray) -> np.ndarray:
    """Map CIC-IDS-2017 Label_id 0 to Benign and 1..6 to Attack."""
    labels = np.asarray(original_labels, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Cannot convert an empty label array.")
    if np.any((labels < 0) | (labels > 6)):
        raise ValueError("Expected original CIC-IDS-2017 Label_id values in 0..6.")
    return (labels != 0).astype(np.int64)


def configure_binary_training() -> None:
    """Use the selected and previously validated binary-FedAvg settings."""
    engine.LEARNING_RATE = 0.015
    engine.LOCAL_EPOCHS = 2
    engine.BATCH_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real FL experiment with delayed-onset label flipping and online CCTRT."
    )
    parser.add_argument("--malicious-client", required=True, type=int, choices=CLIENT_IDS)
    parser.add_argument("--poison-fraction", type=float, default=0.20,
                        help="Fraction of eligible non-Benign labels flipped to Benign once attack begins.")
    parser.add_argument("--attack-start-round", required=True, type=int, choices=(4, 6),
                        help="First round in which the selected client is poisoned.")
    parser.add_argument("--training-seed", type=int, default=44)
    parser.add_argument("--poison-seed", type=int, default=2028)
    parser.add_argument("--partition-root", default=str(PROJECT_ROOT / "data" / "processed" / "federated_noniid_alpha05_seed43"))
    parser.add_argument("--output-tag", required=True)
    parser.add_argument(
        "--calibration", default=None,
        help="Optional clean-reference calibration JSON; enables calibrated client screening and accepted-only aggregation."
    )
    parser.add_argument(
        "--persistence", type=int, default=2,
        help="Consecutive calibrated alerts required before rejecting a client (default: 2)."
    )
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    engine.set_random_seed()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(engine.CPU_THREADS)


def load_partition_metadata(data_root: Path) -> dict[str, Any]:
    metadata_path = data_root / "metadata.json"
    if not data_root.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Non-IID partition or metadata not found: {data_root}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not math.isclose(float(metadata.get("dirichlet_alpha", -1)), DIRICHLET_ALPHA, abs_tol=1e-9):
        raise ValueError(f"Expected Dirichlet alpha {DIRICHLET_ALPHA}; found {metadata.get('dirichlet_alpha')}")
    if int(metadata.get("random_seed", -1)) != 43:
        raise ValueError("This Phase 9 package expects the established non-IID partition seed 43.")
    return metadata



def evaluate(model: torch.nn.Module, loader) -> dict[str, float]:
    model.eval()
    true: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(engine.DEVICE))
            prediction = torch.argmax(logits, dim=1)
            true.extend(y.detach().cpu().numpy().astype(int).tolist())
            predicted.extend(prediction.detach().cpu().numpy().astype(int).tolist())
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, predicted, average="weighted", zero_division=0)),
    }


def run(args: argparse.Namespace) -> Path:
    if not 0.0 < args.poison_fraction < 1.0:
        raise ValueError("--poison-fraction must be greater than 0 and less than 1.")
    if isinstance(args.persistence, bool) or int(args.persistence) < 1:
        raise ValueError("--persistence must be a positive integer.")
    tag = "".join(c for c in args.output_tag if c.isalnum() or c in "-_")
    if not tag:
        raise ValueError("--output-tag must contain letters, digits, '-' or '_'.")
    data_root = Path(args.partition_root).resolve()
    partition_metadata = load_partition_metadata(data_root)
    calibration = None
    if args.calibration:
        calibration = load_calibration(
            args.calibration,
            expected_partition_seed=int(partition_metadata["random_seed"]),
            expected_alpha=DIRICHLET_ALPHA,
            expected_rounds=ROUNDS,
            expected_clients=len(CLIENT_IDS),
            expected_classification_task="binary_benign_vs_attack",
            expected_model_config={"input_features": 78, "num_classes": 2},
            expected_training_configuration={
                "local_epochs": 2,
                "learning_rate": 0.015,
                "batch_size": 512,
            },
        )
        if int(args.training_seed) in [int(seed) for seed in calibration["training_seeds"]]:
            raise ValueError(
                "Evaluation training seed also appears in calibration seeds; "
                "use an independent training seed to avoid leakage."
            )
    alert_streaks = {cid: 0 for cid in CLIENT_IDS}
    scenario = (f"phase9_delayed_client{args.malicious_client:02d}_p{args.poison_fraction:.4f}"
                f"_start{args.attack_start_round}_s{args.training_seed}_ps{args.poison_seed}_{tag}")
    output_dir = PROJECT_ROOT / "results" / scenario
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains results: {output_dir}. Choose a new --output-tag.")
    output_dir.mkdir(parents=True, exist_ok=True)

    engine.DATA_ROOT = data_root
    configure_binary_training()
    set_reproducible_seed(args.training_seed)

    print("\n" + "=" * 78)
    print("PHASE 9 — DELAYED-ONSET POISONING WITH ONLINE CCTRT")
    print("=" * 78)
    print("Partition:", data_root)
    print("Dirichlet alpha:", partition_metadata.get("dirichlet_alpha"))
    print("Partition seed:", partition_metadata.get("random_seed"))
    print("Training seed:", args.training_seed)
    print("Poison seed:", args.poison_seed)
    print("Malicious client:", args.malicious_client)
    print("Poison fraction of eligible non-Benign records:", f"{args.poison_fraction:.2%}")
    print("Attack start round:", args.attack_start_round)
    print("Warm-up rounds:", args.attack_start_round - 1)
    print(
        "Screening:",
        "CALIBRATED TRFA" if calibration is not None else "DISABLED; all clients aggregated"
    )
    if calibration is not None:
        print("Calibration clean seeds:", calibration["training_seeds"])
        print("Calibration threshold z:", f'{float(calibration["threshold_z"]):.6f}')
        print("Consecutive alerts required:", args.persistence)
    print("Binary mapping: Label_id 0 -> Benign; Label_id 1..6 -> Attack")
    print("Model: 78-128-64-2 MLP; local epochs=2; learning rate=0.015")
    print("Independent test set: NOT USED")

    global_model = IntrusionDetectionMLP(input_features=78, num_classes=2).to(engine.DEVICE)
    validation_X, validation_y, feature_columns = engine.load_validation_data()
    validation_X = np.asarray(validation_X, dtype=np.float32)
    validation_y = to_binary_labels(validation_y)
    validation_loader = engine.create_validation_loader(validation_X, validation_y)

    consistency_tracker = TrainingConsistencyTracker(client_ids=CLIENT_IDS)
    historical_tracker = HistoricalTrustTrackerV2(
        client_ids=CLIENT_IDS,
        retention=HISTORY_RETENTION,
        initial_trust=INITIAL_HISTORICAL_TRUST,
    )
    cctrt_state = OnlineCCTRTState(CLIENT_IDS)

    client_round_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    poison_count = 0
    eligible_count = 0
    selected_client_count = 0

    for round_no in range(1, ROUNDS + 1):
        attack_active = attack_active_this_round(round_no, args.attack_start_round)
        print(f"\nRound {round_no}/{ROUNDS} | attack_active={attack_active}")
        client_states: dict[int, dict] = {}
        client_updates: dict[int, torch.Tensor] = {}
        client_sizes: dict[int, int] = {}
        client_losses: dict[int, float] = {}
        round_poisoned = 0

        for cid in CLIENT_IDS:
            X, original_y, n = engine.load_client_data(cid, feature_columns)
            X = np.asarray(X, dtype=np.float32)
            y = to_binary_labels(original_y)
            if cid == args.malicious_client:
                selected_client_count = int(n)
                if attack_active:
                    y, round_poisoned, eligible_count = poison_eligible_labels(
                        y, cid, args.poison_fraction, args.poison_seed, TARGET_LABEL
                    )
                    poison_count = round_poisoned
            local_model, train_loss = engine.train_local_model(global_model, X, y)
            local_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
            client_updates[cid] = flatten_model_update(local_state, global_model.state_dict())
            client_states[cid] = local_state
            client_sizes[cid] = int(n)
            client_losses[cid] = float(train_loss)
            del X, original_y, y, local_model
            gc.collect()

        # Current-round signals: V2 similarity plus Historical Trust V2 evidence.
        similarity_rows = calculate_update_similarity_v2(client_updates)
        reference_update = calculate_reference_update(client_updates)
        historical_similarity = calculate_update_similarity(client_updates, reference_update)
        consistency_scores = consistency_tracker.score(client_updates)
        previous_history = historical_tracker.get_scores(CLIENT_IDS)
        combined_trust = calculate_combined_trust_v2(
            {cid: historical_similarity[cid] for cid in CLIENT_IDS},
            consistency_scores,
            previous_history,
            alpha=TRUST_ALPHA, beta=TRUST_BETA, gamma=TRUST_GAMMA,
        )
        current_evidence = calculate_current_evidence(
            {cid: historical_similarity[cid] for cid in CLIENT_IDS},
            consistency_scores,
            alpha=TRUST_ALPHA, beta=TRUST_BETA,
        )

        # Online CCTRT is calculated BEFORE advancing to the next round.
        cctrt_rows = cctrt_state.score_round(round_no, similarity_rows, combined_trust)

        # Optional calibrated screening. Calibration profiles are built only
        # from independent all-honest runs; an alert must persist before exclusion.
        calibrated_rows: dict[int, dict[str, Any]] = {}
        if calibration is not None:
            calibrated_rows = score_calibrated_round(
                {
                    cid: {
                        str(calibration["feature"]): float(
                            cctrt_rows[cid][str(calibration["feature"])]
                        )
                    }
                    for cid in CLIENT_IDS
                },
                round_no,
                calibration,
            )
            for cid in CLIENT_IDS:
                alert = bool(calibrated_rows[cid]["Calibrated_Alert"])
                alert_streaks[cid] = alert_streaks[cid] + 1 if alert else 0
                calibrated_rows[cid]["Consecutive_Alert_Rounds"] = int(alert_streaks[cid])
                calibrated_rows[cid]["Rejected_By_Calibrated_TRFA"] = bool(
                    alert_streaks[cid] >= int(args.persistence)
                )

        rejected_clients = (
            [cid for cid in CLIENT_IDS
             if bool(calibrated_rows[cid]["Rejected_By_Calibrated_TRFA"])]
            if calibration is not None else []
        )
        accepted_clients = [cid for cid in CLIENT_IDS if cid not in rejected_clients]
        if len(accepted_clients) < 2:
            raise RuntimeError(
                f"Calibrated screening left only {len(accepted_clients)} accepted clients "
                f"in round {round_no}; aggregation stopped rather than silently reverting to FedAvg."
            )

        aggregated_state, aggregation_weights = aggregate_accepted_models(
            client_states=client_states,
            sample_sizes=client_sizes,
            accepted_client_ids=accepted_clients,
            minimum_accepted_clients=2,
        )
        global_model.load_state_dict(aggregated_state)
        updated_history = historical_tracker.update(current_evidence, round_number=round_no)
        consistency_tracker.update(client_updates)
        metrics = evaluate(global_model, validation_loader)
        val_loss = float(engine.evaluate_validation_loss(global_model, validation_loader))

        for cid in CLIENT_IDS:
            active_for_client = bool(cid == args.malicious_client and attack_active)
            client_round_rows.append({
                "Round": round_no,
                "Client": cid,
                "Is_Malicious": bool(cid == args.malicious_client),
                "Attack_Active": active_for_client,
                "Attack_Start_Round": int(args.attack_start_round),
                "Poisoned_Records": int(round_poisoned if active_for_client else 0),
                "Client_Samples": int(client_sizes[cid]),
                "Client_Training_Loss": float(client_losses[cid]),
                "Update_Similarity_Historical": float(historical_similarity[cid]),
                "Temporal_Consistency": float(consistency_scores[cid]),
                "Historical_Trust_Before": float(previous_history[cid]),
                "Current_Evidence": float(current_evidence[cid]),
                "Combined_Trust": float(combined_trust[cid]),
                "Historical_Trust_After": float(updated_history[cid]),
                "Calibrated_Anomaly_Z": (
                    float(calibrated_rows[cid]["Calibrated_Anomaly_Z"])
                    if calibration is not None else np.nan
                ),
                "Calibrated_Alert": (
                    bool(calibrated_rows[cid]["Calibrated_Alert"])
                    if calibration is not None else False
                ),
                "Consecutive_Alert_Rounds": (
                    int(calibrated_rows[cid]["Consecutive_Alert_Rounds"])
                    if calibration is not None else 0
                ),
                "Rejected_By_Calibrated_TRFA": cid in rejected_clients,
                "Accepted_For_Aggregation": cid in accepted_clients,
                **{k: float(v) for k, v in similarity_rows[cid].items()},
                **cctrt_rows[cid],
            })

        round_rows.append({
            "Round": round_no,
            "Attack_Active": bool(attack_active),
            "Poisoned_Records": int(round_poisoned),
            "Validation_Loss": val_loss,
            **{f"Validation_{k}": v for k, v in metrics.items()},
            "Accepted_Clients": len(accepted_clients),
            "Rejected_Clients": len(rejected_clients),
            "Accepted_Client_IDs": json.dumps(accepted_clients),
            "Rejected_Client_IDs": json.dumps(rejected_clients),
            "Screening_Enabled": calibration is not None,
            "Aggregation_Weight_Sum": float(sum(aggregation_weights.values())),
        })
        torch.save({
            "round": round_no,
            "model_state_dict": global_model.state_dict(),
            "model_config": {"input_features": 78, "num_classes": 2},
            "label_mapping": {"Benign": 0, "Attack": 1},
            "training_seed": int(args.training_seed),
            "historical_trust_state": historical_tracker.state_dict(),
            "temporal_consistency_state": consistency_tracker.state_dict(),
            "attack_active": bool(attack_active),
            "attack_start_round": int(args.attack_start_round),
            "validation_metrics": metrics,
            "validation_loss": val_loss,
        }, output_dir / f"phase9_round_{round_no:02d}.pt")
        print(f"validation_accuracy={metrics['accuracy']:.6f}; macro_f1={metrics['macro_f1']:.6f}; poisoned={round_poisoned}")
        del client_states, client_updates
        gc.collect()

    scores = pd.DataFrame(client_round_rows).sort_values(["Round", "Client"]).reset_index(drop=True)
    rounds_df = pd.DataFrame(round_rows)

    # Per-round ranking/evaluation. Before onset, no ROC-AUC is calculated because
    # all clients are behaving honestly; target ranks are only baseline diagnostics.
    metric_rows: list[dict[str, Any]] = []
    for rnd, group in scores.groupby("Round", sort=True):
        group = group.sort_values("Client").copy()
        target_row = group[group["Client"] == args.malicious_client].iloc[0]
        ranked = group.sort_values(["CCTRT_Causal_Score", "Client"], ascending=[False, True]).reset_index(drop=True)
        target_rank = int(ranked.index[ranked["Client"] == args.malicious_client][0]) + 1
        active = bool(int(rnd) >= args.attack_start_round)
        row = {
            "Round": int(rnd),
            "Attack_Active": active,
            "Target_Client": int(args.malicious_client),
            "Target_CCTRT_Score": float(target_row["CCTRT_Causal_Score"]),
            "Target_Rank_Most_Suspicious_First": target_rank,
            "Target_Top1": target_rank == 1,
            "Target_Top3": target_rank <= 3,
            "Honest_Mean_CCTRT_Score": float(group[group["Client"] != args.malicious_client]["CCTRT_Causal_Score"].mean()),
            "Target_Calibrated_Anomaly_Z": (
                float(target_row["Calibrated_Anomaly_Z"]) if calibration is not None else np.nan
            ),
            "Target_Calibrated_Alert": (
                bool(target_row["Calibrated_Alert"]) if calibration is not None else False
            ),
            "Target_Rejected": (
                bool(target_row["Rejected_By_Calibrated_TRFA"]) if calibration is not None else False
            ),
            "Honest_Excluded_Count": int(
                group[
                    (group["Client"] != args.malicious_client)
                    & (group["Rejected_By_Calibrated_TRFA"] == True)
                ].shape[0]
            ),
        }
        if active:
            y_true = group["Client"].eq(args.malicious_client).astype(int).to_numpy()
            y_score = group["CCTRT_Causal_Score"].astype(float).to_numpy()
            row["ROC_AUC"] = float(roc_auc_score(y_true, y_score))
            row["Average_Precision"] = float(average_precision_score(y_true, y_score))
            if calibration is not None:
                calibrated_score = group["Calibrated_Anomaly_Z"].astype(float).to_numpy()
                row["Calibrated_ROC_AUC"] = float(roc_auc_score(y_true, calibrated_score))
                row["Calibrated_Average_Precision"] = float(
                    average_precision_score(y_true, calibrated_score)
                )
            else:
                row["Calibrated_ROC_AUC"] = np.nan
                row["Calibrated_Average_Precision"] = np.nan
        else:
            row["ROC_AUC"] = np.nan
            row["Average_Precision"] = np.nan
            row["Calibrated_ROC_AUC"] = np.nan
            row["Calibrated_Average_Precision"] = np.nan
        metric_rows.append(row)

    scores_path = output_dir / "phase9_client_round_scores.csv"
    rounds_path = output_dir / "phase9_round_metrics.csv"
    metrics_path = output_dir / "phase9_online_detection_metrics.csv"
    scores.to_csv(scores_path, index=False)
    rounds_df.to_csv(rounds_path, index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(metrics_path, index=False)

    active_metrics = metrics_df[metrics_df["Attack_Active"]]
    pre_metrics = metrics_df[~metrics_df["Attack_Active"]]
    metadata = {
        "phase": "Phase 9 delayed-onset poisoning",
        "method": "Online/Causal CCTRT V1",
        "dataset": "CIC-IDS-2017",
        "classification_task": "binary_benign_vs_attack",
        "label_mapping": {"Benign": 0, "Attack": 1},
        "model_config": {"input_features": 78, "num_classes": 2},
        "local_epochs": engine.LOCAL_EPOCHS,
        "learning_rate": engine.LEARNING_RATE,
        "batch_size": engine.BATCH_SIZE,
        "partition_root": str(data_root),
        "dirichlet_alpha": float(partition_metadata["dirichlet_alpha"]),
        "partition_seed": partition_metadata.get("random_seed"),
        "training_seed": int(args.training_seed),
        "poison_seed": int(args.poison_seed),
        "clients": 10,
        "rounds": ROUNDS,
        "malicious_client": int(args.malicious_client),
        "attack_start_round": int(args.attack_start_round),
        "warmup_rounds": int(args.attack_start_round - 1),
        "poison_fraction_of_eligible_nonbenign": float(args.poison_fraction),
        "poisoned_records_per_active_round": int(poison_count),
        "eligible_nonbenign_records": int(eligible_count),
        "malicious_client_records": int(selected_client_count),
        "actual_fraction_of_eligible": float(poison_count / eligible_count) if eligible_count else 0.0,
        "actual_fraction_of_client": float(poison_count / selected_client_count) if selected_client_count else 0.0,
        "active_rounds": list(range(args.attack_start_round, ROUNDS + 1)),
        "mean_active_round_roc_auc": float(active_metrics["ROC_AUC"].mean()) if not active_metrics.empty else None,
        "active_round_top3_rate": float(active_metrics["Target_Top3"].mean()) if not active_metrics.empty else None,
        "pre_attack_target_top3_rounds": int(pre_metrics["Target_Top3"].sum()),
        "screening_enabled": calibration is not None,
        "screening_method": "clean-reference CCTRT calibrated threshold with persistence" if calibration is not None else "none",
        "calibration_training_seeds": calibration["training_seeds"] if calibration is not None else [],
        "calibration_threshold_z": float(calibration["threshold_z"]) if calibration is not None else None,
        "calibration_quantile": float(calibration["quantile"]) if calibration is not None else None,
        "persistence_rounds": int(args.persistence) if calibration is not None else None,
        "all_clients_aggregated": calibration is None,
        "independent_test_set_used": False,
        "cavc_used_in_online_score": False,
        "causality": "Each round score is calculated before next round using current round and prior client observations only.",
        "limitations": [
            "One malicious client and one delayed-onset schedule per run.",
            "Percentile scores are cohort-relative and are not calibrated attack probabilities.",
            "CAVC is excluded because the existing CAVC V2 tool is a frozen-checkpoint local-candidate diagnostic, not a per-round online signal.",
            "The same non-IID partition is retained; this is a delayed-onset test, not partition generalization.",
            "Server-validation metrics are development diagnostics; this run does not use the reserved test split. That split was inspected in earlier seven-class project analyses and must not be described as historically unseen.",
        ],
        "files": {
            "client_round_scores": scores_path.name,
            "round_metrics": rounds_path.name,
            "online_detection_metrics": metrics_path.name,
        },
    }
    with (output_dir / "phase9_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summary = [
        "PHASE 9 — BINARY DELAYED-ONSET ONLINE/Causal CCTRT",
        "============================================",
        f"Malicious client: {args.malicious_client}",
        f"Attack start: round {args.attack_start_round}",
        f"Warm-up rounds: {args.attack_start_round - 1}",
        f"Poison fraction of eligible non-Benign records: {args.poison_fraction:.6%}",
        f"Poisoned records per active round: {poison_count}",
        f"Active rounds: {args.attack_start_round}-{ROUNDS}",
        f"Mean active-round ROC-AUC: {metadata['mean_active_round_roc_auc']}",
        f"Active-round top-3 rate: {metadata['active_round_top3_rate']}",
        f"Pre-attack target top-3 rounds: {metadata['pre_attack_target_top3_rounds']} of {len(pre_metrics)}",
        "",
        (
            "Calibrated screening enabled; accepted-only sample-size-weighted FedAvg used."
            if calibration is not None else
            "No screening; all clients aggregated."
        ),
        "Independent test set not used.",
        "ROC-AUC is reported only after attack onset. Before onset, all clients behave honestly.",
    ]
    (output_dir / "phase9_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "=" * 78)
    print("PHASE 9 DELAYED-ONSET EXPERIMENT COMPLETED")
    print("=" * 78)
    print("Attack begins at round:", args.attack_start_round)
    print("Warm-up rounds:", args.attack_start_round - 1)
    print("Poisoned records per active round:", poison_count)
    print("Actual fraction of eligible:", f"{(poison_count / eligible_count):.6%}" if eligible_count else "0.000000%")
    print("Mean ROC-AUC after onset:", f"{metadata['mean_active_round_roc_auc']:.6f}" if metadata["mean_active_round_roc_auc"] is not None else "N/A")
    print("Top-3 rate after onset:", f"{metadata['active_round_top3_rate']:.6f}" if metadata["active_round_top3_rate"] is not None else "N/A")
    print("Pre-attack target top-3 rounds:", f"{metadata['pre_attack_target_top3_rounds']}/{len(pre_metrics)}")
    print("Output directory:", output_dir)
    print("Screening enabled:", "YES" if calibration is not None else "NO")
    print("Independent test used: NO")

    del validation_X, validation_y, validation_loader, global_model
    gc.collect()
    return output_dir


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"\nPHASE 9 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
