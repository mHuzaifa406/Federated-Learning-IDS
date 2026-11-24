"""Controlled non-IID experiment for Historical Trust V2.

This experiment isolates the Historical Trust V2 update lifecycle while
keeping the existing update-similarity calculation and temporal consistency
signal unchanged. Screening is intentionally disabled (threshold=0.0), so all
clients are aggregated. It is a trust-tracker integration/calibration run,
not evidence that the current signals detect poisoning.

Run from project root:
    python scripts\\run_historical_trust_v2_noniid.py --condition clean
    python scripts\\run_historical_trust_v2_noniid.py --condition poison-client07

The poisoned condition reproduces the matched Client 07 label-flipping setup:
1,897 records (approximately 4.055672% of Client 07's records) are changed to
Benign using the same deterministic selection rule used in the previous
matched-poison experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import scripts.federated_engine as engine
from models.ids_model import IntrusionDetectionMLP
from trfa.trust_metrics import (
    flatten_model_update,
    calculate_reference_update,
    calculate_update_similarity,
    TrainingConsistencyTracker,
)
from trfa.historical_trust_v2 import (
    HistoricalTrustTrackerV2,
    calculate_current_evidence,
    calculate_combined_trust_v2,
)
from trfa.aggregation import aggregate_accepted_models


# ---------------------------------------------------------------------------
# Fixed experiment configuration (matched to existing non-IID experiments)
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "federated_noniid_alpha05_seed43"
OUTPUT_ROOT = PROJECT_ROOT / "results"
TOTAL_ROUNDS = 10
NUM_CLIENTS = 10
CLIENT_IDS = list(range(1, NUM_CLIENTS + 1))
TRAINING_SEED = 43
PARTITION_SEED = 43
DIRICHLET_ALPHA = 0.5

ALPHA = 1.0 / 3.0
BETA = 1.0 / 3.0
GAMMA = 1.0 / 3.0
HISTORY_RETENTION = 0.90
INITIAL_HISTORICAL_TRUST = 0.50

# All-client calibration only. Do not interpret threshold 0.0 as a useful
# screening threshold; it deliberately ensures that no client is excluded.
CALIBRATION_THRESHOLD = 0.0

# Matched Client 07 poisoning configuration.
MALICIOUS_CLIENT = 7
TARGET_LABEL = 0  # Benign
POISON_SEED = 2027
POISON_CLIENT_FRACTION = 8676 / 213858


class ExperimentError(RuntimeError):
    """Raised when experiment inputs or outputs are inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Historical Trust V2 non-IID calibration experiment."
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=("clean", "poison-client07"),
        help="All-honest condition or matched Client 07 label-flipping condition.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional unique suffix for a repeat run; existing output is never overwritten.",
    )
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    engine.set_random_seed()
    random_seed = int(seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.set_num_threads(engine.CPU_THREADS)


def load_and_validate_partition() -> dict:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Non-IID partition not found: {DATA_ROOT}")

    metadata_path = DATA_ROOT / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Partition metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not math.isclose(float(metadata.get("dirichlet_alpha", -1)), DIRICHLET_ALPHA):
        raise ExperimentError("Partition Dirichlet alpha does not match 0.5.")
    if int(metadata.get("random_seed", -1)) != PARTITION_SEED:
        raise ExperimentError("Partition seed does not match 43.")
    return metadata


def make_poisoned_labels(original_y: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply the matched Client 07 label flip deterministically."""
    y = np.array(original_y, dtype=np.int64, copy=True)
    eligible = np.flatnonzero(y != TARGET_LABEL)
    poison_count = int(np.floor(len(y) * POISON_CLIENT_FRACTION))

    if poison_count > len(eligible):
        raise ExperimentError("Not enough eligible non-Benign records to poison.")

    rng = np.random.default_rng(POISON_SEED + MALICIOUS_CLIENT)
    selected = rng.choice(eligible, size=poison_count, replace=False)
    y[selected] = TARGET_LABEL

    actual = int(np.count_nonzero(y != original_y))
    if actual != poison_count:
        raise ExperimentError(
            f"Expected {poison_count} changed labels, observed {actual}."
        )
    return y, actual


def evaluate_global_model(model: torch.nn.Module, loader) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            logits = model(batch_X.to(engine.DEVICE))
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(batch_y.detach().cpu().numpy().astype(int).tolist())
            y_pred.extend(predictions.detach().cpu().numpy().astype(int).tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def run_experiment(condition: str, output_tag: str = "") -> Path:
    metadata = load_and_validate_partition()
    poison_enabled = condition == "poison-client07"

    condition_name = (
        "historical_trust_v2_clean_noniid_seed43"
        if not poison_enabled
        else "historical_trust_v2_poison_client07_noniid_seed43"
    )
    if output_tag:
        safe_tag = "".join(ch for ch in output_tag if ch.isalnum() or ch in "-_")
        if not safe_tag:
            raise ValueError("--output-tag must contain at least one safe character.")
        condition_name += f"_{safe_tag}"

    output_dir = OUTPUT_ROOT / condition_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains results: {output_dir}\n"
            "Use a new --output-tag. Existing results will not be overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    engine.DATA_ROOT = DATA_ROOT
    set_reproducible_seed(TRAINING_SEED)

    print("\n" + "=" * 78)
    print("HISTORICAL TRUST V2 — NON-IID CALIBRATION")
    print("=" * 78)
    print("Condition:", condition)
    print("Partition:", DATA_ROOT)
    print("Dirichlet alpha:", DIRICHLET_ALPHA)
    print("Partition seed:", PARTITION_SEED)
    print("Training seed:", TRAINING_SEED)
    print("Communication rounds:", TOTAL_ROUNDS)
    print("Historical retention:", HISTORY_RETENTION)
    print("Initial historical trust:", INITIAL_HISTORICAL_TRUST)
    print("Screening threshold:", CALIBRATION_THRESHOLD, "(screening deliberately disabled)")
    print("All clients honest:", "NO" if poison_enabled else "YES")

    print("\nInitializing global model...")
    global_model = IntrusionDetectionMLP().to(engine.DEVICE)

    print("Loading clean server-validation data...")
    validation_X, validation_y, feature_columns = engine.load_validation_data()
    validation_X = np.array(validation_X, dtype=np.float32, copy=True)
    validation_y = np.array(validation_y, dtype=np.int64, copy=True)
    validation_loader = engine.create_validation_loader(validation_X, validation_y)

    consistency_tracker = TrainingConsistencyTracker(client_ids=CLIENT_IDS)
    historical_tracker = HistoricalTrustTrackerV2(
        client_ids=CLIENT_IDS,
        retention=HISTORY_RETENTION,
        initial_trust=INITIAL_HISTORICAL_TRUST,
    )

    trust_rows: list[dict] = []
    round_rows: list[dict] = []
    poison_count_total = 0

    for round_number in range(1, TOTAL_ROUNDS + 1):
        print("\n" + "-" * 78)
        print(f"COMMUNICATION ROUND {round_number}/{TOTAL_ROUNDS}")
        print("-" * 78)

        client_states: dict[int, dict] = {}
        client_updates: dict[int, torch.Tensor] = {}
        client_sizes: dict[int, int] = {}
        client_losses: dict[int, float] = {}
        round_poison_count = 0

        for client_id in CLIENT_IDS:
            X, y, sample_count = engine.load_client_data(client_id, feature_columns)
            X = np.array(X, dtype=np.float32, copy=True)
            y = np.array(y, dtype=np.int64, copy=True)

            if poison_enabled and client_id == MALICIOUS_CLIENT:
                y, changed = make_poisoned_labels(y)
                round_poison_count = changed

            local_model, training_loss = engine.train_local_model(global_model, X, y)
            local_state = {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }
            update_vector = flatten_model_update(local_state, global_model.state_dict())

            client_states[client_id] = local_state
            client_updates[client_id] = update_vector
            client_sizes[client_id] = int(sample_count)
            client_losses[client_id] = float(training_loss)

            del X, y, local_model
            gc.collect()

        poison_count_total = max(poison_count_total, round_poison_count)

        # Signal 1: current-round directional similarity to coordinate median.
        reference_update = calculate_reference_update(client_updates)
        similarity_scores = calculate_update_similarity(client_updates, reference_update)

        # Signal 2: provisional temporal consistency. It is retained as the
        # currently tested C signal, not asserted to be a reliable attack detector.
        consistency_scores = consistency_tracker.score(client_updates)

        # Read H^(r-1), calculate current trust, then aggregate.
        previous_history = historical_tracker.get_scores(CLIENT_IDS)
        combined_scores = calculate_combined_trust_v2(
            similarity_scores,
            consistency_scores,
            previous_history,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA,
        )

        accepted = [cid for cid in CLIENT_IDS if combined_scores[cid] >= CALIBRATION_THRESHOLD]
        excluded = [cid for cid in CLIENT_IDS if cid not in accepted]
        if excluded:
            raise ExperimentError(
                "Calibration threshold unexpectedly excluded clients. "
                "This experiment is intended to aggregate all clients."
            )

        aggregated_state, aggregation_weights = aggregate_accepted_models(
            client_states=client_states,
            sample_sizes=client_sizes,
            accepted_client_ids=accepted,
            minimum_accepted_clients=2,
        )
        global_model.load_state_dict(aggregated_state)

        # Update H only from current-round evidence, not combined trust.
        current_evidence = calculate_current_evidence(
            similarity_scores,
            consistency_scores,
            alpha=ALPHA,
            beta=BETA,
        )
        updated_history = historical_tracker.update(
            current_evidence,
            round_number=round_number,
        )

        # Store current updates only after this round's temporal score is computed.
        consistency_tracker.update(client_updates)

        metrics = evaluate_global_model(global_model, validation_loader)
        validation_loss = float(engine.evaluate_validation_loss(global_model, validation_loader))

        for cid in CLIENT_IDS:
            trust_rows.append({
                "Round": round_number,
                "Client": cid,
                "Is_Malicious": bool(poison_enabled and cid == MALICIOUS_CLIENT),
                "Update_Similarity": float(similarity_scores[cid]),
                "Temporal_Consistency": float(consistency_scores[cid]),
                "Historical_Trust_Before": float(previous_history[cid]),
                "Current_Evidence": float(current_evidence[cid]),
                "Combined_Trust": float(combined_scores[cid]),
                "Historical_Trust_After": float(updated_history[cid]),
                "Client_Training_Loss": float(client_losses[cid]),
                "Client_Samples": int(client_sizes[cid]),
                "Accepted": cid in accepted,
                "Poisoned_Records": int(round_poison_count if cid == MALICIOUS_CLIENT and poison_enabled else 0),
            })

        round_rows.append({
            "Round": round_number,
            "Validation_Loss": validation_loss,
            **{f"Validation_{k}": v for k, v in metrics.items()},
            "Accepted_Clients": len(accepted),
            "Excluded_Clients": len(excluded),
            "Poisoned_Records": int(round_poison_count),
            "Aggregation_Weight_Sum": float(sum(aggregation_weights.values())),
        })

        checkpoint_path = output_dir / f"historical_v2_round_{round_number:02d}.pt"
        torch.save({
            "round": round_number,
            "condition": condition,
            "model_state_dict": global_model.state_dict(),
            "historical_trust_state": historical_tracker.state_dict(),
            "temporal_consistency_state": consistency_tracker.state_dict(),
            "validation_loss": validation_loss,
            "validation_metrics": metrics,
            "accepted_clients": accepted,
            "aggregation_weights": aggregation_weights,
        }, checkpoint_path)

        print(
            f"Round {round_number}: accuracy={metrics['accuracy']:.6f}, "
            f"macro_f1={metrics['macro_f1']:.6f}, "
            f"accepted={len(accepted)}/{NUM_CLIENTS}"
        )

        del client_states, client_updates
        gc.collect()

    trust_df = pd.DataFrame(trust_rows)
    round_df = pd.DataFrame(round_rows)
    client_summary = (
        trust_df.groupby(["Client", "Is_Malicious"], as_index=False)
        .agg(
            Similarity_Mean=("Update_Similarity", "mean"),
            Temporal_Consistency_Mean=("Temporal_Consistency", "mean"),
            Historical_Trust_Before_Mean=("Historical_Trust_Before", "mean"),
            Current_Evidence_Mean=("Current_Evidence", "mean"),
            Combined_Trust_Mean=("Combined_Trust", "mean"),
            Combined_Trust_Std=("Combined_Trust", "std"),
            Accepted_Rounds=("Accepted", "sum"),
        )
    )

    trust_path = output_dir / "historical_v2_client_round_trust.csv"
    round_path = output_dir / "historical_v2_round_metrics.csv"
    client_path = output_dir / "historical_v2_client_summary.csv"
    trust_df.to_csv(trust_path, index=False)
    round_df.to_csv(round_path, index=False)
    client_summary.to_csv(client_path, index=False)

    summary = {
        "experiment": "historical_trust_v2_noniid_calibration",
        "condition": condition,
        "dataset": "CIC-IDS-2017",
        "partition_method": "class-wise Dirichlet",
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "partition_seed": PARTITION_SEED,
        "training_seed": TRAINING_SEED,
        "clients": NUM_CLIENTS,
        "rounds": TOTAL_ROUNDS,
        "poisoning_enabled": poison_enabled,
        "malicious_client": MALICIOUS_CLIENT if poison_enabled else None,
        "target_label": TARGET_LABEL if poison_enabled else None,
        "poisoned_records_per_round": poison_count_total,
        "poison_fraction_of_client": (poison_count_total / client_sizes[MALICIOUS_CLIENT]) if poison_enabled else 0.0,
        "screening_enabled": False,
        "calibration_threshold": CALIBRATION_THRESHOLD,
        "aggregation": "sample-size-weighted FedAvg over all clients",
        "consistency_signal": "temporal update consistency (provisional diagnostic)",
        "historical_trust_version": "2.0",
        "historical_retention": HISTORY_RETENTION,
        "initial_historical_trust": INITIAL_HISTORICAL_TRUST,
        "trust_weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
        "test_set_used": False,
        "final_metrics": round_rows[-1],
        "files": {
            "client_round_trust": trust_path.name,
            "round_metrics": round_path.name,
            "client_summary": client_path.name,
        },
        "interpretation_limitations": [
            "Screening was disabled; this experiment does not measure detection rate or false-exclusion rate.",
            "Temporal consistency is provisional and can also be high for persistent malicious clients.",
            "No independent test data was used.",
            "This run integrates and tests the Historical Trust V2 update lifecycle, not the final TRFA methodology.",
        ],
    }
    summary_path = output_dir / "historical_v2_experiment_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("HISTORICAL TRUST V2 NON-IID EXPERIMENT COMPLETED")
    print("=" * 78)
    print("Condition:", condition)
    print("Final validation accuracy:", f"{round_rows[-1]['Validation_accuracy']:.4%}")
    print("Final validation macro F1:", f"{round_rows[-1]['Validation_macro_f1']:.6f}")
    print("Final validation weighted F1:", f"{round_rows[-1]['Validation_weighted_f1']:.6f}")
    print("Poisoned records per round:", poison_count_total)
    print("Screening enabled: NO")
    print("Independent test set used: NO")
    print("Output directory:", output_dir)
    print("Trust history:", trust_path)
    print("Round metrics:", round_path)
    print("Client summary:", client_path)
    print("Summary:", summary_path)
    print("\nIMPORTANT: This run validates V2 history integration, not TRFA detection effectiveness.")

    del validation_X, validation_y, validation_loader, global_model
    gc.collect()
    return output_dir


def main() -> int:
    args = parse_args()
    try:
        run_experiment(args.condition, args.output_tag)
    except Exception as exc:
        print(f"\nEXPERIMENT FAILED: {type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
