
"""
Clean non-IID temporal-consistency calibration — TRFA V2

This is a NEW controlled experiment. It does not overwrite Scenario H/G.
All ten clients are honest and use the existing class-wise Dirichlet
partition (alpha=0.5, partition seed=43).

Purpose:
1. Measure each honest client's temporal update consistency.
2. Compare the new temporal score with the legacy validation-loss score.
3. Keep screening disabled (threshold=0.0) so every client participates;
   this run calibrates the signal and is NOT a threshold/defense evaluation.

The historical-trust EMA remains unchanged in this experiment and will be
revised separately after the temporal signal is inspected.
"""

from pathlib import Path
import sys
import json
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score

import scripts.federated_engine as engine

from models.ids_model import IntrusionDetectionMLP

from trfa.trust_metrics import (
    flatten_model_update,
    calculate_reference_update,
    calculate_update_similarity,
    TrainingConsistencyTracker,
    calculate_training_consistency,  # legacy score, comparison only
    HistoricalTrustTracker,
    calculate_combined_trust,
    screen_clients,
)

from trfa.aggregation import aggregate_accepted_models


# ============================================================
# 1. CONFIGURATION
# ============================================================

NONIID_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "federated_noniid_alpha05_seed43"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "scenario_noniid_temporal_consistency_v2_seed43"
)

TOTAL_ROUNDS = 10
NUM_CLIENTS = 10

TRUST_THRESHOLD = 0.0

ALPHA = 1.0 / 3.0
BETA = 1.0 / 3.0
GAMMA = 1.0 / 3.0

HISTORY_RETENTION = 0.90
INITIAL_HISTORICAL_TRUST = 0.50

EXPERIMENT_SEED = 43


# ============================================================
# 2. VALIDATION METRICS
# ============================================================

def evaluate_validation_metrics(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch_X, batch_y in loader:

            outputs = model(batch_X.to(engine.DEVICE))

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(batch_y.numpy().tolist())

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    return {
        "accuracy": float(
            accuracy_score(all_labels, all_predictions)
        ),
        "macro_f1": float(
            f1_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0
            )
        )
    }


# ============================================================
# 3. MAIN EXPERIMENT
# ============================================================

def run_clean_noniid_temporal_consistency_v2():

    print("\n" + "=" * 78)
    print("CLEAN NON-IID TEMPORAL-CONSISTENCY CALIBRATION — TRFA V2")
    print("=" * 78)

    if not NONIID_DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Non-IID dataset not found: {NONIID_DATA_ROOT}"
        )

    metadata_path = NONIID_DATA_ROOT / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Non-IID partition metadata is missing."
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:
        partition_metadata = json.load(file)

    if partition_metadata["dirichlet_alpha"] != 0.5:
        raise ValueError("Unexpected Dirichlet alpha.")

    if partition_metadata["random_seed"] != 43:
        raise ValueError("Unexpected partition seed.")

    # Point the existing data-loading function to the
    # new non-IID directory. The original dataset is untouched.
    engine.DATA_ROOT = NONIID_DATA_ROOT

    # Reproducible model initialization and local training.
    engine.set_random_seed()

    np.random.seed(EXPERIMENT_SEED)
    torch.manual_seed(EXPERIMENT_SEED)
    torch.set_num_threads(engine.CPU_THREADS)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Non-IID data:", NONIID_DATA_ROOT)
    print("Dirichlet alpha:", partition_metadata["dirichlet_alpha"])
    print("Partition seed:", partition_metadata["random_seed"])
    print("Training seed:", EXPERIMENT_SEED)
    print("Communication rounds:", TOTAL_ROUNDS)
    print("Trust threshold:", TRUST_THRESHOLD)
    print("Screening enabled: NO (threshold 0.0 accepts all valid scores)")
    print("All clients are honest: YES")
    print("Consistency method: previous-round update direction + magnitude")
    print("Legacy validation-loss score: comparison only; not used in trust")

    # --------------------------------------------------------
    # Initialize global model and clean validation data
    # --------------------------------------------------------

    print("\nInitializing global model...")

    global_model = IntrusionDetectionMLP().to(engine.DEVICE)

    print("\nLoading unchanged server-validation data...")

    validation_X, validation_y, feature_columns = (
        engine.load_validation_data()
    )

    validation_loader = engine.create_validation_loader(
        validation_X,
        validation_y
    )

    # --------------------------------------------------------
    # Initialize historical trust
    # --------------------------------------------------------

    client_ids = list(range(1, NUM_CLIENTS + 1))

    tracker = HistoricalTrustTracker(
        client_ids=client_ids,
        retention=HISTORY_RETENTION,
        initial_trust=INITIAL_HISTORICAL_TRUST
    )

    # New stateful temporal-consistency tracker. For each round, call score()
    # first, then update() with this round's updates after scoring.
    consistency_tracker = TrainingConsistencyTracker(
        client_ids=client_ids,
        direction_weight=0.50,
        magnitude_weight=0.50,
        initial_score=0.50
    )

    round_results = []
    trust_history = []

    # --------------------------------------------------------
    # Communication rounds
    # --------------------------------------------------------

    for round_number in range(1, TOTAL_ROUNDS + 1):

        print("\n" + "=" * 70)
        print(
            f"NON-IID COMMUNICATION ROUND "
            f"{round_number}/{TOTAL_ROUNDS}"
        )
        print("=" * 70)

        client_states = {}
        client_updates = {}
        client_sizes = {}
        client_training_losses = {}
        candidate_validation_losses = {}

        # ----------------------------------------------------
        # STEP 1: Train all honest clients
        # ----------------------------------------------------

        print("\n[1/7] Training honest non-IID clients...")

        for client_id in client_ids:

            print(f"\nTraining client {client_id:02d}...")

            X, y, sample_count = engine.load_client_data(
                client_id,
                feature_columns
            )

            # No poisoning is applied.
            local_model, training_loss = engine.train_local_model(
                global_model,
                X,
                y
            )

            local_state = {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }

            candidate_loss = engine.evaluate_validation_loss(
                local_model,
                validation_loader
            )

            update_vector = flatten_model_update(
                local_state,
                global_model.state_dict()
            )

            client_states[client_id] = local_state
            client_updates[client_id] = update_vector

            client_sizes[client_id] = int(sample_count)

            client_training_losses[client_id] = float(
                training_loss
            )

            candidate_validation_losses[client_id] = float(
                candidate_loss
            )

            print("  Training records:", f"{sample_count:,}")
            print("  Training loss:", f"{training_loss:.6f}")
            print("  Candidate validation loss:", f"{candidate_loss:.6f}")

            del X, y, local_model
            gc.collect()

        if set(client_updates) != set(client_ids):
            raise RuntimeError(
                "The current round does not contain updates from all configured clients."
            )

        # ----------------------------------------------------
        # STEP 2: Update similarity
        # ----------------------------------------------------

        print("\n[2/7] Calculating update similarity...")

        reference_update = calculate_reference_update(
            client_updates
        )

        similarity_scores = calculate_update_similarity(
            client_updates,
            reference_update
        )

        # ----------------------------------------------------
        # STEP 3: Training consistency
        # ----------------------------------------------------

        print("\n[3/7] Calculating temporal training consistency...")

        # The new score compares each client's update with its own previous-
        # round update. score() MUST be called before update() in each round.
        consistency_scores = consistency_tracker.score(client_updates)

        # Keep the old validation-loss score only as a diagnostic comparator.
        # It does not contribute to the V2 combined trust score.
        baseline_validation_loss = engine.evaluate_validation_loss(
            global_model,
            validation_loader
        )
        legacy_consistency_scores = calculate_training_consistency(
            baseline_validation_loss,
            candidate_validation_losses
        )

        # ----------------------------------------------------
        # STEP 4: Historical and combined trust
        # ----------------------------------------------------

        print("\n[4/7] Calculating combined trust...")

        historical_scores = tracker.get_scores(client_ids)

        combined_scores = calculate_combined_trust(
            similarity_scores=similarity_scores,
            consistency_scores=consistency_scores,
            historical_scores=historical_scores,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA
        )

        print("\nTRUST SCORE TABLE")
        print(
            "{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
                "Client", "Similarity", "Temporal", "LegacyVal",
                "History", "Combined"
            )
        )

        for client_id in client_ids:
            print(
                "{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                    f"Client {client_id:02d}",
                    similarity_scores[client_id],
                    consistency_scores[client_id],
                    legacy_consistency_scores[client_id],
                    historical_scores[client_id],
                    combined_scores[client_id]
                )
            )

        # ----------------------------------------------------
        # STEP 5: Screen clients
        # ----------------------------------------------------

        print("\n[5/7] Applying the candidate trust threshold...")

        accepted, excluded = screen_clients(
            combined_scores,
            threshold=TRUST_THRESHOLD
        )

        # This is a signal-calibration run, not a screening evaluation.
        # Threshold=0.0 is deliberately used to retain every client.
        if excluded or set(accepted) != set(client_ids):
            raise RuntimeError(
                "Calibration run unexpectedly excluded a client. "
                "Screening is intended to be disabled at threshold 0.0."
            )

        print("Accepted clients:", accepted)
        print("Excluded honest clients:", excluded)
        print("Note: zero exclusions are guaranteed by calibration mode; "
              "do not interpret this as measured screening performance.")

        # ----------------------------------------------------
        # STEP 6: Aggregate accepted client models
        # ----------------------------------------------------

        print("\n[6/7] Aggregating accepted client models...")

        if len(accepted) < 2:
            raise RuntimeError(
                "Fewer than two clients passed screening. "
                "Aggregation stopped."
            )

        aggregated_state, aggregation_weights = (
            aggregate_accepted_models(
                client_states=client_states,
                sample_sizes=client_sizes,
                accepted_client_ids=accepted,
                minimum_accepted_clients=2
            )
        )

        global_model.load_state_dict(aggregated_state)

        print("\nAggregation weights:")

        for client_id, weight in aggregation_weights.items():

            print(
                f"Client {client_id:02d}: {weight:.6f}"
            )

        print(
            "Total aggregation weight:",
            f"{sum(aggregation_weights.values()):.6f}"
        )

        # Store current client updates only AFTER computing this round's
        # temporal consistency scores. They become the previous-round
        # reference for the next communication round.
        consistency_tracker.update(client_updates)

        # Existing historical-trust update is intentionally retained for this
        # isolated experiment. Its feedback-loop redesign is a separate step.
        tracker.update(combined_scores)

        updated_historical_scores = tracker.get_scores(client_ids)

        # ----------------------------------------------------
        # STEP 7: Evaluate global model
        # ----------------------------------------------------

        print("\n[7/7] Evaluating global model...")

        validation_loss = engine.evaluate_validation_loss(
            global_model,
            validation_loader
        )

        metrics = evaluate_validation_metrics(
            global_model,
            validation_loader
        )

        print("\nROUND RESULTS")

        print(
            "Validation loss:",
            f"{validation_loss:.6f}"
        )

        print(
            "Validation accuracy:",
            f"{metrics['accuracy'] * 100:.4f}%"
        )

        print(
            "Validation macro F1:",
            f"{metrics['macro_f1']:.6f}"
        )

        print(
            "Validation weighted F1:",
            f"{metrics['weighted_f1']:.6f}"
        )

        # ----------------------------------------------------
        # Record detailed client trust information
        # ----------------------------------------------------

        for client_id in client_ids:

            trust_history.append({
                "Round": round_number,
                "Client": client_id,
                "Update_Similarity": similarity_scores[client_id],
                "Training_Consistency_Temporal": consistency_scores[client_id],
                "Training_Consistency_Legacy_Validation": legacy_consistency_scores[client_id],
                "Client_Update_Norm": float(torch.linalg.vector_norm(client_updates[client_id]).item()),
                "Historical_Trust_Before": historical_scores[client_id],
                "Combined_Trust": combined_scores[client_id],
                "Historical_Trust_After": updated_historical_scores[client_id],
                "Client_Training_Loss": client_training_losses[client_id],
                "Candidate_Validation_Loss": candidate_validation_losses[client_id],
                "Client_Samples": client_sizes[client_id],
                "Accepted": client_id in accepted,
                "False_Exclusion": client_id in excluded
            })

        round_result = {
            "round": round_number,
            "validation_loss": float(validation_loss),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_weighted_f1": metrics["weighted_f1"],
            "accepted_clients": accepted,
            "excluded_honest_clients": excluded,
            "client_sizes": client_sizes,
            "similarity_scores": similarity_scores,
            "temporal_consistency_scores": consistency_scores,
            "legacy_validation_consistency_scores": legacy_consistency_scores,
            "historical_scores_before_update": historical_scores,
            "combined_trust_scores": combined_scores,
            "aggregation_weights": aggregation_weights
        }

        round_results.append(round_result)

        # ----------------------------------------------------
        # Save round checkpoint
        # ----------------------------------------------------

        checkpoint_path = (
            OUTPUT_DIR /
            f"clean_noniid_round_{round_number:02d}.pt"
        )

        torch.save(
            {
                "round": round_number,
                "model_state_dict": global_model.state_dict(),
                "validation_loss": validation_loss,
                "validation_metrics": metrics,
                "accepted_clients": accepted,
                "excluded_honest_clients": excluded,
                "aggregation_weights": aggregation_weights,
                "temporal_consistency_tracker_state": consistency_tracker.state_dict(),
                "historical_trust_after_update": updated_historical_scores,
                "experiment_seed": EXPERIMENT_SEED,
                "consistency_method": "temporal_direction_and_magnitude_v1"
            },
            checkpoint_path
        )

        print("Checkpoint saved:", checkpoint_path)

        del client_states, client_updates
        gc.collect()

    # ========================================================
    # SAVE COMPLETE EXPERIMENT RESULTS
    # ========================================================

    total_honest_opportunities = NUM_CLIENTS * TOTAL_ROUNDS

    # The threshold is zero and screening is disabled in this calibration run.
    # A false-exclusion rate is therefore NOT an evaluated metric here.
    summary = {
        "scenario": "noniid_temporal_consistency_v2_seed43",
        "script_version": "1.0",
        "purpose": "Clean non-IID calibration of temporal consistency; screening disabled.",
        "dataset": "CIC-IDS-2017",
        "partition_method": "Class-wise Dirichlet",
        "dirichlet_alpha": partition_metadata["dirichlet_alpha"],
        "partition_seed": partition_metadata["random_seed"],
        "training_seed": EXPERIMENT_SEED,
        "clients": NUM_CLIENTS,
        "communication_rounds": TOTAL_ROUNDS,
        "poisoning": "None",
        "trust_threshold": TRUST_THRESHOLD,
        "screening_enabled": False,
        "screening_note": "Threshold 0.0 intentionally accepts all valid trust scores; false-exclusion performance is not evaluated.",
        "training_consistency_method": "Temporal consistency from previous-round update direction and magnitude.",
        "temporal_direction_weight": 0.50,
        "temporal_magnitude_weight": 0.50,
        "legacy_validation_score_role": "Diagnostic comparison only; not included in combined trust.",
        "historical_trust_update": "Existing EMA fed by combined trust; unchanged in this isolated experiment and scheduled for separate revision.",
        "trust_weights": {
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA
        },
        "historical_retention": HISTORY_RETENTION,
        "total_honest_opportunities": total_honest_opportunities,
        "total_false_exclusions": None,
        "honest_false_exclusion_rate": None,
        "client_sizes": partition_metadata["client_sizes"],
        "client_class_counts": partition_metadata["client_class_counts"],
        "test_set_used": False,
        "round_results": round_results
    }

    summary_path = OUTPUT_DIR / "noniid_temporal_consistency_v2_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    metrics_rows = []

    for result in round_results:

        metrics_rows.append({
            "Round": result["round"],
            "Validation_Loss": result["validation_loss"],
            "Validation_Accuracy": result["validation_accuracy"],
            "Validation_Macro_F1": result["validation_macro_f1"],
            "Validation_Weighted_F1": result["validation_weighted_f1"],
            "Accepted_Clients": len(result["accepted_clients"]),
            "Excluded_Clients": len(result["excluded_honest_clients"]),
            "Screening_Enabled": False
        })

    metrics_path = OUTPUT_DIR / "noniid_temporal_consistency_v2_metrics.csv"

    pd.DataFrame(metrics_rows).to_csv(
        metrics_path,
        index=False
    )

    trust_path = OUTPUT_DIR / "noniid_temporal_consistency_v2_trust_history.csv"

    pd.DataFrame(trust_history).to_csv(
        trust_path,
        index=False
    )

    # Per-client temporal consistency summary. This describes the observed
    # signal distribution; it is not a malicious-client detection result.
    trust_df = pd.DataFrame(trust_history)
    temporal_summary = (
        trust_df.groupby("Client")["Training_Consistency_Temporal"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={
            "mean": "Temporal_Consistency_Mean",
            "std": "Temporal_Consistency_Std",
            "min": "Temporal_Consistency_Min",
            "max": "Temporal_Consistency_Max",
        })
    )
    temporal_summary_path = OUTPUT_DIR / "client_temporal_consistency_summary.csv"
    temporal_summary.to_csv(temporal_summary_path, index=False)

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 78)
    print("CLEAN NON-IID TEMPORAL-CONSISTENCY CALIBRATION COMPLETED")
    print("=" * 78)

    print("Communication rounds:", TOTAL_ROUNDS)
    print("Screening enabled: NO")
    print("False-exclusion rate: NOT EVALUATED (screening disabled)")
    print("\nPer-client temporal consistency summary:")
    print(temporal_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print(
        "Final validation accuracy:",
        f"{round_results[-1]['validation_accuracy'] * 100:.4f}%"
    )

    print(
        "Final validation macro F1:",
        f"{round_results[-1]['validation_macro_f1']:.6f}"
    )

    print(
        "Final validation weighted F1:",
        f"{round_results[-1]['validation_weighted_f1']:.6f}"
    )

    print("\nSummary:", summary_path)
    print("Round metrics:", metrics_path)
    print("Trust history:", trust_path)
    print("Client temporal summary:", temporal_summary_path)

    print("\nIndependent test dataset was not used.")
    print("Original Scenario H/G files and results were not modified.")

    del validation_X, validation_y, validation_loader
    gc.collect()


if __name__ == "__main__":
    run_clean_noniid_temporal_consistency_v2()
