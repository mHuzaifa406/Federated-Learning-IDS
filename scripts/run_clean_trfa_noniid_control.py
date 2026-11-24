
"""
Scenario G: Clean Non-IID TRFA Control

All ten clients are honest.
Training records are distributed using:
- Class-wise Dirichlet partitioning
- Alpha = 0.5
- Partition seed = 43

Purpose:
Evaluate whether the existing TRFA threshold incorrectly
excludes honest clients under non-IID conditions.
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
    calculate_training_consistency,
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
    / "scenario_G_clean_trfa_noniid_control"
)

TOTAL_ROUNDS = 10
NUM_CLIENTS = 10

TRUST_THRESHOLD = 0.7422773619

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

def run_clean_noniid_control():

    print("\n" + "=" * 70)
    print("SCENARIO G: CLEAN NON-IID TRFA CONTROL")
    print("=" * 70)

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
    print("All clients are honest: YES")

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

    round_results = []
    trust_history = []

    total_false_exclusions = 0

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

        print("\n[3/7] Calculating training consistency...")

        baseline_validation_loss = (
            engine.evaluate_validation_loss(
                global_model,
                validation_loader
            )
        )

        consistency_scores = calculate_training_consistency(
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
            "{:<12} {:<12} {:<12} {:<12} {:<12}".format(
                "Client",
                "Similarity",
                "Consistency",
                "History",
                "Combined"
            )
        )

        for client_id in client_ids:

            print(
                "{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                    f"Client {client_id:02d}",
                    similarity_scores[client_id],
                    consistency_scores[client_id],
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

        # All clients are honest, so every exclusion is
        # counted as an honest-client false exclusion.
        false_exclusions = list(excluded)

        total_false_exclusions += len(false_exclusions)

        print("Accepted clients:", accepted)
        print("Excluded honest clients:", excluded)

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

        # Update historical trust for the next round.
        tracker.update(combined_scores)

        updated_historical_scores = (
            tracker.get_scores(client_ids)
        )

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
                "Training_Consistency": consistency_scores[client_id],
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
            "consistency_scores": consistency_scores,
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
                "aggregation_weights": aggregation_weights
            },
            checkpoint_path
        )

        print("Checkpoint saved:", checkpoint_path)

        del client_states, client_updates
        gc.collect()

    # ========================================================
    # SAVE COMPLETE EXPERIMENT RESULTS
    # ========================================================

    total_honest_opportunities = (
        NUM_CLIENTS * TOTAL_ROUNDS
    )

    false_exclusion_rate = (
        total_false_exclusions /
        total_honest_opportunities
    )

    summary = {
        "scenario": "G_clean_trfa_noniid_control",
        "dataset": "CIC-IDS-2017",
        "partition_method": "Class-wise Dirichlet",
        "dirichlet_alpha": partition_metadata["dirichlet_alpha"],
        "partition_seed": partition_metadata["random_seed"],
        "training_seed": EXPERIMENT_SEED,
        "clients": NUM_CLIENTS,
        "communication_rounds": TOTAL_ROUNDS,
        "poisoning": "None",
        "trust_threshold": TRUST_THRESHOLD,
        "trust_weights": {
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA
        },
        "historical_retention": HISTORY_RETENTION,
        "total_honest_opportunities": total_honest_opportunities,
        "total_false_exclusions": total_false_exclusions,
        "honest_false_exclusion_rate": false_exclusion_rate,
        "client_sizes": partition_metadata["client_sizes"],
        "client_class_counts": partition_metadata["client_class_counts"],
        "test_set_used": False,
        "round_results": round_results
    }

    summary_path = OUTPUT_DIR / "clean_noniid_trfa_summary.json"

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
            "False_Exclusions": len(result["excluded_honest_clients"])
        })

    metrics_path = OUTPUT_DIR / "clean_noniid_trfa_metrics.csv"

    pd.DataFrame(metrics_rows).to_csv(
        metrics_path,
        index=False
    )

    trust_path = OUTPUT_DIR / "clean_noniid_trfa_trust_history.csv"

    pd.DataFrame(trust_history).to_csv(
        trust_path,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("SCENARIO G: CLEAN NON-IID TRFA CONTROL COMPLETED")
    print("=" * 70)

    print("Communication rounds:", TOTAL_ROUNDS)

    print(
        "Honest false exclusions:",
        total_false_exclusions,
        "/",
        total_honest_opportunities
    )

    print(
        "Honest false-exclusion rate:",
        f"{false_exclusion_rate * 100:.2f}%"
    )

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
    print("Metrics:", metrics_path)
    print("Trust history:", trust_path)

    print("\nIndependent test dataset was not used.")

    del validation_X, validation_y, validation_loader
    gc.collect()


if __name__ == "__main__":
    run_clean_noniid_control()
