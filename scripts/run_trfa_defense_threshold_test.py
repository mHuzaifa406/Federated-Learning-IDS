
"""
Scenario C: Poisoned Federated Learning with TRFA Defense

Uses the same fixed label-flipping attack as Scenario B.

TRFA signals:
1. Update Similarity
2. Training Consistency
3. Historical Trust

Accepted clients are aggregated using sample-size-weighted FedAvg.
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

from models.ids_model import IntrusionDetectionMLP

from scripts.federated_engine import (
    set_random_seed,
    load_validation_data,
    create_validation_loader,
    load_client_data,
    train_local_model,
    evaluate_validation_loss,
    NUM_CLIENTS,
    DEVICE,
    CPU_THREADS,
)

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
# 1. EXPERIMENT CONFIGURATION
# ============================================================

TOTAL_ROUNDS = 10

MALICIOUS_CLIENT = 10
POISON_RATIO = 0.20
TARGET_LABEL = 0
POISON_SEED = 2026

# TRFA configuration
ALPHA = 1.0 / 3.0
BETA = 1.0 / 3.0
GAMMA = 1.0 / 3.0

TRUST_THRESHOLD = 0.7422773619
HISTORY_RETENTION = 0.90
INITIAL_HISTORICAL_TRUST = 0.50

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "scenario_C_threshold_07423"
)


# ============================================================
# 2. REPRODUCIBLE LABEL-FLIPPING ATTACK
# ============================================================

def apply_label_flipping(y, client_id):

    original_labels = y.copy()
    poisoned_labels = y.copy()

    statistics = {
        "client_id": client_id,
        "malicious": client_id == MALICIOUS_CLIENT,
        "original_samples": int(len(y)),
        "eligible_non_benign_samples": 0,
        "poisoned_samples": 0,
        "overall_client_poison_fraction": 0.0,
        "eligible_record_poison_fraction": 0.0,
        "original_selected_label_counts": {}
    }

    if client_id != MALICIOUS_CLIENT:
        return poisoned_labels, statistics

    eligible_indices = np.flatnonzero(
        original_labels != TARGET_LABEL
    )

    statistics["eligible_non_benign_samples"] = int(
        len(eligible_indices)
    )

    if len(eligible_indices) == 0:
        raise ValueError(
            "Malicious client has no eligible non-Benign records."
        )

    poison_count = int(
        np.floor(len(eligible_indices) * POISON_RATIO)
    )

    if poison_count < 1:
        raise ValueError(
            "Poisoning ratio selected fewer than one record."
        )

    # The same seed and data ordering reproduce the same
    # poisoned record selection used in Scenario B.
    rng = np.random.default_rng(
        POISON_SEED + client_id
    )

    selected_indices = rng.choice(
        eligible_indices,
        size=poison_count,
        replace=False
    )

    selected_original_labels = original_labels[
        selected_indices
    ]

    unique_labels, label_counts = np.unique(
        selected_original_labels,
        return_counts=True
    )

    original_selected_label_counts = {
        str(int(label)): int(count)
        for label, count in zip(unique_labels, label_counts)
    }

    poisoned_labels[selected_indices] = TARGET_LABEL

    actual_changes = int(
        np.sum(poisoned_labels != original_labels)
    )

    statistics.update({
        "poisoned_samples": actual_changes,
        "overall_client_poison_fraction": (
            actual_changes / len(y)
        ),
        "eligible_record_poison_fraction": (
            actual_changes / len(eligible_indices)
        ),
        "original_selected_label_counts": (
            original_selected_label_counts
        )
    })

    if actual_changes != poison_count:
        raise RuntimeError(
            "Unexpected mismatch in poisoned record count."
        )

    return poisoned_labels, statistics


# ============================================================
# 3. VALIDATION METRICS
# ============================================================

def evaluate_validation_metrics(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(DEVICE)

            outputs = model(batch_X)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                batch_y.numpy().tolist()
            )

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
# 4. MAIN TRFA EXPERIMENT
# ============================================================

def run_trfa_experiment():

    print("\n" + "=" * 70)
    print("SCENARIO C: POISONED FEDAVG WITH TRFA DEFENSE")
    print("=" * 70)

    set_random_seed()
    torch.set_num_threads(CPU_THREADS)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Initialize global model
    # --------------------------------------------------------

    print("\nInitializing global IDS model...")

    global_model = IntrusionDetectionMLP().to(DEVICE)

    print("Device:", DEVICE)
    print("Clients:", NUM_CLIENTS)
    print("Communication rounds:", TOTAL_ROUNDS)
    print("Malicious client:", MALICIOUS_CLIENT)
    print("Poison ratio:", POISON_RATIO)
    print("Trust threshold:", TRUST_THRESHOLD)

    # --------------------------------------------------------
    # Load clean server-validation data
    # --------------------------------------------------------

    print("\nLoading clean server-validation data...")

    validation_X, validation_y, feature_columns = (
        load_validation_data()
    )

    validation_loader = create_validation_loader(
        validation_X,
        validation_y
    )

    # --------------------------------------------------------
    # Initialize historical trust tracker
    # --------------------------------------------------------

    client_ids = list(range(1, NUM_CLIENTS + 1))

    historical_tracker = HistoricalTrustTracker(
        client_ids=client_ids,
        retention=HISTORY_RETENTION,
        initial_trust=INITIAL_HISTORICAL_TRUST
    )

    history = []
    trust_history = []

    poisoning_statistics = None

    total_malicious_detections = 0
    total_honest_false_exclusions = 0

    total_malicious_opportunities = TOTAL_ROUNDS

    total_honest_opportunities = (
        TOTAL_ROUNDS * (NUM_CLIENTS - 1)
    )

    # --------------------------------------------------------
    # Communication rounds
    # --------------------------------------------------------

    for round_number in range(1, TOTAL_ROUNDS + 1):

        print("\n" + "=" * 70)
        print(
            f"COMMUNICATION ROUND {round_number}/{TOTAL_ROUNDS}"
        )
        print("=" * 70)

        # Store all candidate model states and update vectors.
        client_states = {}
        client_updates = {}

        client_sizes = {}
        client_training_losses = {}
        candidate_validation_losses = {}

        # ----------------------------------------------------
        # STEP 1: Train all local clients
        # ----------------------------------------------------

        print("\n[1/7] Training local client models...")

        for client_id in client_ids:

            print(
                f"\nTraining client {client_id:02d}/{NUM_CLIENTS}..."
            )

            X, original_y, sample_count = load_client_data(
                client_id,
                feature_columns
            )

            # Apply poisoning only to client 10.
            training_y, poison_stats = apply_label_flipping(
                original_y,
                client_id
            )

            if (
                client_id == MALICIOUS_CLIENT
                and poisoning_statistics is None
            ):

                poisoning_statistics = poison_stats

                print("\nATTACK VERIFICATION")

                print(
                    "Eligible non-Benign records:",
                    f"{poison_stats['eligible_non_benign_samples']:,}"
                )

                print(
                    "Poisoned records:",
                    f"{poison_stats['poisoned_samples']:,}"
                )

                print(
                    "Eligible-record poisoning fraction:",
                    f"{poison_stats['eligible_record_poison_fraction']:.4%}"
                )

                print(
                    "Entire-client poisoning fraction:",
                    f"{poison_stats['overall_client_poison_fraction']:.4%}"
                )

            local_model, training_loss = train_local_model(
                global_model,
                X,
                training_y
            )

            local_state = {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }

            # Evaluate the candidate client model using the
            # same clean validation records for every client.
            candidate_validation_loss = evaluate_validation_loss(
                local_model,
                validation_loader
            )

            # Calculate its model update relative to the current
            # global model.
            global_state = global_model.state_dict()

            update_vector = flatten_model_update(
                local_state,
                global_state
            )

            client_states[client_id] = local_state
            client_updates[client_id] = update_vector

            # The number of records is not changed by label flipping.
            client_sizes[client_id] = int(sample_count)

            client_training_losses[client_id] = float(
                training_loss
            )

            candidate_validation_losses[client_id] = float(
                candidate_validation_loss
            )

            print("  Samples:", f"{sample_count:,}")
            print("  Training loss:", f"{training_loss:.6f}")
            print(
                "  Candidate validation loss:",
                f"{candidate_validation_loss:.6f}"
            )

            del X, original_y, training_y, local_model
            gc.collect()

        # ----------------------------------------------------
        # STEP 2: Calculate update similarity
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
        # STEP 3: Calculate training consistency
        # ----------------------------------------------------

        print("\n[3/7] Calculating training consistency...")

        baseline_validation_loss = evaluate_validation_loss(
            global_model,
            validation_loader
        )

        consistency_scores = calculate_training_consistency(
            baseline_validation_loss,
            candidate_validation_losses
        )

        # ----------------------------------------------------
        # STEP 4: Retrieve historical trust and combine scores
        # ----------------------------------------------------

        print("\n[4/7] Calculating combined trust scores...")

        historical_scores = historical_tracker.get_scores(
            client_ids
        )

        combined_trust_scores = calculate_combined_trust(
            similarity_scores=similarity_scores,
            consistency_scores=consistency_scores,
            historical_scores=historical_scores,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA
        )

        print("\nTRFA TRUST SCORE TABLE")

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
                    combined_trust_scores[client_id]
                )
            )

        # ----------------------------------------------------
        # STEP 5: Screen clients
        # ----------------------------------------------------

        print("\n[5/7] Screening clients using TRFA...")

        accepted_clients, rejected_clients = screen_clients(
            combined_trust_scores,
            threshold=TRUST_THRESHOLD
        )

        print("Trust threshold:", TRUST_THRESHOLD)
        print("Accepted clients:", accepted_clients)
        print("Excluded clients:", rejected_clients)

        # Measure attacker detection and honest-client exclusions.
        malicious_detected = (
            MALICIOUS_CLIENT in rejected_clients
        )

        honest_excluded = [
            client_id
            for client_id in rejected_clients
            if client_id != MALICIOUS_CLIENT
        ]

        if malicious_detected:
            total_malicious_detections += 1

        total_honest_false_exclusions += len(honest_excluded)

        # ----------------------------------------------------
        # STEP 6: Aggregate accepted clients
        # ----------------------------------------------------

        print("\n[6/7] Aggregating accepted client models...")

        if len(accepted_clients) < 2:
            raise RuntimeError(
                "Fewer than two clients passed TRFA screening. "
                "Aggregation was stopped."
            )

        aggregated_state, aggregation_weights = (
            aggregate_accepted_models(
                client_states=client_states,
                sample_sizes=client_sizes,
                accepted_client_ids=accepted_clients,
                minimum_accepted_clients=2
            )
        )

        # Load the complete aggregated global model state.
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

        # Update historical scores after calculating current-round
        # trust. The next communication round uses these values.
        historical_tracker.update(
            combined_trust_scores
        )

        updated_historical_scores = (
            historical_tracker.get_scores(client_ids)
        )

        # ----------------------------------------------------
        # STEP 7: Evaluate aggregated global model
        # ----------------------------------------------------

        print("\n[7/7] Evaluating aggregated global model...")

        validation_loss = evaluate_validation_loss(
            global_model,
            validation_loader
        )

        metrics = evaluate_validation_metrics(
            global_model,
            validation_loader
        )

        print("\nROUND METRICS")

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

        print("Malicious client detected:", malicious_detected)
        print("Honest clients falsely excluded:", honest_excluded)

        # ----------------------------------------------------
        # Record detailed trust information
        # ----------------------------------------------------

        for client_id in client_ids:

            trust_history.append({
                "Round": round_number,
                "Client": client_id,
                "Malicious": client_id == MALICIOUS_CLIENT,
                "Update_Similarity": similarity_scores[client_id],
                "Training_Consistency": consistency_scores[client_id],
                "Historical_Trust_Before": historical_scores[client_id],
                "Combined_Trust": combined_trust_scores[client_id],
                "Historical_Trust_After": updated_historical_scores[client_id],
                "Candidate_Validation_Loss": candidate_validation_losses[client_id],
                "Client_Training_Loss": client_training_losses[client_id],
                "Accepted": client_id in accepted_clients,
                "Excluded": client_id in rejected_clients
            })

        round_result = {
            "round": round_number,
            "validation_loss": float(validation_loss),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_weighted_f1": metrics["weighted_f1"],
            "baseline_validation_loss": float(baseline_validation_loss),
            "accepted_clients": accepted_clients,
            "rejected_clients": rejected_clients,
            "malicious_client_detected": malicious_detected,
            "honest_clients_false_excluded": honest_excluded,
            "aggregation_weights": aggregation_weights,
            "similarity_scores": similarity_scores,
            "consistency_scores": consistency_scores,
            "historical_scores_before_update": historical_scores,
            "combined_trust_scores": combined_trust_scores,
            "historical_scores_after_update": updated_historical_scores
        }

        history.append(round_result)

        # ----------------------------------------------------
        # Save round checkpoint
        # ----------------------------------------------------

        checkpoint_path = (
            OUTPUT_DIR /
            f"trfa_round_{round_number:02d}.pt"
        )

        torch.save(
            {
                "round": round_number,
                "model_state_dict": global_model.state_dict(),
                "validation_loss": validation_loss,
                "validation_metrics": metrics,
                "accepted_clients": accepted_clients,
                "rejected_clients": rejected_clients,
                "trust_scores": combined_trust_scores,
                "aggregation_weights": aggregation_weights
            },
            checkpoint_path
        )

        print("Checkpoint saved:", checkpoint_path)

        del client_states, client_updates
        gc.collect()

    # ========================================================
    # FINAL DETECTION AND FALSE-EXCLUSION STATISTICS
    # ========================================================

    malicious_detection_rate = (
        total_malicious_detections /
        total_malicious_opportunities
    )

    honest_false_exclusion_rate = (
        total_honest_false_exclusions /
        total_honest_opportunities
    )

    # ========================================================
    # SAVE EXPERIMENT SUMMARY
    # ========================================================

    summary = {
        "scenario": "C_poisoned_fedavg_with_trfa",
        "dataset": "CIC-IDS-2017",
        "clients": NUM_CLIENTS,
        "communication_rounds": TOTAL_ROUNDS,
        "local_epochs": 1,
        "attack_type": "Targeted label flipping",
        "malicious_client": MALICIOUS_CLIENT,
        "poison_ratio": POISON_RATIO,
        "target_label": TARGET_LABEL,
        "poisoning_statistics": poisoning_statistics,
        "trfa_configuration": {
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "threshold": TRUST_THRESHOLD,
            "historical_retention": HISTORY_RETENTION,
            "initial_historical_trust": INITIAL_HISTORICAL_TRUST
        },
        "total_malicious_detections": total_malicious_detections,
        "malicious_detection_rate": malicious_detection_rate,
        "total_honest_false_exclusions": total_honest_false_exclusions,
        "honest_false_exclusion_rate": honest_false_exclusion_rate,
        "test_set_used_during_training": False,
        "round_results": history
    }

    summary_path = (
        OUTPUT_DIR / "trfa_experiment_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # Save per-round metric history.
    metrics_rows = []

    for result in history:

        metrics_rows.append({
            "Round": result["round"],
            "Validation_Loss": result["validation_loss"],
            "Validation_Accuracy": result["validation_accuracy"],
            "Validation_Macro_F1": result["validation_macro_f1"],
            "Validation_Weighted_F1": result["validation_weighted_f1"],
            "Accepted_Clients": len(result["accepted_clients"]),
            "Rejected_Clients": len(result["rejected_clients"]),
            "Malicious_Client_Detected": result["malicious_client_detected"],
            "Honest_False_Exclusions": len(
                result["honest_clients_false_excluded"]
            )
        })

    metrics_csv = (
        OUTPUT_DIR / "trfa_round_metrics.csv"
    )

    pd.DataFrame(metrics_rows).to_csv(
        metrics_csv,
        index=False
    )

    # Save per-client, per-round trust details.
    trust_csv = (
        OUTPUT_DIR / "trfa_client_trust_history.csv"
    )

    pd.DataFrame(trust_history).to_csv(
        trust_csv,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("SCENARIO C: TRFA EXPERIMENT COMPLETED")
    print("=" * 70)

    print("Communication rounds:", TOTAL_ROUNDS)
    print("Malicious client:", MALICIOUS_CLIENT)

    print(
        "Poisoned records:",
        poisoning_statistics["poisoned_samples"]
    )

    print(
        "Malicious detection rate:",
        f"{malicious_detection_rate * 100:.2f}%"
    )

    print(
        "Honest-client false-exclusion rate:",
        f"{honest_false_exclusion_rate * 100:.2f}%"
    )

    print(
        "Final validation accuracy:",
        f"{history[-1]['validation_accuracy'] * 100:.4f}%"
    )

    print(
        "Final validation macro F1:",
        f"{history[-1]['validation_macro_f1']:.6f}"
    )

    print(
        "Final validation weighted F1:",
        f"{history[-1]['validation_weighted_f1']:.6f}"
    )

    print("\nSaved files:")
    print("Summary:", summary_path)
    print("Round metrics:", metrics_csv)
    print("Trust history:", trust_csv)

    print("\nIndependent test set was not used.")

    del validation_X, validation_y, validation_loader
    gc.collect()


if __name__ == "__main__":
    run_trfa_experiment()
