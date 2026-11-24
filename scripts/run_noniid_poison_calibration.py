
"""
Scenario I: Non-IID Poisoned Trust Calibration
================================================

Dataset:
    CIC-IDS-2017

Client distribution:
    Class-wise Dirichlet, alpha = 0.5, seed = 43

Attack:
    Client 10 performs targeted label flipping.
    20% of its non-Benign records are changed to Benign.
    Poisoning seed = 2027.

Training:
    10 communication rounds.
    10 clients.
    1 local epoch per client.

Purpose:
    Observe how malicious and honest client trust scores
    evolve under non-IID conditions without client exclusion.

IMPORTANT:
    Trust threshold = 0.0.
    All valid clients participate in aggregation.
    This is a calibration experiment, not a defense experiment.
"""

from pathlib import Path
import sys
import json
import gc

# ============================================================
# PROJECT PATH
# ============================================================

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
# 1. EXPERIMENT CONFIGURATION
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
    / "scenario_I_noniid_poison_trust_calibration"
)

TOTAL_ROUNDS = 10
NUM_CLIENTS = 10

# Zero threshold allows every valid client to participate.
TRUST_THRESHOLD = 0.0

# Trust weights
ALPHA = 1.0 / 3.0
BETA = 1.0 / 3.0
GAMMA = 1.0 / 3.0

# Historical trust
HISTORY_RETENTION = 0.90
INITIAL_HISTORICAL_TRUST = 0.50

# Attack configuration
MALICIOUS_CLIENT = 10
POISON_RATIO = 0.20
TARGET_LABEL = 0
POISON_SEED = 2027

# Training configuration
EXPERIMENT_SEED = 43


# ============================================================
# 2. LABEL-FLIPPING ATTACK
# ============================================================

def apply_label_flipping(y, client_id):
    """
    Apply targeted label flipping to Client 10.

    The same records are selected in every communication round
    because the same random seed and data ordering are used.

    Only non-Benign labels are eligible for poisoning.
    """

    original_y = y.copy()
    poisoned_y = y.copy()

    if client_id != MALICIOUS_CLIENT:
        return poisoned_y, None

    eligible_indices = np.flatnonzero(
        original_y != TARGET_LABEL
    )

    if len(eligible_indices) == 0:
        raise ValueError(
            "No eligible non-Benign records found "
            "for the malicious client."
        )

    poison_count = int(
        np.floor(len(eligible_indices) * POISON_RATIO)
    )

    if poison_count < 1:
        raise ValueError(
            "Poisoning ratio selected fewer than one record."
        )

    # Reproducible selection.
    rng = np.random.default_rng(
        POISON_SEED + client_id
    )

    selected_indices = rng.choice(
        eligible_indices,
        size=poison_count,
        replace=False
    )

    selected_original_labels = original_y[
        selected_indices
    ]

    # Change selected labels to Benign.
    poisoned_y[selected_indices] = TARGET_LABEL

    actual_changes = int(
        np.sum(poisoned_y != original_y)
    )

    if actual_changes != poison_count:
        raise RuntimeError(
            "Number of changed labels does not match "
            "the expected poisoning count."
        )

    unique_labels, label_counts = np.unique(
        selected_original_labels,
        return_counts=True
    )

    original_label_counts = {
        str(int(label)): int(count)
        for label, count in zip(
            unique_labels,
            label_counts
        )
    }

    statistics = {
        "client_id": client_id,
        "malicious": True,
        "original_samples": int(len(original_y)),
        "eligible_non_benign_records": int(
            len(eligible_indices)
        ),
        "poisoned_records": actual_changes,
        "poison_ratio": POISON_RATIO,
        "eligible_record_poison_fraction": (
            actual_changes / len(eligible_indices)
        ),
        "overall_client_poison_fraction": (
            actual_changes / len(original_y)
        ),
        "original_selected_label_counts": (
            original_label_counts
        ),
        "target_label": TARGET_LABEL,
        "poison_seed": POISON_SEED
    }

    return poisoned_y, statistics


# ============================================================
# 3. VALIDATION METRICS
# ============================================================

def evaluate_validation_metrics(model, loader):
    """
    Calculate accuracy, macro F1 and weighted F1
    using the unchanged server-validation dataset.
    """

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(engine.DEVICE)

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
            accuracy_score(
                all_labels,
                all_predictions
            )
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
# 4. REPRODUCIBILITY
# ============================================================

def configure_random_seeds():

    np.random.seed(EXPERIMENT_SEED)

    torch.manual_seed(EXPERIMENT_SEED)

    torch.set_num_threads(
        engine.CPU_THREADS
    )


# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================

def run_noniid_poison_calibration():

    print("\n" + "=" * 70)
    print("SCENARIO I: NON-IID POISONED TRUST CALIBRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Verify the non-IID dataset
    # --------------------------------------------------------

    if not NONIID_DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Non-IID dataset not found: {NONIID_DATA_ROOT}"
        )

    metadata_path = (
        NONIID_DATA_ROOT / "metadata.json"
    )

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
        raise ValueError(
            "Unexpected Dirichlet alpha."
        )

    if partition_metadata["random_seed"] != 43:
        raise ValueError(
            "Unexpected partition seed."
        )

    # Use the separate non-IID dataset.
    # The original IID dataset remains untouched.
    engine.DATA_ROOT = NONIID_DATA_ROOT

    configure_random_seeds()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Print experiment configuration
    # --------------------------------------------------------

    print("\nEXPERIMENT CONFIGURATION")
    print("-" * 70)

    print("Dataset: CIC-IDS-2017")
    print("Partition method: Class-wise Dirichlet")
    print("Dirichlet alpha:", partition_metadata["dirichlet_alpha"])
    print("Partition seed:", partition_metadata["random_seed"])
    print("Training seed:", EXPERIMENT_SEED)

    print("Number of clients:", NUM_CLIENTS)
    print("Communication rounds:", TOTAL_ROUNDS)

    print("Malicious client:", MALICIOUS_CLIENT)
    print("Poisoning ratio:", POISON_RATIO)
    print("Target label:", TARGET_LABEL)
    print("Poisoning seed:", POISON_SEED)

    print("\nTRFA CONFIGURATION")
    print("-" * 70)

    print("Alpha:", ALPHA)
    print("Beta:", BETA)
    print("Gamma:", GAMMA)

    print("Historical retention:", HISTORY_RETENTION)
    print("Initial historical trust:", INITIAL_HISTORICAL_TRUST)

    print("Trust threshold:", TRUST_THRESHOLD)
    print("Client screening: Disabled for calibration")

    # --------------------------------------------------------
    # Initialize global model
    # --------------------------------------------------------

    print("\nInitializing global IDS model...")

    global_model = IntrusionDetectionMLP().to(
        engine.DEVICE
    )

    print("Execution device:", engine.DEVICE)

    # --------------------------------------------------------
    # Load clean validation dataset
    # --------------------------------------------------------

    print("\nLoading clean server-validation dataset...")

    validation_X, validation_y, feature_columns = (
        engine.load_validation_data()
    )

    # Make writable independent copies for PyTorch.
    validation_X = np.array(
        validation_X,
        dtype=np.float32,
        copy=True
    )

    validation_y = np.array(
        validation_y,
        dtype=np.int64,
        copy=True
    )

    validation_loader = (
        engine.create_validation_loader(
            validation_X,
            validation_y
        )
    )

    print(
        "Validation records:",
        f"{len(validation_y):,}"
    )

    # Initial global validation loss.
    initial_validation_loss = (
        engine.evaluate_validation_loss(
            global_model,
            validation_loader
        )
    )

    print(
        "Initial global validation loss:",
        f"{initial_validation_loss:.6f}"
    )

    # --------------------------------------------------------
    # Initialize historical trust
    # --------------------------------------------------------

    client_ids = list(
        range(1, NUM_CLIENTS + 1)
    )

    historical_tracker = HistoricalTrustTracker(
        client_ids=client_ids,
        retention=HISTORY_RETENTION,
        initial_trust=INITIAL_HISTORICAL_TRUST
    )

    # --------------------------------------------------------
    # Experiment storage
    # --------------------------------------------------------

    round_results = []

    trust_history = []

    client_states_for_round = None

    poisoning_statistics = None

    baseline_validation_loss_history = []

    total_poisoned_records = 0

    # ========================================================
    # 6. COMMUNICATION ROUNDS
    # ========================================================

    for round_number in range(
        1,
        TOTAL_ROUNDS + 1
    ):

        print("\n" + "=" * 70)

        print(
            f"COMMUNICATION ROUND "
            f"{round_number}/{TOTAL_ROUNDS}"
        )

        print("=" * 70)

        client_states = {}
        client_updates = {}

        client_sizes = {}

        client_training_losses = {}

        candidate_validation_losses = {}

        # ----------------------------------------------------
        # STEP 1: Local client training
        # ----------------------------------------------------

        print(
            "\n[1/7] Training non-IID client models..."
        )

        for client_id in client_ids:

            print(
                f"\nTraining client "
                f"{client_id:02d}/{NUM_CLIENTS}..."
            )

            # Load from the separate non-IID partition.
            X, original_y, sample_count = (
                engine.load_client_data(
                    client_id,
                    feature_columns
                )
            )

            # Apply targeted label flipping to Client 10.
            training_y, poison_stats = (
                apply_label_flipping(
                    original_y,
                    client_id
                )
            )

            if (
                client_id == MALICIOUS_CLIENT
                and round_number == 1
            ):

                poisoning_statistics = poison_stats

                total_poisoned_records = (
                    poison_stats["poisoned_records"]
                )

                print("\nMALICIOUS CLIENT ATTACK DETAILS")

                print(
                    "Eligible non-Benign records:",
                    f"{poison_stats['eligible_non_benign_records']:,}"
                )

                print(
                    "Poisoned records:",
                    f"{poison_stats['poisoned_records']:,}"
                )

                print(
                    "Eligible-record poisoning fraction:",
                    f"{poison_stats['eligible_record_poison_fraction']:.4%}"
                )

                print(
                    "Overall client poisoning fraction:",
                    f"{poison_stats['overall_client_poison_fraction']:.4%}"
                )

                print(
                    "Original labels changed:",
                    poison_stats["original_selected_label_counts"]
                )

            # Local training begins from the current global model.
            local_model, training_loss = (
                engine.train_local_model(
                    global_model,
                    X,
                    training_y
                )
            )

            # Keep the complete local model state.
            local_state = {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }

            # Evaluate every candidate on the same clean
            # server-validation dataset.
            candidate_validation_loss = (
                engine.evaluate_validation_loss(
                    local_model,
                    validation_loader
                )
            )

            # Calculate the local update vector.
            update_vector = flatten_model_update(
                local_state,
                global_model.state_dict()
            )

            client_states[client_id] = local_state

            client_updates[client_id] = (
                update_vector
            )

            # Label flipping does not change sample count.
            client_sizes[client_id] = int(
                sample_count
            )

            client_training_losses[client_id] = float(
                training_loss
            )

            candidate_validation_losses[client_id] = float(
                candidate_validation_loss
            )

            print(
                "Training records:",
                f"{sample_count:,}"
            )

            print(
                "Local training loss:",
                f"{training_loss:.6f}"
            )

            print(
                "Candidate validation loss:",
                f"{candidate_validation_loss:.6f}"
            )

            del X
            del original_y
            del training_y
            del local_model

            gc.collect()

        # ----------------------------------------------------
        # STEP 2: Update similarity
        # ----------------------------------------------------

        print(
            "\n[2/7] Calculating update similarity..."
        )

        reference_update = (
            calculate_reference_update(
                client_updates
            )
        )

        similarity_scores = (
            calculate_update_similarity(
                client_updates,
                reference_update
            )
        )

        # ----------------------------------------------------
        # STEP 3: Training consistency
        # ----------------------------------------------------

        print(
            "\n[3/7] Calculating training consistency..."
        )

        baseline_validation_loss = (
            engine.evaluate_validation_loss(
                global_model,
                validation_loader
            )
        )

        baseline_validation_loss_history.append(
            float(baseline_validation_loss)
        )

        consistency_scores = (
            calculate_training_consistency(
                baseline_validation_loss,
                candidate_validation_losses
            )
        )

        # ----------------------------------------------------
        # STEP 4: Historical and combined trust
        # ----------------------------------------------------

        print(
            "\n[4/7] Calculating combined trust scores..."
        )

        historical_scores = (
            historical_tracker.get_scores(
                client_ids
            )
        )

        combined_trust_scores = (
            calculate_combined_trust(
                similarity_scores=similarity_scores,
                consistency_scores=consistency_scores,
                historical_scores=historical_scores,
                alpha=ALPHA,
                beta=BETA,
                gamma=GAMMA
            )
        )

        print("\nTRFA TRUST SCORE TABLE")

        print(
            "{:<12} {:<13} {:<13} {:<13} {:<13}".format(
                "Client",
                "Similarity",
                "Consistency",
                "History",
                "Combined"
            )
        )

        for client_id in client_ids:

            print(
                "{:<12} {:<13.4f} {:<13.4f} {:<13.4f} {:<13.4f}".format(
                    f"Client {client_id:02d}",
                    similarity_scores[client_id],
                    consistency_scores[client_id],
                    historical_scores[client_id],
                    combined_trust_scores[client_id]
                )
            )

        # ----------------------------------------------------
        # STEP 5: Calibration mode - no client exclusion
        # ----------------------------------------------------

        print(
            "\n[5/7] Running trust calibration without exclusion..."
        )

        accepted_clients, excluded_clients = (
            screen_clients(
                combined_trust_scores,
                threshold=TRUST_THRESHOLD
            )
        )

        # Since the threshold is zero and trust scores are
        # valid non-negative numbers, all clients should pass.
        if len(accepted_clients) != NUM_CLIENTS:
            raise RuntimeError(
                "Calibration expected all clients to participate, "
                f"but accepted {len(accepted_clients)}."
            )

        if excluded_clients:
            raise RuntimeError(
                f"Unexpected excluded clients: {excluded_clients}"
            )

        print(
            "Calibration threshold:",
            TRUST_THRESHOLD
        )

        print(
            "Accepted clients:",
            accepted_clients
        )

        print(
            "Excluded clients:",
            excluded_clients
        )

        # ----------------------------------------------------
        # STEP 6: Sample-size-weighted FedAvg
        # ----------------------------------------------------

        print(
            "\n[6/7] Aggregating all client models..."
        )

        aggregated_state, aggregation_weights = (
            aggregate_accepted_models(
                client_states=client_states,
                sample_sizes=client_sizes,
                accepted_client_ids=accepted_clients,
                minimum_accepted_clients=2
            )
        )

        # Load the returned complete global state.
        global_model.load_state_dict(
            aggregated_state
        )

        print("\nAggregation weights:")

        for client_id, weight in (
            aggregation_weights.items()
        ):

            print(
                f"Client {client_id:02d}: {weight:.6f}"
            )

        print(
            "Total aggregation weight:",
            f"{sum(aggregation_weights.values()):.6f}"
        )

        # Update historical trust after current-round
        # trust calculations.
        historical_tracker.update(
            combined_trust_scores
        )

        updated_historical_scores = (
            historical_tracker.get_scores(
                client_ids
            )
        )

        # ----------------------------------------------------
        # STEP 7: Evaluate the aggregated global model
        # ----------------------------------------------------

        print(
            "\n[7/7] Evaluating global model..."
        )

        validation_loss = (
            engine.evaluate_validation_loss(
                global_model,
                validation_loader
            )
        )

        validation_metrics = (
            evaluate_validation_metrics(
                global_model,
                validation_loader
            )
        )

        print("\nROUND RESULTS")

        print(
            "Validation loss:",
            f"{validation_loss:.6f}"
        )

        print(
            "Validation accuracy:",
            f"{validation_metrics['accuracy'] * 100:.4f}%"
        )

        print(
            "Validation macro F1:",
            f"{validation_metrics['macro_f1']:.6f}"
        )

        print(
            "Validation weighted F1:",
            f"{validation_metrics['weighted_f1']:.6f}"
        )

        # ----------------------------------------------------
        # Save detailed trust information
        # ----------------------------------------------------

        for client_id in client_ids:

            trust_history.append({
                "Round": round_number,
                "Client": client_id,
                "Malicious": (
                    client_id == MALICIOUS_CLIENT
                ),
                "Update_Similarity": (
                    similarity_scores[client_id]
                ),
                "Training_Consistency": (
                    consistency_scores[client_id]
                ),
                "Historical_Trust_Before": (
                    historical_scores[client_id]
                ),
                "Combined_Trust": (
                    combined_trust_scores[client_id]
                ),
                "Historical_Trust_After": (
                    updated_historical_scores[client_id]
                ),
                "Candidate_Validation_Loss": (
                    candidate_validation_losses[client_id]
                ),
                "Client_Training_Loss": (
                    client_training_losses[client_id]
                ),
                "Client_Samples": (
                    client_sizes[client_id]
                ),
                "Accepted": (
                    client_id in accepted_clients
                ),
                "Excluded": (
                    client_id in excluded_clients
                )
            })

        # ----------------------------------------------------
        # Record round metrics
        # ----------------------------------------------------

        round_result = {
            "round": round_number,
            "validation_loss": float(validation_loss),
            "validation_accuracy": (
                validation_metrics["accuracy"]
            ),
            "validation_macro_f1": (
                validation_metrics["macro_f1"]
            ),
            "validation_weighted_f1": (
                validation_metrics["weighted_f1"]
            ),
            "baseline_validation_loss": float(
                baseline_validation_loss
            ),
            "accepted_clients": accepted_clients,
            "excluded_clients": excluded_clients,
            "malicious_client": MALICIOUS_CLIENT,
            "client_sizes": client_sizes,
            "client_training_losses": (
                client_training_losses
            ),
            "candidate_validation_losses": (
                candidate_validation_losses
            ),
            "similarity_scores": similarity_scores,
            "consistency_scores": consistency_scores,
            "historical_scores_before_update": (
                historical_scores
            ),
            "combined_trust_scores": (
                combined_trust_scores
            ),
            "historical_scores_after_update": (
                updated_historical_scores
            ),
            "aggregation_weights": (
                aggregation_weights
            )
        }

        round_results.append(
            round_result
        )

        # ----------------------------------------------------
        # Save round checkpoint
        # ----------------------------------------------------

        checkpoint_path = (
            OUTPUT_DIR
            / f"poisoned_noniid_round_{round_number:02d}.pt"
        )

        torch.save(
            {
                "round": round_number,
                "model_state_dict": (
                    global_model.state_dict()
                ),
                "validation_loss": validation_loss,
                "validation_metrics": validation_metrics,
                "accepted_clients": accepted_clients,
                "excluded_clients": excluded_clients,
                "trust_scores": combined_trust_scores,
                "aggregation_weights": aggregation_weights,
                "malicious_client": MALICIOUS_CLIENT
            },
            checkpoint_path
        )

        print(
            "Checkpoint saved:",
            checkpoint_path
        )

        del client_states
        del client_updates

        gc.collect()

    # ========================================================
    # 7. SAVE EXPERIMENT RESULTS
    # ========================================================

    print("\nSaving experiment results...")

    summary = {
        "scenario": "I_noniid_poison_trust_calibration",
        "dataset": "CIC-IDS-2017",
        "partition_method": (
            "Class-wise Dirichlet distribution"
        ),
        "dirichlet_alpha": (
            partition_metadata["dirichlet_alpha"]
        ),
        "partition_seed": (
            partition_metadata["random_seed"]
        ),
        "training_seed": EXPERIMENT_SEED,
        "clients": NUM_CLIENTS,
        "communication_rounds": TOTAL_ROUNDS,
        "local_epochs": 1,
        "attack_type": "Targeted label flipping",
        "malicious_client": MALICIOUS_CLIENT,
        "target_label": TARGET_LABEL,
        "poison_ratio": POISON_RATIO,
        "poisoning_seed": POISON_SEED,
        "poisoning_statistics": poisoning_statistics,
        "trfa_configuration": {
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "threshold": TRUST_THRESHOLD,
            "historical_retention": HISTORY_RETENTION,
            "initial_historical_trust": (
                INITIAL_HISTORICAL_TRUST
            )
        },
        "screening_enabled": False,
        "calibration_description": (
            "All clients participate. Trust scores are observed "
            "without excluding any client."
        ),
        "total_poisoned_records": total_poisoned_records,
        "test_set_used": False,
        "baseline_validation_loss_history": (
            baseline_validation_loss_history
        ),
        "round_results": round_results
    }

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "poisoned_noniid_trfa_summary.json"
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

    # --------------------------------------------------------
    # Save round-level CSV
    # --------------------------------------------------------

    metrics_rows = []

    for result in round_results:

        metrics_rows.append({
            "Round": result["round"],
            "Validation_Loss": (
                result["validation_loss"]
            ),
            "Validation_Accuracy": (
                result["validation_accuracy"]
            ),
            "Validation_Macro_F1": (
                result["validation_macro_f1"]
            ),
            "Validation_Weighted_F1": (
                result["validation_weighted_f1"]
            ),
            "Accepted_Clients": len(
                result["accepted_clients"]
            ),
            "Excluded_Clients": len(
                result["excluded_clients"]
            )
        })

    metrics_path = (
        OUTPUT_DIR
        / "poisoned_noniid_trfa_metrics.csv"
    )

    pd.DataFrame(metrics_rows).to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Save client-round trust history
    # --------------------------------------------------------

    trust_path = (
        OUTPUT_DIR
        / "poisoned_noniid_trfa_trust_history.csv"
    )

    pd.DataFrame(trust_history).to_csv(
        trust_path,
        index=False
    )

    # ========================================================
    # 8. FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("SCENARIO I: NON-IID POISONED TRUST CALIBRATION COMPLETED")
    print("=" * 70)

    print("Communication rounds:", TOTAL_ROUNDS)

    print("Total clients:", NUM_CLIENTS)

    print(
        "Malicious client:",
        MALICIOUS_CLIENT
    )

    print(
        "Poisoned records:",
        total_poisoned_records
    )

    print(
        "Poisoning fraction of eligible records:",
        f"{poisoning_statistics['eligible_record_poison_fraction']:.4%}"
    )

    print(
        "Poisoning fraction of entire malicious client:",
        f"{poisoning_statistics['overall_client_poison_fraction']:.4%}"
    )

    print(
        "Clients accepted in final round:",
        len(round_results[-1]["accepted_clients"])
    )

    print(
        "Clients excluded in final round:",
        len(round_results[-1]["excluded_clients"])
    )

    print(
        "Final validation loss:",
        f"{round_results[-1]['validation_loss']:.6f}"
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

    print("\nSAVED FILES")

    print("Summary:", summary_path)

    print("Round metrics:", metrics_path)

    print("Trust history:", trust_path)

    print("\nScreening was disabled for calibration.")

    print("Independent test dataset was not used.")

    del validation_X
    del validation_y
    del validation_loader

    gc.collect()


# ============================================================
# 9. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_noniid_poison_calibration()
