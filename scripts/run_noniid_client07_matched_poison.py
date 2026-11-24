
"""
SCENARIO K: NON-IID CLIENT 07 MATCHED-STRENGTH POISONING
========================================================

Dataset:
    CIC-IDS-2017

Client distribution:
    Class-wise Dirichlet
    Alpha = 0.5
    Partition seed = 43

Malicious client:
    Client 07

Attack:
    Targeted label flipping
    Target label = Benign (0)

Poisoning strength:
    Matched to Scenario I's overall malicious-client
    poisoning fraction:

    8676 / 213858 = approximately 4.0569%

Training:
    10 clients
    10 communication rounds
    1 local epoch per round
    Training seed = 43

Calibration:
    Trust threshold = 0.0
    All clients participate.
    No client is excluded.

Purpose:
    Compare the effect of poisoning Client 07 with a
    matched overall-client poisoning fraction.
"""


# ============================================================
# 1. PROJECT PATH
# ============================================================

from pathlib import Path
import sys
import json
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. LIBRARIES
# ============================================================

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
# 3. EXPERIMENT CONFIGURATION
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
    / "scenario_K_noniid_client07_matched_poison"
)

TOTAL_ROUNDS = 10
NUM_CLIENTS = 10

EXPERIMENT_SEED = 43

# Malicious client
MALICIOUS_CLIENT = 7

# Targeted label flipping
TARGET_LABEL = 0
POISON_SEED = 2027

# Matched overall-client poisoning fraction.
# Scenario I: 8,676 poisoned records out of 213,858.
POISON_CLIENT_FRACTION = 8676 / 213858

# TRFA configuration
ALPHA = 1.0 / 3.0
BETA = 1.0 / 3.0
GAMMA = 1.0 / 3.0

HISTORY_RETENTION = 0.90
INITIAL_HISTORICAL_TRUST = 0.50

# Calibration threshold: all clients must participate.
TRUST_THRESHOLD = 0.0


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
# 5. MATCHED-STRENGTH LABEL-FLIPPING ATTACK
# ============================================================

def apply_label_flipping(y, client_id):
    """
    Apply targeted label flipping to Client 07.

    The number of poisoned records is calculated as a
    fraction of the entire malicious client's dataset.

    This matches Scenario I's overall-client poisoning
    fraction rather than its eligible-record fraction.

    Only records whose original labels differ from the
    target label are eligible for modification.

    The same records are selected every communication
    round using the fixed random seed and data ordering.
    """

    original_y = y.copy()

    poisoned_y = y.copy()

    # Honest clients retain their original labels.
    if client_id != MALICIOUS_CLIENT:

        return poisoned_y, None

    # Find records that can actually be changed.
    eligible_indices = np.flatnonzero(
        original_y != TARGET_LABEL
    )

    if len(eligible_indices) == 0:

        raise ValueError(
            "No eligible non-Benign records found "
            "for Client 07."
        )

    # Match the poisoning fraction of the entire client.
    poison_count = int(
        np.floor(
            len(original_y) * POISON_CLIENT_FRACTION
        )
    )

    if poison_count < 1:

        raise ValueError(
            "Matched poisoning fraction selected fewer "
            "than one record."
        )

    if poison_count > len(eligible_indices):

        raise ValueError(
            "Insufficient eligible records for poisoning."
        )

    # Reproducible record selection.
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
        np.sum(
            poisoned_y != original_y
        )
    )

    if actual_changes != poison_count:

        raise RuntimeError(
            "Actual poisoning count does not match "
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

        "poison_ratio_definition": (
            "Fraction of entire malicious client dataset"
        ),

        "matched_poison_fraction": float(
            POISON_CLIENT_FRACTION
        ),

        "actual_overall_client_poison_fraction": (
            actual_changes / len(original_y)
        ),

        "actual_eligible_record_poison_fraction": (
            actual_changes / len(eligible_indices)
        ),

        "original_selected_label_counts": (
            original_label_counts
        ),

        "target_label": TARGET_LABEL,

        "poison_seed": POISON_SEED
    }

    return poisoned_y, statistics


# ============================================================
# 6. VALIDATION METRICS
# ============================================================

def evaluate_validation_metrics(model, loader):
    """
    Evaluate the global model using the clean validation data.

    Returns:
        Accuracy
        Macro F1
        Weighted F1
    """

    model.eval()

    all_labels = []

    all_predictions = []

    with torch.no_grad():

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(
                engine.DEVICE
            )

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

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def run_noniid_client07_matched_poison():

    print("\n" + "=" * 70)

    print(
        "SCENARIO K: NON-IID CLIENT 07 MATCHED-STRENGTH POISONING"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Verify non-IID dataset
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
            "Non-IID metadata file is missing."
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
            "Unexpected non-IID partition seed."
        )

    # Use the separate non-IID dataset.
    # The original IID data remains unchanged.
    engine.DATA_ROOT = NONIID_DATA_ROOT

    configure_random_seeds()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("\nEXPERIMENT CONFIGURATION")
    print("-" * 70)

    print("Dataset: CIC-IDS-2017")

    print(
        "Partition method:",
        partition_metadata["partition_method"]
    )

    print(
        "Dirichlet alpha:",
        partition_metadata["dirichlet_alpha"]
    )

    print(
        "Partition seed:",
        partition_metadata["random_seed"]
    )

    print(
        "Training seed:",
        EXPERIMENT_SEED
    )

    print(
        "Number of clients:",
        NUM_CLIENTS
    )

    print(
        "Communication rounds:",
        TOTAL_ROUNDS
    )

    print("\nATTACK CONFIGURATION")
    print("-" * 70)

    print(
        "Malicious client:",
        MALICIOUS_CLIENT
    )

    print(
        "Poisoning fraction of entire client:",
        f"{POISON_CLIENT_FRACTION:.6%}"
    )

    print(
        "Matched reference fraction:",
        "8676 / 213858"
    )

    print(
        "Target label:",
        TARGET_LABEL,
        "(Benign)"
    )

    print(
        "Poisoning seed:",
        POISON_SEED
    )

    print("\nTRFA CONFIGURATION")
    print("-" * 70)

    print("Alpha:", ALPHA)
    print("Beta:", BETA)
    print("Gamma:", GAMMA)

    print(
        "Historical retention:",
        HISTORY_RETENTION
    )

    print(
        "Initial historical trust:",
        INITIAL_HISTORICAL_TRUST
    )

    print(
        "Trust threshold:",
        TRUST_THRESHOLD
    )

    print(
        "Client screening: DISABLED FOR CALIBRATION"
    )

    # --------------------------------------------------------
    # Initialize global model
    # --------------------------------------------------------

    print("\nInitializing global model...")

    global_model = IntrusionDetectionMLP().to(
        engine.DEVICE
    )

    print(
        "Execution device:",
        engine.DEVICE
    )

    # --------------------------------------------------------
    # Load unchanged clean validation data
    # --------------------------------------------------------

    print("\nLoading server-validation data...")

    validation_X, validation_y, feature_columns = (
        engine.load_validation_data()
    )

    # Make independent writable arrays.
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

    print(
        "Validation features:",
        len(feature_columns)
    )

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
    # Result storage
    # --------------------------------------------------------

    round_results = []

    trust_history = []

    poisoning_statistics = None

    baseline_validation_loss_history = []

    # ========================================================
    # 8. COMMUNICATION ROUNDS
    # ========================================================

    for round_number in range(
        1,
        TOTAL_ROUNDS + 1
    ):

        print("\n" + "=" * 70)

        print(
            f"SCENARIO K COMMUNICATION ROUND "
            f"{round_number}/{TOTAL_ROUNDS}"
        )

        print("=" * 70)

        client_states = {}

        client_updates = {}

        client_sizes = {}

        client_training_losses = {}

        candidate_validation_losses = {}

        # ----------------------------------------------------
        # STEP 1: Train all ten clients
        # ----------------------------------------------------

        print(
            "\n[1/7] Training non-IID client models..."
        )

        for client_id in client_ids:

            print(
                f"\nTraining client "
                f"{client_id:02d}/{NUM_CLIENTS}..."
            )

            X, original_y, sample_count = (
                engine.load_client_data(
                    client_id,
                    feature_columns
                )
            )

            # Apply the matched poisoning attack.
            training_y, poison_stats = (
                apply_label_flipping(
                    original_y,
                    client_id
                )
            )

            # Display detailed attack information once.
            if (
                client_id == MALICIOUS_CLIENT
                and round_number == 1
            ):

                poisoning_statistics = poison_stats

                print("\nMATCHED-STRENGTH ATTACK VERIFICATION")

                print(
                    "Original client records:",
                    f"{poison_stats['original_samples']:,}"
                )

                print(
                    "Eligible non-Benign records:",
                    f"{poison_stats['eligible_non_benign_records']:,}"
                )

                print(
                    "Poisoned records:",
                    f"{poison_stats['poisoned_records']:,}"
                )

                print(
                    "Matched poisoning fraction:",
                    f"{poison_stats['matched_poison_fraction']:.6%}"
                )

                print(
                    "Actual fraction of entire client:",
                    f"{poison_stats['actual_overall_client_poison_fraction']:.6%}"
                )

                print(
                    "Actual fraction of eligible records:",
                    f"{poison_stats['actual_eligible_record_poison_fraction']:.6%}"
                )

                print(
                    "Original labels changed:",
                    poison_stats["original_selected_label_counts"]
                )

            # Train local model.
            local_model, training_loss = (
                engine.train_local_model(
                    global_model,
                    X,
                    training_y
                )
            )

            # Save complete local state.
            local_state = {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }

            # Evaluate candidate model on the same clean
            # server-validation records.
            candidate_validation_loss = (
                engine.evaluate_validation_loss(
                    local_model,
                    validation_loader
                )
            )

            # Calculate model update vector.
            update_vector = flatten_model_update(
                local_state,
                global_model.state_dict()
            )

            client_states[client_id] = local_state

            client_updates[client_id] = (
                update_vector
            )

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
                "Training loss:",
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
        # STEP 2: Calculate update similarity
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
        # STEP 3: Calculate training consistency
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
        # STEP 4: Calculate historical and combined trust
        # ----------------------------------------------------

        print(
            "\n[4/7] Calculating combined trust..."
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
        # STEP 5: Calibration without exclusion
        # ----------------------------------------------------

        print(
            "\n[5/7] Checking clients without exclusion..."
        )

        accepted_clients, excluded_clients = (
            screen_clients(
                combined_trust_scores,
                threshold=TRUST_THRESHOLD
            )
        )

        # A zero threshold should accept every client because
        # valid trust scores are always between zero and one.
        if len(accepted_clients) != NUM_CLIENTS:

            raise RuntimeError(
                "Calibration expected all ten clients "
                f"to participate, but accepted "
                f"{len(accepted_clients)}."
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
        # STEP 6: Aggregate all client models
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

        # Update historical trust for the next round.
        historical_tracker.update(
            combined_trust_scores
        )

        updated_historical_scores = (
            historical_tracker.get_scores(
                client_ids
            )
        )

        # ----------------------------------------------------
        # STEP 7: Evaluate global model
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
        # Save detailed client trust information
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
        # Record round-level metrics
        # ----------------------------------------------------

        round_result = {

            "round": round_number,

            "validation_loss": float(
                validation_loss
            ),

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
            / f"matched_poison_noniid_round_{round_number:02d}.pt"
        )

        torch.save(
            {
                "round": round_number,

                "model_state_dict": (
                    global_model.state_dict()
                ),

                "validation_loss": validation_loss,

                "validation_metrics": (
                    validation_metrics
                ),

                "accepted_clients": (
                    accepted_clients
                ),

                "excluded_clients": (
                    excluded_clients
                ),

                "trust_scores": (
                    combined_trust_scores
                ),

                "aggregation_weights": (
                    aggregation_weights
                ),

                "malicious_client": (
                    MALICIOUS_CLIENT
                )
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
    # 9. SAVE EXPERIMENT SUMMARY
    # ========================================================

    print("\nSaving experiment results...")

    summary = {

        "scenario": "K_noniid_client07_matched_poison",

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

        "poisoning_seed": POISON_SEED,

        "poison_ratio_definition": (
            "Fraction of entire malicious client dataset"
        ),

        "matched_poisoning_fraction": float(
            POISON_CLIENT_FRACTION
        ),

        "poisoning_statistics": (
            poisoning_statistics
        ),

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
            "All clients participate. Trust scores are "
            "recorded without excluding any client."
        ),

        "test_set_used": False,

        "initial_validation_loss": float(
            initial_validation_loss
        ),

        "baseline_validation_loss_history": (
            baseline_validation_loss_history
        ),

        "round_results": round_results
    }

    summary_path = (
        OUTPUT_DIR
        / "matched_poison_noniid_trfa_summary.json"
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

    # ========================================================
    # 10. SAVE ROUND METRICS CSV
    # ========================================================

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
        / "matched_poison_noniid_trfa_metrics.csv"
    )

    pd.DataFrame(metrics_rows).to_csv(
        metrics_path,
        index=False
    )

    # ========================================================
    # 11. SAVE CLIENT TRUST HISTORY
    # ========================================================

    trust_path = (
        OUTPUT_DIR
        / "matched_poison_noniid_trfa_trust_history.csv"
    )

    pd.DataFrame(
        trust_history
    ).to_csv(
        trust_path,
        index=False
    )

    # ========================================================
    # 12. FINAL EXPERIMENT REPORT
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "SCENARIO K: NON-IID CLIENT 07 MATCHED-STRENGTH "
        "POISONING COMPLETED"
    )

    print("=" * 70)

    print(
        "Communication rounds:",
        TOTAL_ROUNDS
    )

    print(
        "Malicious client:",
        MALICIOUS_CLIENT
    )

    print(
        "Original client records:",
        poisoning_statistics["original_samples"]
    )

    print(
        "Eligible non-Benign records:",
        poisoning_statistics["eligible_non_benign_records"]
    )

    print(
        "Poisoned records:",
        poisoning_statistics["poisoned_records"]
    )

    print(
        "Matched overall-client poisoning fraction:",
        f"{poisoning_statistics['actual_overall_client_poison_fraction']:.6%}"
    )

    print(
        "Eligible-record poisoning fraction:",
        f"{poisoning_statistics['actual_eligible_record_poison_fraction']:.6%}"
    )

    print(
        "Final accepted clients:",
        len(round_results[-1]["accepted_clients"])
    )

    print(
        "Final excluded clients:",
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

    print(
        "Summary:",
        summary_path
    )

    print(
        "Round metrics:",
        metrics_path
    )

    print(
        "Trust history:",
        trust_path
    )

    print("\nScreening was disabled for calibration.")

    print("Independent test dataset was not used.")

    # Cleanup
    del validation_X
    del validation_y
    del validation_loader

    gc.collect()


# ============================================================
# 13. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_noniid_client07_matched_poison()
