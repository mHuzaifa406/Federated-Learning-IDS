
"""
Scenario B: Poisoned Federated Averaging Without Defense

Configuration:
- 10 federated clients
- Client 10 is malicious
- 20% of non-Benign records in client 10 are label-flipped
- Target class: Benign (0)
- 10 communication rounds
- No defense mechanism

The same records are poisoned in every communication round.
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

from trfa.aggregation import aggregate_accepted_models


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

TOTAL_ROUNDS = 10

MALICIOUS_CLIENT = 10

POISON_RATIO = 0.20

TARGET_LABEL = 0

POISON_SEED = 2026

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "scenario_B_poisoned_no_defense"
)


# ============================================================
# LABEL-FLIPPING ATTACK
# ============================================================

def apply_label_flipping(y, client_id):
    """
    Poison a fixed, reproducible subset of non-Benign records.

    Poison ratio is calculated relative to the malicious
    client's non-Benign records.

    Returns:
        poisoned_labels
        poisoning_statistics
    """

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

    # Only records with a label different from the target
    # can undergo an actual label change.
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

    # Independent deterministic random generator.
    # Recreating it with the same seed ensures the same
    # records are poisoned in every communication round.
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

    # Change selected labels to Benign.
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
            "Unexpected mismatch in the number of poisoned labels."
        )

    return poisoned_labels, statistics


# ============================================================
# VALIDATION METRICS
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
# MAIN EXPERIMENT
# ============================================================

def run_poisoned_experiment():

    print("\n" + "=" * 65)
    print("SCENARIO B: POISONED FEDAVG WITHOUT DEFENSE")
    print("=" * 65)

    set_random_seed()
    torch.set_num_threads(CPU_THREADS)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Initialize global model
    # --------------------------------------------------------

    print("\nInitializing global model...")

    global_model = IntrusionDetectionMLP().to(DEVICE)

    print("Device:", DEVICE)
    print("Clients:", NUM_CLIENTS)
    print("Communication rounds:", TOTAL_ROUNDS)
    print("Malicious client:", MALICIOUS_CLIENT)
    print("Poison ratio:", POISON_RATIO)
    print("Target label:", TARGET_LABEL)

    # --------------------------------------------------------
    # Load clean validation dataset
    # --------------------------------------------------------

    print("\nLoading clean validation data...")

    validation_X, validation_y, feature_columns = (
        load_validation_data()
    )

    validation_loader = create_validation_loader(
        validation_X,
        validation_y
    )

    history = []

    poisoning_statistics = None

    # --------------------------------------------------------
    # Communication rounds
    # --------------------------------------------------------

    for round_number in range(1, TOTAL_ROUNDS + 1):

        print("\n" + "-" * 65)
        print(
            f"COMMUNICATION ROUND {round_number}/{TOTAL_ROUNDS}"
        )
        print("-" * 65)

        client_states = {}
        client_sizes = {}

        round_training_losses = {}

        # ----------------------------------------------------
        # Train all clients
        # ----------------------------------------------------

        for client_id in range(1, NUM_CLIENTS + 1):

            print(
                f"\nTraining client {client_id:02d}/{NUM_CLIENTS}..."
            )

            X, original_y, sample_count = load_client_data(
                client_id,
                feature_columns
            )

            # Apply the attack only to client 10.
            training_y, poison_stats = apply_label_flipping(
                original_y,
                client_id
            )

            if client_id == MALICIOUS_CLIENT:

                if poisoning_statistics is None:

                    poisoning_statistics = poison_stats

                    print("\nMALICIOUS CLIENT POISONING DETAILS")

                    print(
                        "Eligible non-Benign records:",
                        f"{poison_stats['eligible_non_benign_samples']:,}"
                    )

                    print(
                        "Poisoned records:",
                        f"{poison_stats['poisoned_samples']:,}"
                    )

                    print(
                        "Poison fraction of eligible records:",
                        f"{poison_stats['eligible_record_poison_fraction']:.4%}"
                    )

                    print(
                        "Poison fraction of entire client:",
                        f"{poison_stats['overall_client_poison_fraction']:.4%}"
                    )

                    print(
                        "Original labels changed:",
                        poison_stats["original_selected_label_counts"]
                    )

            # Train using the possibly poisoned labels.
            local_model, training_loss = train_local_model(
                global_model,
                X,
                training_y
            )

            client_states[client_id] = {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }

            # Poisoning does not change the client's sample count.
            client_sizes[client_id] = int(sample_count)

            round_training_losses[str(client_id)] = float(
                training_loss
            )

            print("  Training samples:", f"{sample_count:,}")
            print("  Local training loss:", f"{training_loss:.6f}")

            del X, original_y, training_y, local_model
            gc.collect()

        # ----------------------------------------------------
        # FedAvg without defense
        # ----------------------------------------------------

        print(
            "\nAggregating all client models using FedAvg..."
        )

        # No client screening or defense is applied.
        accepted_clients = list(range(1, NUM_CLIENTS + 1))

        aggregated_state, aggregation_weights = (
            aggregate_accepted_models(
                client_states=client_states,
                sample_sizes=client_sizes,
                accepted_client_ids=accepted_clients,
                minimum_accepted_clients=2
            )
        )

        global_model.load_state_dict(aggregated_state)

        # ----------------------------------------------------
        # Evaluate using clean validation data
        # ----------------------------------------------------

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

        print(
            "Total aggregation weight:",
            f"{sum(aggregation_weights.values()):.6f}"
        )

        # ----------------------------------------------------
        # Save round results
        # ----------------------------------------------------

        result = {
            "round": round_number,
            "validation_loss": float(validation_loss),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_weighted_f1": metrics["weighted_f1"],
            "clients_trained": NUM_CLIENTS,
            "malicious_client": MALICIOUS_CLIENT,
            "accepted_clients": accepted_clients,
            "excluded_clients": [],
            "client_training_losses": round_training_losses,
            "aggregation_weights": aggregation_weights
        }

        history.append(result)

        checkpoint_path = (
            OUTPUT_DIR /
            f"poisoned_fedavg_round_{round_number:02d}.pt"
        )

        torch.save(
            {
                "round": round_number,
                "model_state_dict": global_model.state_dict(),
                "validation_loss": validation_loss,
                "validation_metrics": metrics,
                "malicious_client": MALICIOUS_CLIENT,
                "poison_ratio": POISON_RATIO,
                "target_label": TARGET_LABEL
            },
            checkpoint_path
        )

        print("Checkpoint saved:", checkpoint_path)

        del client_states
        gc.collect()

    # ========================================================
    # SAVE EXPERIMENT SUMMARY
    # ========================================================

    summary = {
        "scenario": "B_poisoned_fedavg_without_defense",
        "dataset": "CIC-IDS-2017",
        "clients": NUM_CLIENTS,
        "communication_rounds": TOTAL_ROUNDS,
        "local_epochs": 1,
        "defense": "None",
        "attack_type": "Targeted label flipping",
        "malicious_client": MALICIOUS_CLIENT,
        "malicious_client_fraction": 1.0 / NUM_CLIENTS,
        "target_label": TARGET_LABEL,
        "poison_ratio_definition": (
            "Fraction of non-Benign records in the malicious client"
        ),
        "poison_ratio": POISON_RATIO,
        "poisoning_statistics": poisoning_statistics,
        "test_set_used_during_training": False,
        "round_results": history
    }

    summary_path = (
        OUTPUT_DIR / "poisoned_fedavg_summary.json"
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

    csv_path = OUTPUT_DIR / "poisoned_fedavg_history.csv"

    csv_rows = []

    for result in history:

        csv_rows.append({
            "Round": result["round"],
            "Validation_Loss": result["validation_loss"],
            "Validation_Accuracy": result["validation_accuracy"],
            "Validation_Macro_F1": result["validation_macro_f1"],
            "Validation_Weighted_F1": result["validation_weighted_f1"],
            "Accepted_Clients": len(result["accepted_clients"])
        })

    pd.DataFrame(csv_rows).to_csv(
        csv_path,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 65)
    print("SCENARIO B COMPLETED")
    print("=" * 65)

    print("Communication rounds:", TOTAL_ROUNDS)

    print(
        "Poisoned records:",
        poisoning_statistics["poisoned_samples"]
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

    print("Summary:", summary_path)
    print("CSV:", csv_path)

    print("\nIndependent test set was not used.")

    del validation_X, validation_y, validation_loader
    gc.collect()


if __name__ == "__main__":
    run_poisoned_experiment()
