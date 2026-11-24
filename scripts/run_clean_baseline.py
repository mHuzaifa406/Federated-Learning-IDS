
"""
Scenario A: Clean Federated Averaging Baseline

10 clients, 10 communication rounds, one local epoch
per client per round. No poisoning and no defense.
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

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

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
# CONFIGURATION
# ============================================================

TOTAL_ROUNDS = 10

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "scenario_A_clean_fedavg"
)

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
# MAIN EXPERIMENT
# ============================================================

def run_clean_baseline():

    print("\n" + "=" * 65)
    print("SCENARIO A: CLEAN FEDAVG BASELINE")
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

    # --------------------------------------------------------
    # Load validation dataset
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

    # --------------------------------------------------------
    # Communication rounds
    # --------------------------------------------------------

    for round_number in range(1, TOTAL_ROUNDS + 1):

        print("\n" + "-" * 65)
        print(f"COMMUNICATION ROUND {round_number}/{TOTAL_ROUNDS}")
        print("-" * 65)

        client_states = {}
        client_sizes = {}

        round_training_losses = {}

        # Train each client using the current global model.
        for client_id in range(1, NUM_CLIENTS + 1):

            print(
                f"\nTraining client {client_id:02d}/{NUM_CLIENTS}..."
            )

            X, y, sample_count = load_client_data(
                client_id,
                feature_columns
            )

            local_model, training_loss = train_local_model(
                global_model,
                X,
                y
            )

            client_states[client_id] = {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }

            client_sizes[client_id] = int(sample_count)

            round_training_losses[str(client_id)] = (
                float(training_loss)
            )

            print("  Samples:", f"{sample_count:,}")
            print("  Training loss:", f"{training_loss:.6f}")

            del X, y, local_model
            gc.collect()

        # ----------------------------------------------------
        # Standard FedAvg: all clients are accepted.
        # ----------------------------------------------------

        print("\nAggregating all client models using FedAvg...")

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
        # Evaluate global model using validation data only.
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
        # Record round results
        # ----------------------------------------------------

        round_result = {
            "round": round_number,
            "validation_loss": float(validation_loss),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_weighted_f1": metrics["weighted_f1"],
            "clients_trained": NUM_CLIENTS,
            "accepted_clients": accepted_clients,
            "excluded_clients": [],
            "client_training_losses": round_training_losses,
            "aggregation_weights": aggregation_weights
        }

        history.append(round_result)

        # ----------------------------------------------------
        # Save round checkpoint
        # ----------------------------------------------------

        checkpoint_path = (
            OUTPUT_DIR /
            f"clean_fedavg_round_{round_number:02d}.pt"
        )

        torch.save(
            {
                "round": round_number,
                "model_state_dict": global_model.state_dict(),
                "validation_loss": validation_loss,
                "validation_metrics": metrics
            },
            checkpoint_path
        )

        print("Checkpoint saved:", checkpoint_path)

        del client_states
        gc.collect()

    # ========================================================
    # SAVE COMPLETE EXPERIMENT RESULTS
    # ========================================================

    summary_path = OUTPUT_DIR / "clean_fedavg_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "scenario": "A_clean_fedavg",
                "dataset": "CIC-IDS-2017",
                "clients": NUM_CLIENTS,
                "communication_rounds": TOTAL_ROUNDS,
                "local_epochs": 1,
                "defense": "None",
                "poisoning": "None",
                "test_set_used_during_training": False,
                "round_results": history
            },
            file,
            indent=4
        )

    # Also save a compact CSV for analysis.
    csv_data = []

    for result in history:

        csv_data.append({
            "Round": result["round"],
            "Validation_Loss": result["validation_loss"],
            "Validation_Accuracy": result["validation_accuracy"],
            "Validation_Macro_F1": result["validation_macro_f1"],
            "Validation_Weighted_F1": result["validation_weighted_f1"],
            "Accepted_Clients": len(result["accepted_clients"])
        })

    csv_path = OUTPUT_DIR / "clean_fedavg_history.csv"

    pd.DataFrame(csv_data).to_csv(
        csv_path,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("SCENARIO A COMPLETED")
    print("=" * 65)

    print("Communication rounds:", TOTAL_ROUNDS)

    print(
        "Final validation accuracy:",
        f"{history[-1]['validation_accuracy'] * 100:.4f}%"
    )

    print(
        "Final validation macro F1:",
        f"{history[-1]['validation_macro_f1']:.6f}"
    )

    print("Summary:", summary_path)
    print("CSV:", csv_path)

    print("\nIndependent test set was not used.")

    del validation_X, validation_y, validation_loader
    gc.collect()


if __name__ == "__main__":
    run_clean_baseline()
