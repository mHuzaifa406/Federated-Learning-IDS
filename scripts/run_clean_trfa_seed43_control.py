
"""
Clean TRFA Control Experiment

All ten clients are honest.
No poisoning is applied.
The candidate threshold is tested over ten rounds.

Purpose:
Measure whether TRFA incorrectly excludes honest clients.
"""

from pathlib import Path
import sys
import json
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
# CONFIGURATION
# ============================================================

TOTAL_ROUNDS = 10

ALPHA = 1.0 / 3.0
BETA = 1.0 / 3.0
GAMMA = 1.0 / 3.0

TRUST_THRESHOLD = 0.7422773619

HISTORY_RETENTION = 0.90
INITIAL_HISTORICAL_TRUST = 0.50

OUTPUT_DIR = (
    PROJECT_ROOT / "results"
    / "scenario_F_clean_trfa_seed43_control"
)


# ============================================================
# VALIDATION METRICS
# ============================================================

def evaluate_metrics(model, loader):

    model.eval()

    labels = []
    predictions = []

    with torch.no_grad():

        for X, y in loader:

            outputs = model(X.to(DEVICE))

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            labels.extend(y.numpy().tolist())
            predictions.extend(
                predicted.cpu().numpy().tolist()
            )

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0
            )
        )
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_clean_control():

    print("\n" + "=" * 65)
    print("CLEAN TRFA CONTROL EXPERIMENT")
    print("=" * 65)

    set_random_seed()
    import numpy as np
    np.random.seed(43)
    torch.manual_seed(43)
    torch.set_num_threads(CPU_THREADS)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    global_model = IntrusionDetectionMLP().to(DEVICE)

    validation_X, validation_y, feature_columns = (
        load_validation_data()
    )

    validation_loader = create_validation_loader(
        validation_X,
        validation_y
    )

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
    # COMMUNICATION ROUNDS
    # --------------------------------------------------------

    for round_number in range(1, TOTAL_ROUNDS + 1):

        print("\n" + "-" * 65)
        print(
            f"ROUND {round_number}/{TOTAL_ROUNDS}"
        )
        print("-" * 65)

        client_states = {}
        client_updates = {}
        client_sizes = {}
        candidate_losses = {}

        # -----------------------------------------------
        # Train all honest clients
        # -----------------------------------------------

        for client_id in client_ids:

            print(f"Training client {client_id:02d}...")

            X, y, sample_count = load_client_data(
                client_id,
                feature_columns
            )

            # No label poisoning is applied.
            local_model, training_loss = train_local_model(
                global_model,
                X,
                y
            )

            local_state = {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }

            candidate_losses[client_id] = (
                evaluate_validation_loss(
                    local_model,
                    validation_loader
                )
            )

            update_vector = flatten_model_update(
                local_state,
                global_model.state_dict()
            )

            client_states[client_id] = local_state
            client_updates[client_id] = update_vector
            client_sizes[client_id] = int(sample_count)

            print(
                f"  Samples: {sample_count:,} | "
                f"Training loss: {training_loss:.6f}"
            )

            del X, y, local_model
            gc.collect()

        # -----------------------------------------------
        # Calculate trust signals
        # -----------------------------------------------

        reference_update = calculate_reference_update(
            client_updates
        )

        similarity = calculate_update_similarity(
            client_updates,
            reference_update
        )

        baseline_loss = evaluate_validation_loss(
            global_model,
            validation_loader
        )

        consistency = calculate_training_consistency(
            baseline_loss,
            candidate_losses
        )

        historical = tracker.get_scores(client_ids)

        combined = calculate_combined_trust(
            similarity_scores=similarity,
            consistency_scores=consistency,
            historical_scores=historical,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA
        )

        # -----------------------------------------------
        # Screen all clients
        # -----------------------------------------------

        accepted, excluded = screen_clients(
            combined,
            threshold=TRUST_THRESHOLD
        )

        # Every client is honest in this experiment.
        false_exclusions = list(excluded)

        total_false_exclusions += len(false_exclusions)

        print("\nTrust threshold:", TRUST_THRESHOLD)
        print("Accepted clients:", accepted)
        print("Excluded honest clients:", excluded)

        # Save detailed client-level trust scores.
        for client_id in client_ids:

            trust_history.append({
                "Round": round_number,
                "Client": client_id,
                "Similarity": similarity[client_id],
                "Consistency": consistency[client_id],
                "Historical_Trust": historical[client_id],
                "Combined_Trust": combined[client_id],
                "Accepted": client_id in accepted,
                "False_Exclusion": client_id in excluded
            })

        if len(accepted) < 2:
            raise RuntimeError(
                "Fewer than two clients remain. "
                "Aggregation stopped."
            )

        # -----------------------------------------------
        # Aggregate accepted honest clients
        # -----------------------------------------------

        aggregated_state, weights = (
            aggregate_accepted_models(
                client_states=client_states,
                sample_sizes=client_sizes,
                accepted_client_ids=accepted,
                minimum_accepted_clients=2
            )
        )

        global_model.load_state_dict(aggregated_state)

        tracker.update(combined)

        # -----------------------------------------------
        # Validation evaluation
        # -----------------------------------------------

        validation_loss = evaluate_validation_loss(
            global_model,
            validation_loader
        )

        metrics = evaluate_metrics(
            global_model,
            validation_loader
        )

        print("\nROUND RESULTS")

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

        result = {
            "Round": round_number,
            "Validation_Loss": validation_loss,
            "Validation_Accuracy": metrics["accuracy"],
            "Validation_Macro_F1": metrics["macro_f1"],
            "Validation_Weighted_F1": metrics["weighted_f1"],
            "Accepted_Clients": len(accepted),
            "False_Exclusions": len(false_exclusions)
        }

        round_results.append(result)

        # Save checkpoint.
        torch.save(
            {
                "round": round_number,
                "model_state_dict": global_model.state_dict(),
                "validation_metrics": metrics,
                "accepted_clients": accepted,
                "excluded_clients": excluded
            },
            OUTPUT_DIR / f"clean_trfa_round_{round_number:02d}.pt"
        )

        del client_states, client_updates
        gc.collect()

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    metrics_csv = OUTPUT_DIR / "clean_trfa_metrics.csv"
    trust_csv = OUTPUT_DIR / "clean_trfa_trust_history.csv"
    summary_json = OUTPUT_DIR / "clean_trfa_summary.json"

    pd.DataFrame(round_results).to_csv(
        metrics_csv,
        index=False
    )

    pd.DataFrame(trust_history).to_csv(
        trust_csv,
        index=False
    )

    total_honest_opportunities = (
        NUM_CLIENTS * TOTAL_ROUNDS
    )

    false_exclusion_rate = (
        total_false_exclusions /
        total_honest_opportunities
    )

    summary = {
        "scenario": "F_clean_trfa_seed43_control",
        "clients": NUM_CLIENTS,
        "communication_rounds": TOTAL_ROUNDS,
        "poisoning": "None",
        "trust_threshold": TRUST_THRESHOLD,
        "total_honest_opportunities": total_honest_opportunities,
        "total_false_exclusions": total_false_exclusions,
        "honest_false_exclusion_rate": false_exclusion_rate,
        "test_set_used": False,
        "round_results": round_results
    }

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("CLEAN TRFA CONTROL COMPLETED")
    print("=" * 65)

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

    print("Summary:", summary_json)
    print("Metrics:", metrics_csv)
    print("Trust history:", trust_csv)

    print("\nIndependent test dataset was not used.")

    del validation_X, validation_y, validation_loader
    gc.collect()


if __name__ == "__main__":
    run_clean_control()
