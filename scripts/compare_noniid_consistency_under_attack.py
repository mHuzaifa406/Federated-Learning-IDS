
"""
Compare alternative consistency methods under non-IID poisoning.

All clients use the non-IID partition.
Client 10 performs targeted label flipping.

This is a one-round diagnostic, not a complete defense experiment.
"""

from pathlib import Path
import sys
import json
import gc
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import scripts.federated_engine as engine

from models.ids_model import IntrusionDetectionMLP

from trfa.trust_metrics import (
    flatten_model_update,
    calculate_update_similarity,
    calculate_combined_trust
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "federated_noniid_alpha05_seed43"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "scenario_H_noniid_trust_calibration"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "consistency_methods_under_poisoning.csv"
)

NUM_CLIENTS = 10
NUM_CLASSES = 7

MALICIOUS_CLIENT = 10
TARGET_LABEL = 0
POISON_RATIO = 0.20
POISON_SEED = 2027

TRAINING_SEED = 43

MIN_VALIDATION_SUPPORT = 500

TRUST_THRESHOLD = 0.7422773619

CLASS_NAMES = {
    0: "Benign",
    1: "Bot",
    2: "DoS",
    3: "Infiltration",
    4: "PortScan",
    5: "Web Attack",
    6: "Brute Force"
}


# ============================================================
# PER-CLASS VALIDATION LOSS
# ============================================================

def evaluate_per_class_loss(model, loader):

    model.eval()

    loss_sums = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    class_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    with torch.no_grad():

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(engine.DEVICE)
            batch_y = batch_y.to(engine.DEVICE)

            outputs = model(batch_X)

            losses = F.cross_entropy(
                outputs,
                batch_y,
                reduction="none"
            )

            for class_id in range(NUM_CLASSES):

                mask = batch_y == class_id

                if mask.any():

                    loss_sums[class_id] += (
                        losses[mask].sum().item()
                    )

                    class_counts[class_id] += (
                        mask.sum().item()
                    )

    losses = np.zeros(NUM_CLASSES, dtype=np.float64)

    for class_id in range(NUM_CLASSES):

        if class_counts[class_id] > 0:
            losses[class_id] = (
                loss_sums[class_id] / class_counts[class_id]
            )

        else:
            losses[class_id] = np.nan

    return losses, class_counts


# ============================================================
# SIGMOID CONSISTENCY SCALING
# ============================================================

def convert_improvements_to_scores(values):

    values = np.asarray(values, dtype=np.float64)

    nonzero = np.abs(
        values[np.abs(values) > 1e-12]
    )

    scale = (
        float(np.median(nonzero))
        if len(nonzero) > 0
        else 1.0
    )

    scale = max(scale, 1e-12)

    scores = []

    for improvement in values:

        z = float(
            np.clip(improvement / scale, -60.0, 60.0)
        )

        score = 1.0 / (1.0 + math.exp(-z))

        scores.append(float(score))

    return scores, scale


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():

    print("\nNON-IID CONSISTENCY METHODS UNDER POISONING")
    print("=" * 75)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Non-IID dataset not found: {DATA_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    engine.DATA_ROOT = DATA_ROOT

    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.set_num_threads(engine.CPU_THREADS)

    global_model = IntrusionDetectionMLP().to(
        engine.DEVICE
    )

    # --------------------------------------------------------
    # Load the unchanged common validation dataset
    # --------------------------------------------------------

    print("\nLoading clean server-validation data...")

    validation_X, validation_y, feature_columns = (
        engine.load_validation_data()
    )

    validation_loader = engine.create_validation_loader(
        validation_X,
        validation_y
    )

    baseline_losses, validation_counts = (
        evaluate_per_class_loss(
            global_model,
            validation_loader
        )
    )

    supported_classes = np.flatnonzero(
        validation_counts >= MIN_VALIDATION_SUPPORT
    )

    print("\nSupported validation classes:")
    print([
        CLASS_NAMES[int(class_id)]
        for class_id in supported_classes
    ])

    # --------------------------------------------------------
    # Collect per-client model updates and class effects
    # --------------------------------------------------------

    client_records = []
    client_updates = {}

    poisoning_statistics = None

    print("\nTraining local client models...")

    for client_id in range(1, NUM_CLIENTS + 1):

        print("\nClient:", client_id)

        X, original_y, sample_count = engine.load_client_data(
            client_id,
            feature_columns
        )

        observed_y = original_y.copy()

        # Apply the attack only to client 10.
        if client_id == MALICIOUS_CLIENT:

            eligible = np.flatnonzero(
                original_y != TARGET_LABEL
            )

            poison_count = int(
                np.floor(len(eligible) * POISON_RATIO)
            )

            rng = np.random.default_rng(
                POISON_SEED + client_id
            )

            selected_indices = rng.choice(
                eligible,
                size=poison_count,
                replace=False
            )

            original_selected = original_y[
                selected_indices
            ]

            observed_y[selected_indices] = TARGET_LABEL

            poisoning_statistics = {
                "malicious_client": client_id,
                "eligible_non_benign_records": int(len(eligible)),
                "poisoned_records": int(poison_count),
                "poison_ratio": POISON_RATIO,
                "overall_client_poison_fraction": (
                    poison_count / sample_count
                ),
                "original_selected_label_counts": {
                    str(int(label)): int(count)
                    for label, count in zip(
                        *np.unique(
                            original_selected,
                            return_counts=True
                        )
                    )
                }
            }

            print("\nATTACK APPLIED")
            print(
                "Eligible records:",
                f"{len(eligible):,}"
            )
            print(
                "Poisoned records:",
                f"{poison_count:,}"
            )

        # Clean labels represent the original data distribution.
        # Observed labels are those actually used during training.
        true_distribution = (
            np.bincount(
                original_y,
                minlength=NUM_CLASSES
            ) / len(original_y)
        )

        observed_distribution = (
            np.bincount(
                observed_y,
                minlength=NUM_CLASSES
            ) / len(observed_y)
        )

        local_model, training_loss = engine.train_local_model(
            global_model,
            X,
            observed_y
        )

        local_state = {
            key: value.detach().cpu().clone()
            for key, value in local_model.state_dict().items()
        }

        update_vector = flatten_model_update(
            local_state,
            global_model.state_dict()
        )

        client_updates[client_id] = update_vector

        candidate_losses, _ = evaluate_per_class_loss(
            local_model,
            validation_loader
        )

        for class_id in range(NUM_CLASSES):

            if (
                validation_counts[class_id] == 0
                or not np.isfinite(candidate_losses[class_id])
            ):
                continue

            improvement = (
                baseline_losses[class_id]
                - candidate_losses[class_id]
            )

            client_records.append({
                "Client": client_id,
                "Class_ID": class_id,
                "Class_Name": CLASS_NAMES[class_id],
                "Client_Original_Class_Share": (
                    true_distribution[class_id]
                ),
                "Client_Observed_Class_Share": (
                    observed_distribution[class_id]
                ),
                "Validation_Count": int(
                    validation_counts[class_id]
                ),
                "Validation_Loss_Improvement": float(
                    improvement
                ),
                "Client_Training_Loss": float(training_loss)
            })

        print("Training records:", f"{sample_count:,}")
        print("Training loss:", f"{training_loss:.6f}")

        del X, original_y, observed_y, local_model
        gc.collect()

    # --------------------------------------------------------
    # Build client-level improvements
    # --------------------------------------------------------

    detail = pd.DataFrame(client_records)

    client_methods = []

    for client_id, group in detail.groupby("Client"):

        improvements = group[
            "Validation_Loss_Improvement"
        ].to_numpy(dtype=float)

        validation_weights = group[
            "Validation_Count"
        ].to_numpy(dtype=float)

        true_shares = group[
            "Client_Original_Class_Share"
        ].to_numpy(dtype=float)

        observed_shares = group[
            "Client_Observed_Class_Share"
        ].to_numpy(dtype=float)

        global_improvement = float(
            np.average(
                improvements,
                weights=validation_weights
            )
        )

        class_mask = (
            group["Validation_Count"].to_numpy()
            >= MIN_VALIDATION_SUPPORT
        )

        supported_improvement = float(
            np.mean(improvements[class_mask])
        )

        # Uses the original clean label distribution.
        true_distribution_improvement = float(
            np.sum(improvements * true_shares)
            / np.sum(true_shares)
        )

        # Uses the labels actually used for training.
        # This may be influenced by label poisoning.
        observed_distribution_improvement = float(
            np.sum(improvements * observed_shares)
            / np.sum(observed_shares)
        )

        client_methods.append({
            "Client": int(client_id),
            "Malicious": client_id == MALICIOUS_CLIENT,
            "Global_Improvement": global_improvement,
            "Supported_Class_Improvement": supported_improvement,
            "True_Distribution_Improvement": (
                true_distribution_improvement
            ),
            "Observed_Distribution_Improvement": (
                observed_distribution_improvement
            )
        })

    results = pd.DataFrame(client_methods).sort_values(
        "Client"
    )

    # --------------------------------------------------------
    # Calculate consistency scores using the same
    # median-absolute-improvement scaling approach.
    # --------------------------------------------------------

    improvement_columns = {
        "Global_Consistency": "Global_Improvement",
        "Supported_Class_Consistency": (
            "Supported_Class_Improvement"
        ),
        "True_Distribution_Consistency": (
            "True_Distribution_Improvement"
        ),
        "Observed_Distribution_Consistency": (
            "Observed_Distribution_Improvement"
        )
    }

    scaling_information = {}

    for score_column, improvement_column in (
        improvement_columns.items()
    ):

        scores, scale = convert_improvements_to_scores(
            results[improvement_column].to_numpy()
        )

        results[score_column] = scores

        scaling_information[score_column] = scale

    # --------------------------------------------------------
    # Calculate similarity and combined diagnostic trust.
    # Historical trust is neutral at 0.5 for this one round.
    # --------------------------------------------------------

    similarity_scores = calculate_update_similarity(
        client_updates
    )

    results["Update_Similarity"] = results["Client"].map(
        similarity_scores
    )

    results["Historical_Trust"] = 0.50

    for method_name in improvement_columns:

        combined = []

        for _, row in results.iterrows():

            score = calculate_combined_trust(
                similarity_scores={
                    int(row["Client"]): float(
                        row["Update_Similarity"]
                    )
                },
                consistency_scores={
                    int(row["Client"]): float(
                        row[method_name]
                    )
                },
                historical_scores={
                    int(row["Client"]): 0.50
                }
            )

            combined.append(
                score[int(row["Client"])]
            )

        results[
            method_name.replace(
                "Consistency",
                "Combined_Trust"
            )
        ] = combined

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    display_columns = [
        "Client",
        "Malicious",
        "Update_Similarity",
        "Global_Consistency",
        "Supported_Class_Consistency",
        "True_Distribution_Consistency",
        "Observed_Distribution_Consistency",
        "Global_Combined_Trust",
        "Supported_Class_Combined_Trust",
        "True_Distribution_Combined_Trust",
        "Observed_Distribution_Combined_Trust"
    ]

    print("\nMETHOD COMPARISON BY CLIENT")
    print("-" * 115)

    print(
        results[display_columns].round(6).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Candidate threshold retrospective comparison.
    # This is a one-round diagnostic only.
    # --------------------------------------------------------

    print("\nRETROSPECTIVE THRESHOLD COMPARISON")
    print("-" * 115)

    for column in [
        "Global_Combined_Trust",
        "Supported_Class_Combined_Trust",
        "True_Distribution_Combined_Trust",
        "Observed_Distribution_Combined_Trust"
    ]:

        malicious_row = results[
            results["Malicious"]
        ]

        honest_rows = results[
            ~results["Malicious"]
        ]

        malicious_detected = bool(
            malicious_row[column].iloc[0] < TRUST_THRESHOLD
        )

        honest_excluded = int(
            (
                honest_rows[column] < TRUST_THRESHOLD
            ).sum()
        )

        print(
            f"{column:<42} | "
            f"Malicious detected={malicious_detected} | "
            f"Honest excluded={honest_excluded}/9"
        )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    results_path = OUTPUT_DIR / OUTPUT_FILE.name

    results.to_csv(
        results_path,
        index=False
    )

    detail_path = (
        OUTPUT_DIR
        / "consistency_methods_per_class_under_poisoning.csv"
    )

    detail.to_csv(
        detail_path,
        index=False
    )

    summary_path = (
        OUTPUT_DIR
        / "consistency_methods_under_poisoning.json"
    )

    summary = {
        "experiment": "One-round non-IID poisoning diagnostic",
        "training_seed": TRAINING_SEED,
        "poison_seed": POISON_SEED,
        "malicious_client": MALICIOUS_CLIENT,
        "target_label": TARGET_LABEL,
        "poison_ratio": POISON_RATIO,
        "poisoning_statistics": poisoning_statistics,
        "candidate_threshold": TRUST_THRESHOLD,
        "minimum_validation_support": MIN_VALIDATION_SUPPORT,
        "consistency_scaling": scaling_information,
        "scoring_note": (
            "Historical trust is neutral at 0.5. "
            "Scores are exploratory one-round diagnostics. "
            "They do not represent full multi-round defense performance."
        )
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(summary, file, indent=4)

    print("\nFILES SAVED")
    print("Client comparison:", results_path)
    print("Per-class details:", detail_path)
    print("Experiment metadata:", summary_path)

    print("\nANALYSIS COMPLETED")
    print(
        "No independent test records were used."
    )


if __name__ == "__main__":
    main()
