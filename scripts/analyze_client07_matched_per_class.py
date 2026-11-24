
"""
Client 07 Matched-Poisoning Per-Class Diagnostic

Compares:
A. Honest Client 07 local model.
B. Poisoned Client 07 local model.

Both models start from exactly the same global initialization
and use the same local training random seed.

No global aggregation or client screening is performed.
"""

from pathlib import Path
import sys
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import scripts.federated_engine as engine

from models.ids_model import IntrusionDetectionMLP


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
    / "scenario_K_noniid_client07_matched_poison"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "client07_matched_per_class_effect.csv"
)

CLIENT_ID = 7

TARGET_LABEL = 0

POISON_SEED = 2027

POISON_CLIENT_FRACTION = 8676 / 213858

TRAINING_SEED = 43

NUM_CLASSES = 7

CLASS_NAMES = {
    0: "Benign",
    1: "Bot",
    2: "DoS",
    3: "Infiltration",
    4: "PortScan",
    5: "Web Attack",
    6: "Brute Force"
}

MIN_SUPPORTED_VALIDATION_COUNT = 500


# ============================================================
# REPRODUCIBILITY
# ============================================================

def reset_training_seed():

    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)

    torch.set_num_threads(
        engine.CPU_THREADS
    )


# ============================================================
# MATCHED LABEL-FLIPPING ATTACK
# ============================================================

def poison_client_labels(original_y):

    poisoned_y = original_y.copy()

    eligible_indices = np.flatnonzero(
        original_y != TARGET_LABEL
    )

    poison_count = int(
        np.floor(
            len(original_y) * POISON_CLIENT_FRACTION
        )
    )

    if poison_count > len(eligible_indices):
        raise ValueError(
            "Insufficient eligible records for poisoning."
        )

    rng = np.random.default_rng(
        POISON_SEED + CLIENT_ID
    )

    selected_indices = rng.choice(
        eligible_indices,
        size=poison_count,
        replace=False
    )

    selected_original_labels = original_y[
        selected_indices
    ]

    poisoned_y[selected_indices] = TARGET_LABEL

    statistics = {
        "original_records": len(original_y),
        "eligible_records": len(eligible_indices),
        "poisoned_records": poison_count,
        "overall_client_fraction": (
            poison_count / len(original_y)
        ),
        "eligible_record_fraction": (
            poison_count / len(eligible_indices)
        )
    }

    return (
        poisoned_y,
        selected_original_labels,
        statistics
    )


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

    class_losses = np.full(
        NUM_CLASSES,
        np.nan,
        dtype=np.float64
    )

    for class_id in range(NUM_CLASSES):

        if class_counts[class_id] > 0:

            class_losses[class_id] = (
                loss_sums[class_id]
                / class_counts[class_id]
            )

    return class_losses, class_counts


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():

    print("\nCLIENT 07 MATCHED-POISONING PER-CLASS ANALYSIS")
    print("=" * 75)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(DATA_ROOT)

    engine.DATA_ROOT = DATA_ROOT

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Initialize the same global model for both comparisons
    # --------------------------------------------------------

    reset_training_seed()

    global_model = IntrusionDetectionMLP().to(
        engine.DEVICE
    )

    # --------------------------------------------------------
    # Load Client 07's original records
    # --------------------------------------------------------

    print("\nLoading Client 07 data...")

    validation_X, validation_y, feature_columns = (
        engine.load_validation_data()
    )

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

    validation_loader = engine.create_validation_loader(
        validation_X,
        validation_y
    )

    X, original_y, sample_count = (
        engine.load_client_data(
            CLIENT_ID,
            feature_columns
        )
    )

    print("Client records:", f"{sample_count:,}")

    original_class_counts = np.bincount(
        original_y,
        minlength=NUM_CLASSES
    )

    print("\nOriginal client class distribution:")

    for class_id in range(NUM_CLASSES):

        print(
            f"Class {class_id} ({CLASS_NAMES[class_id]}): "
            f"{original_class_counts[class_id]:,}"
        )

    # --------------------------------------------------------
    # Create matched poisoned labels
    # --------------------------------------------------------

    poisoned_y, selected_original_labels, poison_stats = (
        poison_client_labels(original_y)
    )

    print("\nPOISONING VERIFICATION")

    for key, value in poison_stats.items():

        print(f"{key}: {value}")

    unique_labels, label_counts = np.unique(
        selected_original_labels,
        return_counts=True
    )

    print("\nOriginal labels changed to Benign:")

    for label, count in zip(unique_labels, label_counts):

        print(
            f"{CLASS_NAMES[int(label)]}: {int(count):,}"
        )

    # --------------------------------------------------------
    # Baseline global-model per-class losses
    # --------------------------------------------------------

    print("\nCalculating initial global-model losses...")

    baseline_losses, validation_counts = (
        evaluate_per_class_loss(
            global_model,
            validation_loader
        )
    )

    # --------------------------------------------------------
    # Train the clean client model
    # --------------------------------------------------------

    print("\nTraining clean Client 07 model...")

    reset_training_seed()

    clean_model, clean_training_loss = (
        engine.train_local_model(
            global_model,
            X,
            original_y.copy()
        )
    )

    clean_losses, _ = evaluate_per_class_loss(
        clean_model,
        validation_loader
    )

    # --------------------------------------------------------
    # Train the poisoned client model
    # --------------------------------------------------------

    print("\nTraining poisoned Client 07 model...")

    # Reset seed so the two local runs use the same
    # initialization, shuffling and dropout randomness.
    reset_training_seed()

    poisoned_model, poisoned_training_loss = (
        engine.train_local_model(
            global_model,
            X,
            poisoned_y.copy()
        )
    )

    poisoned_losses, _ = evaluate_per_class_loss(
        poisoned_model,
        validation_loader
    )

    # --------------------------------------------------------
    # Compare class-level effects
    # --------------------------------------------------------

    results = []

    print("\nPER-CLASS VALIDATION COMPARISON")
    print("=" * 110)

    print(
        f"{'Class':<16}"
        f"{'Count':>10}"
        f"{'Baseline':>14}"
        f"{'Clean':>14}"
        f"{'Poisoned':>14}"
        f"{'Clean Improve':>16}"
        f"{'Poison Improve':>17}"
        f"{'Poison Effect':>16}"
    )

    for class_id in range(NUM_CLASSES):

        if validation_counts[class_id] == 0:
            continue

        baseline = baseline_losses[class_id]

        clean_loss = clean_losses[class_id]

        poisoned_loss = poisoned_losses[class_id]

        # Positive improvement means loss decreased.
        clean_improvement = baseline - clean_loss

        poisoned_improvement = (
            baseline - poisoned_loss
        )

        # Positive effect means poisoning worsened loss
        # compared with the clean client model.
        poison_effect = (
            poisoned_loss - clean_loss
        )

        results.append({
            "Class_ID": class_id,
            "Class_Name": CLASS_NAMES[class_id],
            "Validation_Count": int(
                validation_counts[class_id]
            ),
            "Baseline_Loss": float(baseline),
            "Clean_Candidate_Loss": float(clean_loss),
            "Poisoned_Candidate_Loss": float(poisoned_loss),
            "Clean_Improvement": float(clean_improvement),
            "Poisoned_Improvement": float(
                poisoned_improvement
            ),
            "Poison_Effect": float(poison_effect)
        })

        print(
            f"{CLASS_NAMES[class_id]:<16}"
            f"{validation_counts[class_id]:>10,}"
            f"{baseline:>14.6f}"
            f"{clean_loss:>14.6f}"
            f"{poisoned_loss:>14.6f}"
            f"{clean_improvement:>16.6f}"
            f"{poisoned_improvement:>17.6f}"
            f"{poison_effect:>16.6f}"
        )

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Overall validation loss comparison
    # --------------------------------------------------------

    clean_model_loss = engine.evaluate_validation_loss(
        clean_model,
        validation_loader
    )

    poisoned_model_loss = engine.evaluate_validation_loss(
        poisoned_model,
        validation_loader
    )

    baseline_model_loss = engine.evaluate_validation_loss(
        global_model,
        validation_loader
    )

    print("\nOVERALL MODEL COMPARISON")
    print("-" * 75)

    print(
        "Initial global-model loss:",
        f"{baseline_model_loss:.6f}"
    )

    print(
        "Clean candidate validation loss:",
        f"{clean_model_loss:.6f}"
    )

    print(
        "Poisoned candidate validation loss:",
        f"{poisoned_model_loss:.6f}"
    )

    print(
        "Overall poisoning effect:",
        f"{poisoned_model_loss - clean_model_loss:+.6f}"
    )

    # --------------------------------------------------------
    # Supported-class macro comparison
    # --------------------------------------------------------

    supported = result_df[
        result_df["Validation_Count"]
        >= MIN_SUPPORTED_VALIDATION_COUNT
    ]

    if not supported.empty:

        clean_macro_improvement = (
            supported["Clean_Improvement"].mean()
        )

        poisoned_macro_improvement = (
            supported["Poisoned_Improvement"].mean()
        )

        print("\nSUPPORTED-CLASS MACRO COMPARISON")
        print("-" * 75)

        print(
            "Clean macro improvement:",
            f"{clean_macro_improvement:.6f}"
        )

        print(
            "Poisoned macro improvement:",
            f"{poisoned_macro_improvement:.6f}"
        )

        print(
            "Difference:",
            f"{poisoned_macro_improvement - clean_macro_improvement:+.6f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "client07_matched_per_class_effect.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    metadata_path = (
        OUTPUT_DIR
        / "client07_matched_per_class_metadata.txt"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Client 07 matched poisoning per-class diagnostic\n"
        )

        file.write(
            f"Original records: {sample_count}\n"
        )

        file.write(
            f"Poisoned records: {poison_stats['poisoned_records']}\n"
        )

        file.write(
            f"Overall-client poisoning fraction: "
            f"{poison_stats['overall_client_fraction']:.8%}\n"
        )

        file.write(
            f"Clean training loss: {clean_training_loss:.8f}\n"
        )

        file.write(
            f"Poisoned training loss: {poisoned_training_loss:.8f}\n"
        )

        file.write(
            f"Clean candidate validation loss: "
            f"{clean_model_loss:.8f}\n"
        )

        file.write(
            f"Poisoned candidate validation loss: "
            f"{poisoned_model_loss:.8f}\n"
        )

    print("\nFILES SAVED")
    print("Per-class comparison:", output_path)
    print("Metadata:", metadata_path)

    print("\nINTERPRETATION:")
    print(
        "Positive Poison_Effect means the poisoned model "
        "has higher validation loss for that class."
    )

    print(
        "Negative Poison_Effect means the poisoned model "
        "has lower validation loss for that class."
    )

    print(
        "Very rare validation classes have less stable "
        "loss estimates."
    )

    print("\nNo independent test data was used.")

    del X, original_y, poisoned_y
    del validation_X, validation_y
    del validation_loader
    del clean_model, poisoned_model

    gc.collect()

    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
