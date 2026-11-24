
"""
Non-IID Per-Class Update Diagnostic

All clients are honest.

Measures how each client's local model update affects
validation loss for each of the seven intrusion classes.

No poisoning or client screening is applied.
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
    / "scenario_H_noniid_trust_calibration"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "per_class_candidate_diagnostic.csv"
)

NUM_CLASSES = 7
NUM_CLIENTS = 10
SEED = 43

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

    class_loss_sums = np.zeros(
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

                    class_loss_sums[class_id] += (
                        losses[mask].sum().item()
                    )

                    class_counts[class_id] += (
                        mask.sum().item()
                    )

    class_losses = {}

    for class_id in range(NUM_CLASSES):

        if class_counts[class_id] > 0:

            class_losses[class_id] = (
                class_loss_sums[class_id]
                / class_counts[class_id]
            )

        else:

            class_losses[class_id] = None

    return class_losses, class_counts


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():

    print("\nNON-IID PER-CLASS UPDATE DIAGNOSTIC")
    print("=" * 70)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Non-IID dataset missing: {DATA_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Use the separate non-IID data directory.
    engine.DATA_ROOT = DATA_ROOT

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(engine.CPU_THREADS)

    # --------------------------------------------------------
    # Initialize global model
    # --------------------------------------------------------

    global_model = IntrusionDetectionMLP().to(
        engine.DEVICE
    )

    # --------------------------------------------------------
    # Load unchanged server validation data
    # --------------------------------------------------------

    print("\nLoading server-validation data...")

    validation_X, validation_y, feature_columns = (
        engine.load_validation_data()
    )

    validation_loader = engine.create_validation_loader(
        validation_X,
        validation_y
    )

    # Calculate global model's per-class baseline loss.
    baseline_losses, validation_counts = (
        evaluate_per_class_loss(
            global_model,
            validation_loader
        )
    )

    print("\nGLOBAL VALIDATION BASELINE")

    for class_id in range(NUM_CLASSES):

        print(
            f"Class {class_id} "
            f"({CLASS_NAMES[class_id]}): "
            f"records={validation_counts[class_id]:,} | "
            f"loss={baseline_losses[class_id]}"
        )

    # --------------------------------------------------------
    # Train each honest client independently from global model
    # --------------------------------------------------------

    all_results = []

    for client_id in range(1, NUM_CLIENTS + 1):

        print("\n" + "-" * 65)
        print(f"ANALYZING HONEST CLIENT {client_id:02d}")
        print("-" * 65)

        X, y, sample_count = engine.load_client_data(
            client_id,
            feature_columns
        )

        local_class_counts = np.bincount(
            y,
            minlength=NUM_CLASSES
        )

        local_model, training_loss = (
            engine.train_local_model(
                global_model,
                X,
                y
            )
        )

        candidate_losses, _ = (
            evaluate_per_class_loss(
                local_model,
                validation_loader
            )
        )

        print("Training records:", f"{sample_count:,}")
        print("Local training loss:", f"{training_loss:.6f}")

        for class_id in range(NUM_CLASSES):

            baseline = baseline_losses[class_id]
            candidate = candidate_losses[class_id]

            if baseline is None or candidate is None:
                continue

            # Positive improvement means candidate loss decreased.
            improvement = baseline - candidate

            local_share = (
                local_class_counts[class_id] / sample_count
            )

            result = {
                "Client": client_id,
                "Class_ID": class_id,
                "Class_Name": CLASS_NAMES[class_id],
                "Client_Training_Count": int(
                    local_class_counts[class_id]
                ),
                "Client_Class_Share": float(local_share),
                "Validation_Count": int(
                    validation_counts[class_id]
                ),
                "Baseline_Class_Loss": float(baseline),
                "Candidate_Class_Loss": float(candidate),
                "Validation_Loss_Improvement": float(improvement)
            }

            all_results.append(result)

        # Show the client's three most represented classes.
        dominant_classes = np.argsort(
            local_class_counts
        )[::-1][:3]

        print("\nMain client classes:")

        for class_id in dominant_classes:

            if local_class_counts[class_id] == 0:
                continue

            baseline = baseline_losses[class_id]
            candidate = candidate_losses[class_id]

            improvement = baseline - candidate

            print(
                f"  {CLASS_NAMES[class_id]:15} | "
                f"Client share={local_class_counts[class_id] / sample_count:.3%} | "
                f"Validation improvement={improvement:+.6f}"
            )

        del X, y, local_model
        gc.collect()

    # --------------------------------------------------------
    # Save detailed output
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("PER-CLASS DIAGNOSTIC COMPLETED")
    print("=" * 70)

    print("Clients analyzed:", NUM_CLIENTS)
    print("Classes analyzed:", NUM_CLASSES)
    print("Diagnostic records:", len(results_df))

    print("Saved:", OUTPUT_FILE)

    print("\nPositive loss improvement = validation loss decreased.")
    print("Negative loss improvement = validation loss increased.")

    print(
        "\nNote: Very rare validation classes have fewer examples, "
        "so their loss estimates are less stable."
    )

    print("\nNo test data was used.")

    del validation_X, validation_y, validation_loader
    gc.collect()


if __name__ == "__main__":
    main()
