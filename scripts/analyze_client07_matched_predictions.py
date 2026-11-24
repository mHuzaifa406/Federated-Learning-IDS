"""
Scenario K: Prediction-Level Matched-Poisoning Analysis

Compares three models on the same clean server-validation data:
1. Trained clean non-IID global model from Scenario H, round 10
2. Clean Client 07 local candidate model
3. Matched-poisoned Client 07 local candidate model

The two local candidate models start from the same trained global checkpoint.
Clean and poisoned local training use matched random seeds.

Metrics:
- Accuracy, macro F1 and weighted F1
- Per-class precision, recall and F1
- Confusion matrices
- Predicted class counts for every class
- Overall validation attack-to-Benign rate
- Attack-class-specific attack-to-Benign rates

IMPORTANT:
This is a local-update diagnostic using server-validation data. It is not
an independent test evaluation and does not compare fully aggregated
federated models. The validation attack-to-Benign rate is a diagnostic
proxy, not a conventional backdoor attack-success evaluation on a
separate attack-triggered test set.
"""

from pathlib import Path
import sys
import gc

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

# Use a new output folder so the earlier Scenario K diagnostic is preserved.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "scenario_K_noniid_client07_matched_poison"
    / "trained_checkpoint_prediction_diagnostic"
)

GLOBAL_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "scenario_H_noniid_trust_calibration"
    / "clean_noniid_round_10.pt"
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
    6: "Brute Force",
}

LABEL_IDS = list(range(NUM_CLASSES))


# ============================================================
# RANDOM SEED
# ============================================================

def reset_training_seed():
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)
    torch.set_num_threads(engine.CPU_THREADS)


# ============================================================
# MATCHED POISONING
# ============================================================

def create_poisoned_labels(original_y):
    """Flip a fixed fraction of Client 07's records to the Benign label."""
    poisoned_y = original_y.copy()
    eligible_indices = np.flatnonzero(original_y != TARGET_LABEL)

    poison_count = int(np.floor(len(original_y) * POISON_CLIENT_FRACTION))
    if poison_count > len(eligible_indices):
        raise ValueError("Insufficient eligible records for poisoning.")

    rng = np.random.default_rng(POISON_SEED + CLIENT_ID)
    selected_indices = rng.choice(
        eligible_indices,
        size=poison_count,
        replace=False,
    )

    poisoned_y[selected_indices] = TARGET_LABEL
    actual_changes = int(np.sum(poisoned_y != original_y))

    if actual_changes != poison_count:
        raise RuntimeError("Unexpected poisoning count.")

    return poisoned_y, actual_changes


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def get_predictions(model, loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(engine.DEVICE)
            outputs = model(batch_X)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(batch_y.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    return (
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_predictions, dtype=np.int64),
    )


def print_prediction_distribution(model_name, predictions):
    """Print the number of validation records predicted as each class."""
    counts = np.bincount(predictions, minlength=NUM_CLASSES)
    print(f"\nPredicted class counts — {model_name}")
    print("-" * 55)
    for class_id, count in enumerate(counts):
        print(f"{CLASS_NAMES[class_id]:15s}: {count:>10,}")
    print(f"{'TOTAL':15s}: {int(counts.sum()):>10,}")


# ============================================================
# MODEL METRICS
# ============================================================

def calculate_model_metrics(actual_labels, predictions, model_name):
    precision, recall, f1, support = precision_recall_fscore_support(
        actual_labels,
        predictions,
        labels=LABEL_IDS,
        zero_division=0,
    )

    accuracy = accuracy_score(actual_labels, predictions)
    macro_f1 = precision_recall_fscore_support(
        actual_labels,
        predictions,
        average="macro",
        zero_division=0,
    )[2]
    weighted_f1 = precision_recall_fscore_support(
        actual_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )[2]

    class_results = []
    for index, class_id in enumerate(LABEL_IDS):
        class_results.append(
            {
                "Model": model_name,
                "Class_ID": class_id,
                "Class_Name": CLASS_NAMES[class_id],
                "Precision": float(precision[index]),
                "Recall": float(recall[index]),
                "F1": float(f1[index]),
                "Support": int(support[index]),
            }
        )

    prediction_counts = np.bincount(predictions, minlength=NUM_CLASSES)
    summary = {
        "Model": model_name,
        "Accuracy": float(accuracy),
        "Macro_F1": float(macro_f1),
        "Weighted_F1": float(weighted_f1),
        "Total_Validation_Records": int(len(actual_labels)),
    }

    prediction_count_results = []
    for class_id in LABEL_IDS:
        prediction_count_results.append(
            {
                "Model": model_name,
                "Class_ID": class_id,
                "Class_Name": CLASS_NAMES[class_id],
                "Predicted_Count": int(prediction_counts[class_id]),
                "Predicted_Fraction": float(
                    prediction_counts[class_id] / len(predictions)
                ),
            }
        )

    return summary, class_results, prediction_count_results


# ============================================================
# VALIDATION ATTACK-TO-BENIGN RATE
# ============================================================

def calculate_attack_success(actual_labels, predictions):
    """Measure attack-class validation records classified as Benign.

    Validation labels remain clean. This is a diagnostic proxy and should
    not be described as final independent-test backdoor ASR.
    """
    attack_mask = actual_labels != TARGET_LABEL
    attack_count = int(np.sum(attack_mask))
    successful_target_predictions = int(
        np.sum((predictions == TARGET_LABEL) & attack_mask)
    )

    if attack_count == 0:
        return {
            "Attack_Records": 0,
            "Attack_To_Benign": 0,
            "Validation_ASR": None,
        }

    return {
        "Attack_Records": attack_count,
        "Attack_To_Benign": successful_target_predictions,
        "Validation_ASR": successful_target_predictions / attack_count,
    }


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():
    print("\nCLIENT 07 MATCHED-POISONING PREDICTION ANALYSIS")
    print("=" * 78)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Non-IID data not found: {DATA_ROOT}")
    if not GLOBAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Trained Scenario H checkpoint not found: "
            f"{GLOBAL_CHECKPOINT}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine.DATA_ROOT = DATA_ROOT
    torch.set_num_threads(engine.CPU_THREADS)

    # --------------------------------------------------------
    # Load clean server-validation data
    # --------------------------------------------------------
    print("\nLoading clean server-validation data...")
    validation_X, validation_y, feature_columns = engine.load_validation_data()

    validation_X = np.array(validation_X, dtype=np.float32, copy=True)
    validation_y = np.array(validation_y, dtype=np.int64, copy=True)

    validation_loader = engine.create_validation_loader(
        validation_X,
        validation_y,
    )
    print(f"Validation records: {len(validation_y):,}")

    # --------------------------------------------------------
    # Load Client 07 training data
    # --------------------------------------------------------
    print("\nLoading Client 07 training data...")
    X, original_y, sample_count = engine.load_client_data(
        CLIENT_ID,
        feature_columns,
    )

    # Make explicit writable arrays with the expected dtypes.
    X = np.array(X, dtype=np.float32, copy=True)
    original_y = np.array(original_y, dtype=np.int64, copy=True)

    if len(X) != len(original_y):
        raise ValueError("Client feature and label counts do not match.")
    if len(original_y) != sample_count:
        raise ValueError("Reported Client 07 sample count does not match data.")

    print(f"Client training records: {sample_count:,}")

    # --------------------------------------------------------
    # Create and verify matched poisoned labels
    # --------------------------------------------------------
    poisoned_y, poisoned_count = create_poisoned_labels(original_y)

    print("\nPOISONING VERIFICATION")
    print("Malicious client:", CLIENT_ID)
    print("Target label:", f"{TARGET_LABEL} ({CLASS_NAMES[TARGET_LABEL]})")
    print("Poisoned records:", f"{poisoned_count:,}")
    print(
        "Overall client poisoning fraction:",
        f"{poisoned_count / sample_count:.6%}",
    )
    print(
        "Eligible non-Benign records:",
        f"{int(np.sum(original_y != TARGET_LABEL)):,}",
    )

    # --------------------------------------------------------
    # Load trained common global model from Scenario H, round 10
    # --------------------------------------------------------
    print("\nLoading trained clean non-IID global model...")
    checkpoint = torch.load(
        GLOBAL_CHECKPOINT,
        map_location=engine.DEVICE,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("The global checkpoint is not a dictionary.")
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "The checkpoint does not contain 'model_state_dict'."
        )

    global_model = IntrusionDetectionMLP().to(engine.DEVICE)
    global_model.load_state_dict(checkpoint["model_state_dict"])
    global_model.eval()

    checkpoint_round = checkpoint.get("round", "not recorded")
    print("Loaded checkpoint:", GLOBAL_CHECKPOINT)
    print("Checkpoint communication round:", checkpoint_round)
    print("Model initialized from trained non-IID global weights.")

    global_labels, global_predictions = get_predictions(
        global_model,
        validation_loader,
    )

    # --------------------------------------------------------
    # Train clean local candidate model
    # --------------------------------------------------------
    print("\nTraining CLEAN Client 07 local candidate model...")
    reset_training_seed()
    clean_model, clean_training_loss = engine.train_local_model(
        global_model,
        X,
        original_y.copy(),
    )

    # --------------------------------------------------------
    # Train poisoned local candidate model
    # --------------------------------------------------------
    print("\nTraining POISONED Client 07 local candidate model...")
    # Resetting to the same seed matches local training randomness.
    reset_training_seed()
    poisoned_model, poisoned_training_loss = engine.train_local_model(
        global_model,
        X,
        poisoned_y.copy(),
    )

    # --------------------------------------------------------
    # Get predictions for all three models
    # --------------------------------------------------------
    print("\nGenerating validation predictions...")
    clean_labels, clean_predictions = get_predictions(
        clean_model,
        validation_loader,
    )
    poisoned_labels, poisoned_predictions = get_predictions(
        poisoned_model,
        validation_loader,
    )

    if not np.array_equal(global_labels, clean_labels):
        raise RuntimeError("Validation labels differ between model evaluations.")
    if not np.array_equal(clean_labels, poisoned_labels):
        raise RuntimeError("Validation labels differ between model evaluations.")

    predictions_by_model = {
        "Trained_Global_Scenario_H_Round10": global_predictions,
        "Clean_Client07_Model": clean_predictions,
        "Poisoned_Client07_Model": poisoned_predictions,
    }

    # Print full prediction distribution so collapse to one class is visible.
    for model_name, predictions in predictions_by_model.items():
        print_prediction_distribution(model_name, predictions)

    # --------------------------------------------------------
    # Calculate model metrics and confusion matrices
    # --------------------------------------------------------
    summaries = []
    class_results = []
    prediction_count_results = []
    asr_results = []
    confusion_results = []

    for model_name, predictions in predictions_by_model.items():
        summary, per_class, prediction_counts = calculate_model_metrics(
            clean_labels,
            predictions,
            model_name,
        )
        summaries.append(summary)
        class_results.extend(per_class)
        prediction_count_results.extend(prediction_counts)

        asr = calculate_attack_success(clean_labels, predictions)
        asr["Model"] = model_name
        asr_results.append(asr)

        matrix = confusion_matrix(
            clean_labels,
            predictions,
            labels=LABEL_IDS,
        )
        for actual_class in LABEL_IDS:
            for predicted_class in LABEL_IDS:
                confusion_results.append(
                    {
                        "Model": model_name,
                        "Actual_Class_ID": actual_class,
                        "Actual_Class": CLASS_NAMES[actual_class],
                        "Predicted_Class_ID": predicted_class,
                        "Predicted_Class": CLASS_NAMES[predicted_class],
                        "Count": int(matrix[actual_class, predicted_class]),
                    }
                )

    # --------------------------------------------------------
    # Print overall metrics
    # --------------------------------------------------------
    print("\nOVERALL MODEL METRICS")
    print("=" * 95)
    summary_df = pd.DataFrame(summaries)
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # --------------------------------------------------------
    # Print per-class metrics
    # --------------------------------------------------------
    print("\nPER-CLASS PRECISION, RECALL AND F1")
    print("=" * 115)
    class_df = pd.DataFrame(class_results)

    for model_name in predictions_by_model:
        model_rows = class_df[class_df["Model"] == model_name]
        print(f"\n{model_name}")
        print("-" * 90)
        print(
            model_rows[
                ["Class_Name", "Precision", "Recall", "F1", "Support"]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )

    # --------------------------------------------------------
    # Print prediction counts by class
    # --------------------------------------------------------
    print("\nPREDICTED CLASS COUNTS")
    print("=" * 95)
    prediction_counts_df = pd.DataFrame(prediction_count_results)
    print(
        prediction_counts_df[
            ["Model", "Class_Name", "Predicted_Count", "Predicted_Fraction"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # --------------------------------------------------------
    # Print overall validation attack-to-Benign rate
    # --------------------------------------------------------
    print("\nVALIDATION ATTACK-TO-BENIGN RATE (DIAGNOSTIC PROXY)")
    print("=" * 95)
    asr_df = pd.DataFrame(asr_results)
    print(
        asr_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # --------------------------------------------------------
    # Per-attack-class attack-to-Benign rate
    # --------------------------------------------------------
    print("\nATTACK-TO-BENIGN RATE BY ORIGINAL VALIDATION CLASS")
    print("=" * 95)
    attack_class_results = []

    for model_name, predictions in predictions_by_model.items():
        for class_id in range(1, NUM_CLASSES):
            mask = clean_labels == class_id
            count = int(np.sum(mask))
            target_predictions = int(
                np.sum(predictions[mask] == TARGET_LABEL)
            )
            rate = target_predictions / count if count > 0 else None

            attack_class_results.append(
                {
                    "Model": model_name,
                    "Attack_Class": CLASS_NAMES[class_id],
                    "Original_Records": count,
                    "Predicted_As_Benign": target_predictions,
                    "Attack_To_Benign_Rate": rate,
                }
            )

    attack_class_df = pd.DataFrame(attack_class_results)
    print(
        attack_class_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    # --------------------------------------------------------
    # Save all results in the new diagnostic folder
    # --------------------------------------------------------
    summary_path = OUTPUT_DIR / "client07_prediction_model_summary.csv"
    class_path = OUTPUT_DIR / "client07_prediction_per_class_metrics.csv"
    prediction_counts_path = OUTPUT_DIR / "client07_prediction_class_counts.csv"
    asr_path = OUTPUT_DIR / "client07_validation_attack_to_benign_rate.csv"
    attack_class_path = OUTPUT_DIR / "client07_attack_to_benign_by_class.csv"
    confusion_path = OUTPUT_DIR / "client07_prediction_confusion_matrices.csv"

    summary_df.to_csv(summary_path, index=False)
    class_df.to_csv(class_path, index=False)
    prediction_counts_df.to_csv(prediction_counts_path, index=False)
    asr_df.to_csv(asr_path, index=False)
    attack_class_df.to_csv(attack_class_path, index=False)
    pd.DataFrame(confusion_results).to_csv(confusion_path, index=False)

    # Save diagnostic configuration and checkpoint provenance.
    metadata = {
        "scenario": "K - Client 07 matched poisoning prediction diagnostic",
        "data_root": str(DATA_ROOT),
        "global_checkpoint": str(GLOBAL_CHECKPOINT),
        "checkpoint_round": checkpoint_round,
        "client_id": CLIENT_ID,
        "target_label": TARGET_LABEL,
        "target_class": CLASS_NAMES[TARGET_LABEL],
        "poison_seed": POISON_SEED,
        "training_seed": TRAINING_SEED,
        "poisoned_records": poisoned_count,
        "client_records": int(sample_count),
        "client_poisoning_fraction": float(poisoned_count / sample_count),
        "validation_records": int(len(validation_y)),
        "evaluation_data": "server validation; not independent test data",
    }
    metadata_path = OUTPUT_DIR / "client07_prediction_diagnostic_metadata.json"
    pd.Series(metadata).to_json(metadata_path, indent=2)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    print("\n" + "=" * 78)
    print("CLIENT 07 MATCHED-POISONING PREDICTION ANALYSIS COMPLETED")
    print("=" * 78)
    print(f"Clean local training loss:    {clean_training_loss:.6f}")
    print(f"Poisoned local training loss: {poisoned_training_loss:.6f}")
    print("\nSaved files:")
    for path in (
        summary_path,
        class_path,
        prediction_counts_path,
        asr_path,
        attack_class_path,
        confusion_path,
        metadata_path,
    ):
        print(path)

    print("\nIMPORTANT INTERPRETATION NOTES:")
    print(
        "1. The starting model is the trained Scenario H round-10 global model, "
        "not a newly initialized random model."
    )
    print(
        "2. Validation attack-to-Benign rate is a diagnostic proxy, not final "
        "independent-test backdoor ASR."
    )
    print(
        "3. These are local candidate models; they are not fully aggregated "
        "federated models."
    )
    print("4. The independent test dataset was not used.")
    print(
        "5. Previous Scenario K diagnostic files were not overwritten; "
        "this run uses a separate output subfolder."
    )

    # Cleanup
    del X, original_y, poisoned_y
    del validation_X, validation_y, validation_loader
    del global_model, clean_model, poisoned_model
    gc.collect()
    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
