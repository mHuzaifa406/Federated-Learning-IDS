"""Run a non-IID CAVC signal-calibration experiment.

Two matched conditions are available:
  --condition clean
  --condition poison-client07

Both conditions use the same CIC-IDS-2017 non-IID partition, model,
initialization seed, local training configuration and validation records.
The poisoned condition repeats the previously tested persistent targeted
label flip for Client 07 (selected records are relabeled to Benign).

Screening is intentionally disabled. All ten clients are aggregated with
sample-size-weighted FedAvg so this script measures CAVC signal behaviour,
not the effectiveness of a defense. No independent test data is used.

Run from project root:
  python scripts/analyze_cavc_noniid.py --condition clean
  python scripts/analyze_cavc_noniid.py --condition poison-client07
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import scripts.federated_engine as engine
from models.ids_model import IntrusionDetectionMLP
from trfa.aggregation import aggregate_accepted_models
from trfa.trust_metrics_cavc_v1 import calculate_class_aware_validation_contribution


NONIID_DATA_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "federated_noniid_alpha05_seed43"
)

TOTAL_ROUNDS = 10
NUM_CLIENTS = 10
EXPERIMENT_SEED = 43
MALICIOUS_CLIENT_ID = 7
TARGET_LABEL = 0
POISON_CLIENT_FRACTION = 8676 / 213858
POISON_SEED = 2027
MIN_SUPPORT_THRESHOLDS = (50, 100, 200)

CLASS_NAMES = {
    0: "Benign",
    1: "Bot",
    2: "DoS",
    3: "Infiltration",
    4: "PortScan",
    5: "Web Attack",
    6: "Brute Force",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=("clean", "poison-client07"),
        required=True,
        help="Clean all-honest run or persistent Client 07 label-flipping run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=EXPERIMENT_SEED,
        help="Training/model seed. Use 43 to match the previous calibration runs.",
    )
    return parser.parse_args()


def set_experiment_seed(seed: int):
    engine.set_random_seed()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(engine.CPU_THREADS)


def create_poisoned_labels(original_y: np.ndarray, client_id: int):
    """Reproduce the previously used fixed client-level targeted label flip."""
    labels = np.array(original_y, dtype=np.int64, copy=True)
    eligible_indices = np.flatnonzero(labels != TARGET_LABEL)
    poison_count = int(np.floor(len(labels) * POISON_CLIENT_FRACTION))

    if poison_count <= 0 or poison_count > len(eligible_indices):
        raise ValueError("Configured poison count is invalid for this client.")

    rng = np.random.default_rng(POISON_SEED + client_id)
    selected = rng.choice(eligible_indices, size=poison_count, replace=False)
    labels[selected] = TARGET_LABEL

    actual_changes = int(np.count_nonzero(labels != original_y))
    if actual_changes != poison_count:
        raise RuntimeError("Poisoning verification failed.")

    return labels, actual_changes, int(len(eligible_indices))


def get_predictions(model, loader):
    model.eval()
    labels = []
    predictions = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            logits = model(batch_X.to(engine.DEVICE))
            predicted = torch.argmax(logits, dim=1)
            labels.extend(batch_y.detach().cpu().numpy().tolist())
            predictions.extend(predicted.detach().cpu().numpy().tolist())

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
    )


def model_summary(actual, predictions):
    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "macro_f1_all_present_labels": float(
            f1_score(actual, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(actual, predictions, average="weighted", zero_division=0)
        ),
        "balanced_accuracy_all_present_labels": float(
            balanced_accuracy_score(actual, predictions)
        ),
    }


def main():
    args = parse_args()
    condition = args.condition
    seed = int(args.seed)

    if not NONIID_DATA_ROOT.exists():
        raise FileNotFoundError(f"Non-IID data not found: {NONIID_DATA_ROOT}")

    metadata_path = NONIID_DATA_ROOT / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Partition metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        partition_metadata = json.load(f)

    if float(partition_metadata["dirichlet_alpha"]) != 0.5:
        raise ValueError("Expected Dirichlet alpha 0.5.")
    if int(partition_metadata["random_seed"]) != 43:
        raise ValueError("Expected non-IID partition seed 43.")

    suffix = f"seed{seed}"
    folder = (
        f"scenario_noniid_cavc_clean_{suffix}"
        if condition == "clean"
        else f"scenario_noniid_cavc_poison_client07_{suffix}"
    )
    output_dir = PROJECT_ROOT / "results" / folder

    # Avoid overwriting any existing experiment output.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files: {output_dir}\n"
            "Preserve the old run and use a new seed/output folder for a repeat."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    engine.DATA_ROOT = NONIID_DATA_ROOT
    set_experiment_seed(seed)

    print("\n" + "=" * 78)
    print("NON-IID CLASS-AWARE VALIDATION CONTRIBUTION (CAVC) CALIBRATION")
    print("=" * 78)
    print("Condition:", condition)
    print("Dataset: CIC-IDS-2017")
    print("Dirichlet alpha:", partition_metadata["dirichlet_alpha"])
    print("Partition seed:", partition_metadata["random_seed"])
    print("Training seed:", seed)
    print("Clients:", NUM_CLIENTS)
    print("Communication rounds:", TOTAL_ROUNDS)
    print("Minimum class support thresholds:", list(MIN_SUPPORT_THRESHOLDS))
    print("Screening: DISABLED (all ten clients are aggregated)")
    print("Target label:", TARGET_LABEL, "(Benign)")
    if condition == "poison-client07":
        print("Malicious client:", MALICIOUS_CLIENT_ID)
        print("Configured client poison fraction:", f"{POISON_CLIENT_FRACTION:.6%}")
    print("Independent test set: NOT USED")

    global_model = IntrusionDetectionMLP().to(engine.DEVICE)

    validation_X, validation_y, feature_columns = engine.load_validation_data()
    validation_loader = engine.create_validation_loader(validation_X, validation_y)
    validation_y = np.asarray(validation_y, dtype=np.int64)

    observed_labels, _ = get_predictions(global_model, validation_loader)
    if not np.array_equal(observed_labels, validation_y):
        raise RuntimeError("Validation loader order/labels do not match source labels.")

    client_ids = list(range(1, NUM_CLIENTS + 1))
    contribution_rows = []
    per_class_rows = []
    round_rows = []
    poisoning_summary = {}

    for round_number in range(1, TOTAL_ROUNDS + 1):
        print("\n" + "=" * 72)
        print(f"ROUND {round_number}/{TOTAL_ROUNDS}")
        print("=" * 72)

        base_labels, global_predictions = get_predictions(global_model, validation_loader)
        if not np.array_equal(base_labels, validation_y):
            raise RuntimeError("Global model validation labels changed order.")

        global_round_metrics = model_summary(base_labels, global_predictions)
        client_states = {}
        client_sizes = {}

        for client_id in client_ids:
            print(f"Evaluating candidate for Client {client_id:02d}...")
            X, original_y, sample_count = engine.load_client_data(client_id, feature_columns)
            X = np.array(X, dtype=np.float32, copy=True)
            original_y = np.array(original_y, dtype=np.int64, copy=True)

            is_malicious = condition == "poison-client07" and client_id == MALICIOUS_CLIENT_ID
            if is_malicious:
                y_for_training, poisoned_count, eligible_count = create_poisoned_labels(
                    original_y, client_id
                )
                poisoning_summary = {
                    "malicious_client_id": client_id,
                    "target_label": TARGET_LABEL,
                    "poisoned_records_per_round": poisoned_count,
                    "eligible_non_target_records": eligible_count,
                    "actual_fraction_of_client": poisoned_count / int(sample_count),
                    "configured_fraction_of_client": POISON_CLIENT_FRACTION,
                    "poison_seed": POISON_SEED,
                }
            else:
                y_for_training = original_y.copy()
                poisoned_count = 0
                eligible_count = int(np.count_nonzero(original_y != TARGET_LABEL))

            local_model, training_loss = engine.train_local_model(
                global_model, X, y_for_training
            )

            local_state = {
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            }
            client_states[client_id] = local_state
            client_sizes[client_id] = int(sample_count)

            candidate_labels, candidate_predictions = get_predictions(
                local_model, validation_loader
            )
            if not np.array_equal(base_labels, candidate_labels):
                raise RuntimeError(
                    f"Validation labels/order differ for Client {client_id}."
                )

            for min_support in MIN_SUPPORT_THRESHOLDS:
                result = calculate_class_aware_validation_contribution(
                    actual_labels=base_labels,
                    global_predictions=global_predictions,
                    candidate_predictions=candidate_predictions,
                    min_class_support=int(min_support),
                    target_label=TARGET_LABEL,
                    class_names=CLASS_NAMES,
                )

                contribution_rows.append({
                    "Condition": condition,
                    "Round": round_number,
                    "Client": client_id,
                    "Is_Malicious": bool(is_malicious),
                    "Min_Class_Support": int(min_support),
                    "Eligible_Classes": ",".join(map(str, result["eligible_class_ids"])),
                    "Excluded_Low_Support_Classes": ",".join(map(str, result["excluded_low_support_class_ids"])),
                    "CAVC_Raw_Macro_Recall_Delta": result["raw_macro_recall_delta"],
                    "CAVC_Score": result["cavc_score"],
                    "Global_Macro_Recall": result["global_macro_recall"],
                    "Candidate_Macro_Recall": result["candidate_macro_recall"],
                    "Global_Macro_F1": result["global_macro_f1"],
                    "Candidate_Macro_F1": result["candidate_macro_f1"],
                    "Macro_F1_Delta": result["macro_f1_delta"],
                    "Macro_Precision_Delta": result["macro_precision_delta"],
                    "Mean_Attack_To_Benign_Rate_Delta": result["mean_attack_to_target_rate_delta"],
                    "Training_Loss": float(training_loss),
                    "Client_Samples": int(sample_count),
                    "Poisoned_Records": int(poisoned_count),
                    "Eligible_NonBenign_Records": int(eligible_count),
                })

                for row in result["per_class"]:
                    per_class_rows.append({
                        "Condition": condition,
                        "Round": round_number,
                        "Client": client_id,
                        "Is_Malicious": bool(is_malicious),
                        "Min_Class_Support": int(min_support),
                        **row,
                    })

            print(
                f"  CAVC@100={next(x['CAVC_Score'] for x in reversed(contribution_rows) if x['Client']==client_id and x['Round']==round_number and x['Min_Class_Support']==100):.4f}; "
                f"training_loss={training_loss:.6f}; poisoned={poisoned_count:,}"
            )

            del X, original_y, y_for_training, local_model
            gc.collect()

        # Signal-calibration only: aggregate all clients exactly by sample count.
        aggregated_state, aggregation_weights = aggregate_accepted_models(
            client_states=client_states,
            sample_sizes=client_sizes,
            accepted_client_ids=client_ids,
            minimum_accepted_clients=2,
        )
        global_model.load_state_dict(aggregated_state)
        post_labels, post_predictions = get_predictions(global_model, validation_loader)
        if not np.array_equal(post_labels, validation_y):
            raise RuntimeError("Post-aggregation validation labels/order changed.")
        post_metrics = model_summary(post_labels, post_predictions)

        round_rows.append({
            "Condition": condition,
            "Round": round_number,
            "Pre_Aggregation_Validation_Accuracy": global_round_metrics["accuracy"],
            "Post_Aggregation_Validation_Accuracy": post_metrics["accuracy"],
            "Post_Aggregation_Macro_F1_All_Present_Labels": post_metrics["macro_f1_all_present_labels"],
            "Post_Aggregation_Weighted_F1": post_metrics["weighted_f1"],
            "Post_Aggregation_Balanced_Accuracy_All_Present_Labels": post_metrics["balanced_accuracy_all_present_labels"],
            "Accepted_Clients": NUM_CLIENTS,
            "Screening_Enabled": False,
        })

        checkpoint_path = output_dir / f"cavc_{condition}_round_{round_number:02d}.pt"
        torch.save({
            "round": round_number,
            "condition": condition,
            "seed": seed,
            "model_state_dict": global_model.state_dict(),
            "post_aggregation_validation_metrics": post_metrics,
            "screening_enabled": False,
        }, checkpoint_path)

        del client_states
        gc.collect()

    contribution_df = pd.DataFrame(contribution_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    round_df = pd.DataFrame(round_rows)

    contribution_path = output_dir / "cavc_client_round_scores.csv"
    per_class_path = output_dir / "cavc_per_class_diagnostics.csv"
    rounds_path = output_dir / "cavc_round_model_metrics.csv"
    contribution_df.to_csv(contribution_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    round_df.to_csv(rounds_path, index=False)

    default_df = contribution_df[contribution_df["Min_Class_Support"] == 100]
    summary_rows = []
    for client_id, group in default_df.groupby("Client"):
        summary_rows.append({
            "Client": int(client_id),
            "Is_Malicious": bool(group["Is_Malicious"].iloc[0]),
            "CAVC_Mean": float(group["CAVC_Score"].mean()),
            "CAVC_Std": float(group["CAVC_Score"].std(ddof=0)),
            "CAVC_Min": float(group["CAVC_Score"].min()),
            "CAVC_Max": float(group["CAVC_Score"].max()),
            "Raw_Macro_Recall_Delta_Mean": float(group["CAVC_Raw_Macro_Recall_Delta"].mean()),
            "Attack_To_Benign_Delta_Mean": float(group["Mean_Attack_To_Benign_Rate_Delta"].mean()),
            "Mean_Rank_Lowest_First": float(group["CAVC_Score"].rank(method="average").mean()),
        })
    client_summary_df = pd.DataFrame(summary_rows).sort_values("Client")
    client_summary_path = output_dir / "cavc_client_summary_min_support_100.csv"
    client_summary_df.to_csv(client_summary_path, index=False)

    summary = {
        "experiment": "CAVC non-IID signal calibration",
        "cavc_version": "1.0-prototype",
        "condition": condition,
        "dataset": "CIC-IDS-2017",
        "partition_method": "class-wise Dirichlet",
        "dirichlet_alpha": partition_metadata["dirichlet_alpha"],
        "partition_seed": partition_metadata["random_seed"],
        "training_seed": seed,
        "clients": NUM_CLIENTS,
        "rounds": TOTAL_ROUNDS,
        "screening_enabled": False,
        "aggregation": "sample-size-weighted FedAvg over all 10 clients",
        "minimum_class_support_thresholds": list(MIN_SUPPORT_THRESHOLDS),
        "validation_class_support": {
            str(k): int(v) for k, v in zip(*np.unique(validation_y, return_counts=True))
        },
        "default_minimum_class_support": 100,
        "target_label": TARGET_LABEL,
        "poisoning": poisoning_summary if condition == "poison-client07" else None,
        "final_post_aggregation_validation_metrics": round_rows[-1],
        "independent_test_set_used": False,
        "interpretation_note": (
            "CAVC is a server-validation contribution signal, not a calibrated probability of honesty. "
            "The run has screening disabled and therefore does not evaluate defense detection rates."
        ),
        "files": {
            "client_round_scores": contribution_path.name,
            "per_class_diagnostics": per_class_path.name,
            "round_model_metrics": rounds_path.name,
            "client_summary_min_support_100": client_summary_path.name,
        },
    }

    summary_path = output_dir / "cavc_experiment_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("CAVC NON-IID SIGNAL CALIBRATION COMPLETED")
    print("=" * 78)
    print("Condition:", condition)
    print("Output directory:", output_dir)
    print("Client summary (min support 100):")
    print(client_summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved:")
    print("  Client-round scores:", contribution_path)
    print("  Per-class diagnostics:", per_class_path)
    print("  Round model metrics:", rounds_path)
    print("  Client summary:", client_summary_path)
    print("  Experiment summary:", summary_path)
    print("Independent test set used: NO")
    print("Screening enabled: NO")

    del validation_X, validation_y, validation_loader, global_model
    gc.collect()


if __name__ == "__main__":
    main()
