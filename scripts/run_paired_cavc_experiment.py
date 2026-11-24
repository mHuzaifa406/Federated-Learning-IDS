"""Paired frozen-checkpoint CAVC V2 experiment for non-IID Client 07.

Run from the TRFA project root:
    python scripts/run_paired_cavc_experiment.py

The script loads the Scenario H round-10 clean global checkpoint. It trains
one clean local candidate for each of the 10 non-IID clients from that SAME
frozen global model. It then retrains Client 07 with the exact matched
label-flipping configuration, resetting the same random seed used for its
clean local candidate. All predictions are evaluated on the same unchanged
server-validation records. No federated aggregation or screening occurs.

This is a local candidate-update diagnostic, not a full FL defense result.
It does not use the independent test set.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

import scripts.federated_engine as engine
from models.ids_model import IntrusionDetectionMLP
from trfa.trust_metrics_cavc_v2 import calculate_cavc_v2, rank_clients_within_condition


NONIID_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "federated_noniid_alpha05_seed43"
GLOBAL_CHECKPOINT = (
    PROJECT_ROOT / "results" / "scenario_H_noniid_trust_calibration" / "clean_noniid_round_10.pt"
)

NUM_CLIENTS = 10
MALICIOUS_CLIENT_ID = 7
TARGET_LABEL = 0
CLASS_NAMES = {
    0: "Benign",
    1: "Bot",
    2: "DoS",
    3: "Infiltration",
    4: "PortScan",
    5: "Web Attack",
    6: "Brute Force",
}
MIN_SUPPORT_THRESHOLDS = (50, 100, 200)
DEFAULT_SEED = 43
DEFAULT_POISON_SEED = 2027
# Retain the exact client-level fraction used in the previous matched Scenario K.
POISON_CLIENT_FRACTION = 8676 / 213858


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(engine.CPU_THREADS)


def create_poisoned_labels(original_y: np.ndarray, client_id: int, poison_seed: int):
    labels = np.array(original_y, dtype=np.int64, copy=True)
    eligible = np.flatnonzero(labels != TARGET_LABEL)
    poison_count = int(np.floor(len(labels) * POISON_CLIENT_FRACTION))
    if poison_count <= 0 or poison_count > len(eligible):
        raise ValueError(
            f"Invalid poison count {poison_count} for {len(eligible)} eligible records."
        )
    rng = np.random.default_rng(poison_seed + client_id)
    selected = rng.choice(eligible, size=poison_count, replace=False)
    labels[selected] = TARGET_LABEL
    changed = int(np.count_nonzero(labels != original_y))
    if changed != poison_count:
        raise RuntimeError("Poison count verification failed.")
    return labels, changed, int(len(eligible))


def get_predictions(model, loader):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.inference_mode():
        for batch_X, batch_y in loader:
            logits = model(batch_X.to(engine.DEVICE))
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(batch_y.detach().cpu().numpy().astype(np.int64).tolist())
            all_predictions.extend(preds.detach().cpu().numpy().astype(np.int64).tolist())
    return np.asarray(all_labels, dtype=np.int64), np.asarray(all_predictions, dtype=np.int64)


def load_global_model():
    if not GLOBAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Required trained clean global checkpoint was not found: "
            f"{GLOBAL_CHECKPOINT}\nRun/restore the original Scenario H experiment first."
        )
    try:
        checkpoint = torch.load(GLOBAL_CHECKPOINT, map_location=engine.DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(GLOBAL_CHECKPOINT, map_location=engine.DEVICE)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")
    model = IntrusionDetectionMLP().to(engine.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Paired CAVC V2 frozen-checkpoint experiment.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--poison-seed", type=int, default=DEFAULT_POISON_SEED)
    parser.add_argument("--output-tag", type=str, default="seed43", help="Unique output-folder suffix.")
    args = parser.parse_args()

    if not args.output_tag.strip() or any(ch in args.output_tag for ch in "\\/:*?\"<>| "):
        raise ValueError("--output-tag must be a non-empty folder-safe token without spaces or path separators.")

    output_dir = PROJECT_ROOT / "results" / f"scenario_cavc_v2_paired_client07_{args.output_tag}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}\n"
            "Choose a new --output-tag. Existing results will not be overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not NONIID_DATA_ROOT.exists():
        raise FileNotFoundError(f"Non-IID partition not found: {NONIID_DATA_ROOT}")
    metadata_path = NONIID_DATA_ROOT / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Non-IID partition metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        partition_metadata = json.load(handle)
    if float(partition_metadata.get("dirichlet_alpha", -1)) != 0.5:
        raise ValueError("This paired diagnostic expects the alpha=0.5 non-IID partition.")
    if int(partition_metadata.get("random_seed", -1)) != 43:
        raise ValueError("This paired diagnostic expects partition seed 43.")

    engine.DATA_ROOT = NONIID_DATA_ROOT
    set_seed(args.seed)

    print("\n" + "=" * 78)
    print("PAIRED CAVC V2 FROZEN-CHECKPOINT EXPERIMENT")
    print("=" * 78)
    print("Non-IID partition:", NONIID_DATA_ROOT)
    print("Checkpoint:", GLOBAL_CHECKPOINT)
    print("Training seed:", args.seed)
    print("Poison seed:", args.poison_seed)
    print("Malicious client:", MALICIOUS_CLIENT_ID)
    print("Target label:", TARGET_LABEL, "(Benign)")
    print("Configured client poison fraction:", f"{POISON_CLIENT_FRACTION:.6%}")
    print("Minimum class supports:", list(MIN_SUPPORT_THRESHOLDS))
    print("Screening: DISABLED; aggregation: NOT PERFORMED")

    global_model, checkpoint = load_global_model()
    print("Loaded checkpoint round:", checkpoint.get("round", "not recorded"))

    validation_X, validation_y_from_engine, feature_columns = engine.load_validation_data()
    validation_X = np.array(validation_X, dtype=np.float32, copy=True)
    validation_y_from_engine = np.array(validation_y_from_engine, dtype=np.int64, copy=True)
    validation_loader = engine.create_validation_loader(validation_X, validation_y_from_engine)

    base_labels, global_predictions = get_predictions(global_model, validation_loader)
    if not np.array_equal(base_labels, validation_y_from_engine):
        raise RuntimeError("Validation labels/order from loader differ from engine validation labels.")

    # Obtain a clean local candidate for every client from the same frozen
    # global checkpoint. Per-client seed is fixed; Client 07 is repeated below
    # with this exact seed so the clean/poison pair shares initialization and
    # data-loader random order.
    clean_predictions_by_client = {}
    clean_candidate_losses = {}
    client_sample_counts = {}
    client_data_cache_for_07 = None

    for client_id in range(1, NUM_CLIENTS + 1):
        print(f"\nTraining clean local candidate for Client {client_id:02d}...")
        X, y, sample_count = engine.load_client_data(client_id, feature_columns)
        X = np.array(X, dtype=np.float32, copy=True)
        y = np.array(y, dtype=np.int64, copy=True)
        if len(X) != len(y) or len(y) != int(sample_count):
            raise RuntimeError(f"Feature/label/sample count mismatch for Client {client_id}.")
        client_sample_counts[client_id] = int(sample_count)
        if client_id == MALICIOUS_CLIENT_ID:
            client_data_cache_for_07 = (X.copy(), y.copy(), int(sample_count))

        client_seed = int(args.seed + client_id)
        set_seed(client_seed)
        local_model, train_loss = engine.train_local_model(global_model, X, y.copy())
        labels, predictions = get_predictions(local_model, validation_loader)
        if not np.array_equal(labels, base_labels):
            raise RuntimeError(f"Validation order changed for Client {client_id}.")
        clean_predictions_by_client[client_id] = predictions
        clean_candidate_losses[client_id] = float(train_loss)
        print(f"  records={sample_count:,}; train_loss={train_loss:.6f}")
        del local_model, X, y
        gc.collect()

    if client_data_cache_for_07 is None:
        raise RuntimeError("Client 07 data was not loaded.")

    # Retrain Client 07 with the exact same per-client seed as its clean
    # candidate. Only labels differ; features, starting global weights, seed,
    # and validation records are held fixed.
    X07, clean_y07, count07 = client_data_cache_for_07
    poisoned_y07, poisoned_count, eligible_count = create_poisoned_labels(
        clean_y07, MALICIOUS_CLIENT_ID, args.poison_seed
    )
    print("\nTraining paired POISONED Client 07 candidate...")
    set_seed(int(args.seed + MALICIOUS_CLIENT_ID))
    poisoned_model, poisoned_train_loss = engine.train_local_model(
        global_model, X07, poisoned_y07
    )
    poison_labels, poisoned_predictions = get_predictions(poisoned_model, validation_loader)
    if not np.array_equal(poison_labels, base_labels):
        raise RuntimeError("Validation labels/order changed for poisoned Client 07.")

    # Confirm the paired local models are not accidentally identical.
    paired_clean_preds = clean_predictions_by_client[MALICIOUS_CLIENT_ID]
    changed_prediction_count = int(np.count_nonzero(paired_clean_preds != poisoned_predictions))

    # Build two conditions with exactly the same frozen global model and the
    # same nine honest candidate predictions. Only Client 07 differs.
    condition_predictions = {
        "clean": dict(clean_predictions_by_client),
        "poison_client07": dict(clean_predictions_by_client),
    }
    condition_predictions["poison_client07"][MALICIOUS_CLIENT_ID] = poisoned_predictions

    component_rows = []
    per_class_rows = []
    prediction_count_rows = []
    comparison_rows = []

    for condition, predictions_by_client in condition_predictions.items():
        rows_for_ranking = []
        for client_id in range(1, NUM_CLIENTS + 1):
            is_malicious = condition == "poison_client07" and client_id == MALICIOUS_CLIENT_ID
            for min_support in MIN_SUPPORT_THRESHOLDS:
                result = calculate_cavc_v2(
                    actual_labels=base_labels,
                    global_predictions=global_predictions,
                    candidate_predictions=predictions_by_client[client_id],
                    min_class_support=min_support,
                    target_label=TARGET_LABEL,
                    class_names=CLASS_NAMES,
                )
                row = {
                    "condition": condition,
                    "client_id": client_id,
                    "is_malicious": bool(is_malicious),
                    "min_class_support": min_support,
                    "eligible_attack_class_ids": ",".join(map(str, result["eligible_attack_class_ids"])),
                    "excluded_low_support_attack_class_ids": ",".join(map(str, result["excluded_low_support_attack_class_ids"])),
                    "attack_recall_contribution": result["attack_recall_contribution"],
                    "attack_detection_contribution_score": result["attack_detection_contribution_score"],
                    "attack_macro_f1_delta": result["attack_macro_f1_delta"],
                    "attack_to_target_rate_delta_mean": result["attack_to_target_rate_delta_mean"],
                    "targeted_damage_penalty": result["targeted_damage_penalty"],
                    "global_benign_false_positive_rate": result["global_benign_false_positive_rate"],
                    "candidate_benign_false_positive_rate": result["candidate_benign_false_positive_rate"],
                    "benign_false_positive_rate_delta": result["benign_false_positive_rate_delta"],
                    "benign_fpr_damage_penalty": result["benign_fpr_damage_penalty"],
                    "global_benign_recall": result["global_benign_recall"],
                    "candidate_benign_recall": result["candidate_benign_recall"],
                    "benign_recall_delta": result["benign_recall_delta"],
                    "client_samples": client_sample_counts[client_id],
                    "candidate_training_loss": (
                        float(poisoned_train_loss) if is_malicious else clean_candidate_losses[client_id]
                    ),
                    "poisoned_records": poisoned_count if is_malicious else 0,
                    "client_poison_fraction": (poisoned_count / count07) if is_malicious else 0.0,
                }
                component_rows.append(row)
                if min_support == 100:
                    rows_for_ranking.append({
                        "client_id": client_id,
                        "attack_detection_contribution_score": result["attack_detection_contribution_score"],
                    })
                for class_row in result["per_class"]:
                    per_class_rows.append({
                        "condition": condition,
                        "client_id": client_id,
                        "is_malicious": bool(is_malicious),
                        "min_class_support": min_support,
                        **class_row,
                    })

            # Prediction distributions provide context for any recall changes.
            pred_counts = np.bincount(predictions_by_client[client_id], minlength=7)
            for class_id, predicted_count in enumerate(pred_counts[:7]):
                prediction_count_rows.append({
                    "condition": condition,
                    "client_id": client_id,
                    "is_malicious": bool(is_malicious),
                    "predicted_class_id": class_id,
                    "predicted_class_name": CLASS_NAMES[class_id],
                    "predicted_count": int(predicted_count),
                    "predicted_fraction": float(predicted_count / len(base_labels)),
                })

        ranked = rank_clients_within_condition(rows_for_ranking)
        rank_lookup = {int(x["client_id"]): float(x["rank_lowest_first"]) for x in ranked}
        for row in component_rows:
            if row["condition"] == condition and row["min_class_support"] == 100:
                row["rank_lowest_first"] = rank_lookup[int(row["client_id"])]

    # Paired Client 07 summary at each support level.
    components_df = pd.DataFrame(component_rows)
    for support in MIN_SUPPORT_THRESHOLDS:
        clean_row = components_df[
            (components_df.condition == "clean") &
            (components_df.client_id == MALICIOUS_CLIENT_ID) &
            (components_df.min_class_support == support)
        ].iloc[0]
        poison_row = components_df[
            (components_df.condition == "poison_client07") &
            (components_df.client_id == MALICIOUS_CLIENT_ID) &
            (components_df.min_class_support == support)
        ].iloc[0]
        comparison_rows.append({
            "min_class_support": support,
            "client07_clean_attack_detection_score": float(clean_row.attack_detection_contribution_score),
            "client07_poisoned_attack_detection_score": float(poison_row.attack_detection_contribution_score),
            "client07_attack_detection_score_change": float(poison_row.attack_detection_contribution_score - clean_row.attack_detection_contribution_score),
            "client07_clean_targeted_damage_penalty": float(clean_row.targeted_damage_penalty),
            "client07_poisoned_targeted_damage_penalty": float(poison_row.targeted_damage_penalty),
            "client07_targeted_damage_penalty_change": float(poison_row.targeted_damage_penalty - clean_row.targeted_damage_penalty),
            "client07_clean_benign_fpr_damage_penalty": float(clean_row.benign_fpr_damage_penalty),
            "client07_poisoned_benign_fpr_damage_penalty": float(poison_row.benign_fpr_damage_penalty),
            "client07_benign_fpr_damage_penalty_change": float(poison_row.benign_fpr_damage_penalty - clean_row.benign_fpr_damage_penalty),
        })

    per_client_summary_rows = []
    main_support = 100
    for condition in ("clean", "poison_client07"):
        subset = components_df[(components_df.condition == condition) & (components_df.min_class_support == main_support)]
        for _, row in subset.iterrows():
            other_mean = float(subset[subset.client_id != MALICIOUS_CLIENT_ID]["attack_detection_contribution_score"].mean())
            per_client_summary_rows.append({
                "condition": condition,
                "client_id": int(row.client_id),
                "is_malicious": bool(row.is_malicious),
                "attack_detection_contribution_score": float(row.attack_detection_contribution_score),
                "targeted_damage_penalty": float(row.targeted_damage_penalty),
                "benign_fpr_damage_penalty": float(row.benign_fpr_damage_penalty),
                "rank_lowest_first": int(row.rank_lowest_first),
                "other_clients_mean_attack_detection_score": other_mean,
                "score_gap_vs_other_clients": float(row.attack_detection_contribution_score - other_mean),
            })

    components_df.to_csv(output_dir / "cavc_v2_component_scores.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(output_dir / "cavc_v2_per_class_diagnostics.csv", index=False)
    pd.DataFrame(prediction_count_rows).to_csv(output_dir / "cavc_v2_prediction_counts.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output_dir / "client07_paired_clean_vs_poisoned.csv", index=False)
    summary_df = pd.DataFrame(per_client_summary_rows)
    summary_df.to_csv(output_dir / "cavc_v2_client_summary_min_support_100.csv", index=False)

    metadata = {
        "experiment": "Paired CAVC V2 frozen-checkpoint local-candidate diagnostic",
        "cavc_version": "2.0",
        "dataset": "CIC-IDS-2017",
        "partition": str(NONIID_DATA_ROOT),
        "dirichlet_alpha": partition_metadata["dirichlet_alpha"],
        "partition_seed": partition_metadata["random_seed"],
        "global_checkpoint": str(GLOBAL_CHECKPOINT),
        "global_checkpoint_round": checkpoint.get("round"),
        "training_seed": args.seed,
        "poison_seed": args.poison_seed,
        "clients": NUM_CLIENTS,
        "malicious_client_id": MALICIOUS_CLIENT_ID,
        "target_label": TARGET_LABEL,
        "configured_client_poison_fraction": POISON_CLIENT_FRACTION,
        "poisoned_records_client07": int(poisoned_count),
        "client07_records": int(count07),
        "actual_client07_poison_fraction": float(poisoned_count / count07),
        "client07_prediction_changes_vs_clean_candidate": changed_prediction_count,
        "validation_records": int(len(base_labels)),
        "validation_class_support": {str(k): int(v) for k, v in zip(*np.unique(base_labels, return_counts=True))},
        "minimum_support_thresholds": list(MIN_SUPPORT_THRESHOLDS),
        "screening_enabled": False,
        "aggregation_performed": False,
        "independent_test_set_used": False,
        "pairing": "Same frozen global checkpoint; same Client 07 features; same Client 07 local training seed; same poison-selected records across this paired candidate; same validation records/order. Only Client 07 labels differ between its clean and poisoned candidates.",
        "limitations": [
            "This is a local candidate-model diagnostic, not a complete federated defense evaluation.",
            "CAVC V2 components are separate diagnostics and are not fused into a client-honesty score.",
            "The server-validation set is used only for development diagnostics; the independent test set is not used.",
            "Classes below minimum support are reported but excluded from the aggregated attack metrics.",
            "Repeated class metrics from one validation set do not establish generalization across datasets or seeds."
        ],
    }
    with (output_dir / "cavc_v2_experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\n" + "=" * 78)
    print("PAIRED CAVC V2 EXPERIMENT COMPLETED")
    print("=" * 78)
    print("Validation records:", f"{len(base_labels):,}")
    print("Client 07 records:", f"{count07:,}")
    print("Client 07 poisoned records:", f"{poisoned_count:,}")
    print("Actual Client 07 poison fraction:", f"{poisoned_count / count07:.6%}")
    print("Client 07 changed validation predictions:", f"{changed_prediction_count:,}")
    print("\nClient summary (min support 100):")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nPaired Client 07 comparison:")
    print(pd.DataFrame(comparison_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved results:", output_dir)
    print("  cavc_v2_component_scores.csv")
    print("  cavc_v2_per_class_diagnostics.csv")
    print("  cavc_v2_prediction_counts.csv")
    print("  cavc_v2_client_summary_min_support_100.csv")
    print("  client07_paired_clean_vs_poisoned.csv")
    print("  cavc_v2_experiment_metadata.json")
    print("\nScreening enabled: NO")
    print("Federated aggregation performed: NO")
    print("Independent test set used: NO")
    print("This experiment does not claim that CAVC V2 detects or rejects malicious clients.")

    del global_model, poisoned_model, validation_X, validation_y_from_engine, validation_loader
    gc.collect()


if __name__ == "__main__":
    main()
