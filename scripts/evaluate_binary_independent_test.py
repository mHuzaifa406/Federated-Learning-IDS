"""Evaluate a binary federated IDS checkpoint on the reserved CIC-IDS-2017 test split.

Binary label mapping:
    original Label_id == 0      -> Benign (0)
    original Label_id in 1..6   -> Attack (1)

This script performs inference only. It does not train, calibrate, select a
threshold, or modify the checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ids_model import IntrusionDetectionMLP


LABEL_COLUMNS = {"Label_id", "Label", "Label_7class"}
BINARY_NAMES = {0: "Benign", 1: "Attack"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_checkpoint(
    checkpoint_path: Path,
    test_dir: Path,
    normalization_metadata_path: Path,
    expected_records: int,
    expected_file_count: int,
    batch_size: int,
) -> dict[str, Any]:
    normalization = load_json(normalization_metadata_path)
    feature_names = normalization.get("features")
    if not isinstance(feature_names, list) or len(feature_names) != 78:
        raise ValueError(
            "Normalization metadata must define the ordered list of 78 features."
        )

    class_mapping = normalization.get("classes")
    if not isinstance(class_mapping, dict) or len(class_mapping) != 7:
        raise ValueError(
            "Normalization metadata must define all seven original CIC-IDS-2017 classes."
        )
    original_class_names = {
        int(label): str(name) for name, label in class_mapping.items()
    }
    if set(original_class_names) != set(range(7)):
        raise ValueError("Original class mapping must contain Label_id values 0 through 6.")
    if original_class_names[0].strip().lower() != "benign":
        raise ValueError("Original Label_id 0 must map to Benign.")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")

    checkpoint_mapping = checkpoint.get("label_mapping")
    if checkpoint_mapping is not None:
        normalized_mapping = {
            str(name).strip().lower(): int(label)
            for name, label in checkpoint_mapping.items()
        }
        if normalized_mapping != {"benign": 0, "attack": 1}:
            raise ValueError(
                f"Checkpoint binary label mapping is unexpected: {checkpoint_mapping}"
            )

    model = IntrusionDetectionMLP(input_features=78, num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    torch.set_num_threads(4)

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Independent test directory not found: {test_dir}")
    parquet_files = sorted(test_dir.glob("*.parquet"))
    if len(parquet_files) != int(expected_file_count):
        raise ValueError(
            f"Expected {expected_file_count} test Parquet files, found {len(parquet_files)}."
        )

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    total_records = 0
    original_class_support = np.zeros(7, dtype=np.int64)
    file_records: dict[str, int] = {}

    for parquet_path in parquet_files:
        frame = pd.read_parquet(parquet_path)
        if "Label_id" not in frame.columns:
            raise ValueError(f"{parquet_path.name} has no Label_id column.")

        observed_features = [
            column for column in frame.columns if column not in LABEL_COLUMNS
        ]
        if observed_features != feature_names:
            raise ValueError(
                f"Feature names/order in {parquet_path.name} do not match "
                "normalization_metadata.json. Evaluation stopped."
            )

        x = frame[feature_names].to_numpy(dtype=np.float32, copy=True)
        original_y = frame["Label_id"].to_numpy(dtype=np.int64, copy=True)
        if len(original_y) == 0:
            raise ValueError(f"{parquet_path.name} contains no records.")
        if not np.isfinite(x).all():
            raise ValueError(f"{parquet_path.name} contains non-finite feature values.")
        if np.any((original_y < 0) | (original_y > 6)):
            raise ValueError(f"{parquet_path.name} contains labels outside 0..6.")

        binary_y = (original_y != 0).astype(np.int64)
        prediction_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(original_y), int(batch_size)):
                batch_x = torch.from_numpy(x[start:start + int(batch_size)])
                logits = model(batch_x)
                if logits.ndim != 2 or logits.shape[1] != 2:
                    raise ValueError(
                        f"Binary checkpoint produced unexpected output shape: {tuple(logits.shape)}"
                    )
                prediction_parts.append(
                    torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
                )

        predictions = np.concatenate(prediction_parts)
        y_true_parts.append(binary_y)
        y_pred_parts.append(predictions)
        original_class_support += np.bincount(original_y, minlength=7)
        total_records += int(len(original_y))
        file_records[parquet_path.name] = int(len(original_y))

        del frame, x, original_y, binary_y, predictions, prediction_parts

    if total_records != int(expected_records):
        raise ValueError(
            f"Expected {expected_records:,} test records, evaluated {total_records:,}."
        )

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    labels = np.array([0, 1], dtype=np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())

    expected_original_counts_raw = normalization.get("testing_class_counts", {})
    expected_original_counts = {
        int(class_mapping[name]): int(count)
        for name, count in expected_original_counts_raw.items()
        if name in class_mapping
    }
    observed_original_counts = {
        int(class_id): int(original_class_support[class_id])
        for class_id in range(7)
    }
    if expected_original_counts and observed_original_counts != expected_original_counts:
        raise ValueError(
            "Observed original test class counts do not match normalization metadata. "
            f"Observed={observed_original_counts}; expected={expected_original_counts}."
        )

    per_original_class: dict[str, Any] = {}
    for class_id in range(1, 7):
        class_mask = (np.concatenate([
            np.asarray([], dtype=np.int64)
        ]) if False else None)
        # Per-category attack misses are calculated from the same concatenated
        # original labels retained below by a second lightweight read.
        missed = 0
        correct = 0
        total = int(original_class_support[class_id])
        # The per-category breakdown is filled in the second pass below.
        per_original_class[str(class_id)] = {
            "class_name": original_class_names[class_id],
            "support": total,
            "detected_as_attack": None,
            "missed_as_benign": None,
            "attack_recall": None,
        }

    # Re-read the eight test files to produce per-original-attack-category
    # misses without retaining all original labels in memory during inference.
    for parquet_path in parquet_files:
        frame = pd.read_parquet(parquet_path)
        original_y = frame["Label_id"].to_numpy(dtype=np.int64, copy=True)
        x = frame[feature_names].to_numpy(dtype=np.float32, copy=True)
        predictions_for_file: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(original_y), int(batch_size)):
                batch_x = torch.from_numpy(x[start:start + int(batch_size)])
                predictions_for_file.append(
                    torch.argmax(model(batch_x), dim=1).cpu().numpy().astype(np.int64)
                )
        file_predictions = np.concatenate(predictions_for_file)
        for class_id in range(1, 7):
            mask = original_y == class_id
            if np.any(mask):
                total = int(mask.sum())
                detected = int(np.count_nonzero(file_predictions[mask] == 1))
                missed = total - detected
                per_original_class[str(class_id)]["detected_as_attack"] = detected
                per_original_class[str(class_id)]["missed_as_benign"] = missed
                per_original_class[str(class_id)]["attack_recall"] = (
                    float(detected / total) if total else None
                )
        del frame, original_y, x, file_predictions, predictions_for_file

    binary_counts = {
        "Benign": int(original_class_support[0]),
        "Attack": int(original_class_support[1:].sum()),
    }
    if binary_counts["Benign"] != int(support[0]) or binary_counts["Attack"] != int(support[1]):
        raise RuntimeError("Binary label totals do not match the observed test class counts.")

    return {
        "evaluation": "binary_independent_test_checkpoint_evaluation",
        "binary_label_mapping": {
            "original_Label_id_0": "Benign (0)",
            "original_Label_id_1_to_6": "Attack (1)",
        },
        "test_set_used_for_training_or_calibration": False,
        "test_split_history_note": (
            "This split was not used by the binary training or calibration runs. "
            "It was examined in earlier seven-class project analyses, so it should "
            "be described as a held-out binary evaluation split, not as historically unseen."
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_round": checkpoint.get("round"),
        "training_seed": checkpoint.get("training_seed"),
        "device": "cpu",
        "test_directory": str(test_dir),
        "test_files": file_records,
        "test_records": int(total_records),
        "number_of_features": 78,
        "number_of_binary_classes": 2,
        "binary_class_counts": binary_counts,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_binary_class_metrics": {
            BINARY_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(2)
        },
        "benign_false_alarm_count": fp,
        "benign_false_alarm_rate": float(fp / support[0]) if support[0] else None,
        "attack_to_benign_missed_count": fn,
        "attack_to_benign_missed_rate": float(fn / support[1]) if support[1] else None,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "confusion_matrix_rows_true_columns_predicted": matrix.astype(int).tolist(),
        "per_original_attack_category": per_original_class,
        "interpretation_note": (
            "This is an inference-only evaluation of the supplied binary checkpoint. "
            "Do not use these results to tune the model or defense threshold."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--test-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "normalized_v1" / "test"),
    )
    parser.add_argument(
        "--normalization-metadata",
        default=str(
            PROJECT_ROOT / "data" / "processed" / "normalized_v1"
            / "normalization_metadata.json"
        ),
    )
    parser.add_argument(
        "--data-config",
        default=str(PROJECT_ROOT / "configs" / "data_config.json"),
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer.")

    config = load_json(Path(args.data_config).resolve())
    expected_records = int(config["data"]["expected_records"]["test_total"])
    expected_files = int(config["data"]["test_file_count"])

    output_dir = Path(args.output_dir).resolve()
    output_path = output_dir / "binary_independent_test_metrics.json"
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. Choose a new --output-dir."
        )

    summary = evaluate_checkpoint(
        Path(args.checkpoint).resolve(),
        Path(args.test_dir).resolve(),
        Path(args.normalization_metadata).resolve(),
        expected_records,
        expected_files,
        args.batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved binary independent-test metrics: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
