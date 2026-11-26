"""Evaluate a saved federated IDS checkpoint once on the untouched test split.

This script does not train, calibrate, select thresholds, or modify the checkpoint.
It verifies the expected test split size and feature order before inference.
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
        raise ValueError("Normalization metadata must define the ordered 78-feature list.")

    class_mapping = normalization.get("classes", {})
    if not isinstance(class_mapping, dict) or len(class_mapping) != 7:
        raise ValueError("Normalization metadata must define all seven class labels.")
    class_names = {int(label): str(name) for name, label in class_mapping.items()}
    labels = np.arange(7, dtype=int)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # These checkpoints are produced by this project; map_location keeps evaluation on CPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")

    model = IntrusionDetectionMLP(input_features=len(feature_names), num_classes=7)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

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
    per_class_support = np.zeros(7, dtype=np.int64)
    file_records: dict[str, int] = {}

    for parquet_path in parquet_files:
        frame = pd.read_parquet(parquet_path)
        if "Label_id" not in frame.columns:
            raise ValueError(f"{parquet_path} has no Label_id column.")

        observed_features = [
            col for col in frame.columns
            if col not in {"Label_id", "Label", "Label_7class"}
        ]
        if observed_features != feature_names:
            raise ValueError(
                f"Feature names/order in {parquet_path.name} do not match "
                "normalization_metadata.json. Refusing unsafe inference."
            )

        x = frame[feature_names].to_numpy(dtype=np.float32, copy=True)
        y = frame["Label_id"].to_numpy(dtype=np.int64, copy=True)
        if len(y) == 0:
            raise ValueError(f"{parquet_path} contains no records.")
        if not np.isfinite(x).all():
            raise ValueError(f"{parquet_path} contains non-finite feature values.")
        if np.any((y < 0) | (y > 6)):
            raise ValueError(f"{parquet_path} contains labels outside 0..6.")

        predictions: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(y), int(batch_size)):
                batch = torch.from_numpy(x[start:start + int(batch_size)])
                logits = model(batch)
                predictions.append(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64))

        pred = np.concatenate(predictions)
        y_true_parts.append(y)
        y_pred_parts.append(pred)
        per_class_support += np.bincount(y, minlength=7)
        total_records += len(y)
        file_records[parquet_path.name] = int(len(y))

        del frame, x, y, pred, predictions

    if total_records != int(expected_records):
        raise ValueError(
            f"Expected {expected_records:,} test records, evaluated {total_records:,}."
        )

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    benign_count = int(np.sum(y_true == 0))
    attack_mask = y_true != 0
    attack_count = int(np.sum(attack_mask))
    benign_false_alarm_count = int(np.sum((y_true == 0) & (y_pred != 0)))
    attack_to_benign_count = int(np.sum(attack_mask & (y_pred == 0)))

    per_class: dict[str, Any] = {}
    for idx, label in enumerate(labels):
        class_mask = y_true == label
        class_count = int(class_mask.sum())
        class_to_benign = int(np.sum(class_mask & (y_pred == 0))) if label != 0 else None
        per_class[str(int(label))] = {
            "class_name": class_names[int(label)],
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
            "predicted_as_benign_count": class_to_benign,
            "predicted_as_benign_rate": (
                float(class_to_benign / class_count) if class_count and label != 0 else None
            ),
        }

    expected_class_counts_raw = normalization.get("testing_class_counts", {})
    expected_class_counts = {
        int(class_mapping[name]): int(count)
        for name, count in expected_class_counts_raw.items()
        if name in class_mapping
    }
    observed_class_counts = {int(i): int(per_class[str(i)]["support"]) for i in labels}
    if expected_class_counts and observed_class_counts != expected_class_counts:
        raise ValueError(
            "Observed test class counts do not match normalization metadata; "
            f"observed={observed_class_counts}, expected={expected_class_counts}."
        )

    return {
        "evaluation": "independent_test_checkpoint_evaluation",
        "test_set_used_for_training_or_calibration": False,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_round": checkpoint.get("round"),
        "checkpoint_attack_active": checkpoint.get("attack_active"),
        "checkpoint_attack_start_round": checkpoint.get("attack_start_round"),
        "device": "cpu",
        "test_directory": str(test_dir),
        "test_files": file_records,
        "test_records": int(total_records),
        "number_of_features": int(len(feature_names)),
        "number_of_classes": 7,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "benign_false_alarm_count": benign_false_alarm_count,
        "benign_false_alarm_rate": (
            float(benign_false_alarm_count / benign_count) if benign_count else None
        ),
        "attack_to_benign_count": attack_to_benign_count,
        "attack_to_benign_rate": (
            float(attack_to_benign_count / attack_count) if attack_count else None
        ),
        "class_counts": observed_class_counts,
        "per_class_metrics": per_class,
        "confusion_matrix_rows_true_columns_predicted": matrix.astype(int).tolist(),
        "interpretation_note": (
            "This evaluates the saved checkpoint on the held-out test set. "
            "Do not use these results to tune the detector threshold or model."
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
        default=str(PROJECT_ROOT / "data" / "processed" / "normalized_v1" / "normalization_metadata.json"),
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

    summary = evaluate_checkpoint(
        Path(args.checkpoint).resolve(),
        Path(args.test_dir).resolve(),
        Path(args.normalization_metadata).resolve(),
        expected_records,
        expected_files,
        args.batch_size,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "independent_test_metrics.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved independent test metrics: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
