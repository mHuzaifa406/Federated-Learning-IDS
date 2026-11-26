"""Binary clean FedAvg reproduction runner for the Yang et al. IDS task.

The paper's Section 4.1 explicitly describes a binary classifier (labels 0/1),
although its dataset table also lists seven traffic categories. This runner
maps CIC-IDS-2017 Label_id=0 (Benign) to binary class 0 and all non-Benign
classes (Label_id 1..6) to binary class 1 (Attack).

It intentionally starts with clean, unweighted FedAvg: 10 clients, 10 rounds,
one local epoch, SGD lr=0.01, batch size inherited from the project engine.
This is a controlled implementation baseline, not a claim of exact code-level
reproduction of the paper (the paper does not disclose all model details).

Only the server validation split is used. The independent test split is never
read. Do not compare validation accuracy directly with the paper's figure as
a final claim; use the binary task alignment first, then perform one final
held-out test evaluation after freezing the configuration.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from torch.utils.data import DataLoader, TensorDataset

import scripts.federated_engine as engine
from models.ids_model import IntrusionDetectionMLP
from trfa.aggregation import aggregate_accepted_models

CLIENT_IDS = list(range(1, 11))
BINARY_CLASS_NAMES = ["Benign", "Attack"]
LABEL_COLUMNS = {"Label", "Label_7class", "Label_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, default=46)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--partition-root",
        default=str(PROJECT_ROOT / "data" / "processed" / "federated_v2"),
    )
    parser.add_argument("--output-tag", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(engine.CPU_THREADS)


def to_binary_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Cannot convert an empty label array.")
    if np.any((labels < 0) | (labels > 6)):
        raise ValueError("Expected original CIC-IDS-2017 Label_id values in 0..6.")
    return (labels != 0).astype(np.int64)


def load_validation(data_root: Path):
    validation_path = data_root / "server_validation" / "validation.parquet"
    if not validation_path.is_file():
        raise FileNotFoundError(f"Validation file not found: {validation_path}")

    frame = pd.read_parquet(validation_path)
    feature_columns = [c for c in frame.columns if c not in LABEL_COLUMNS]
    if len(feature_columns) != 78:
        raise ValueError(f"Expected 78 features, found {len(feature_columns)}.")
    if "Label_id" not in frame.columns:
        raise ValueError("Validation parquet is missing Label_id.")

    x = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    original_y = frame["Label_id"].to_numpy(dtype=np.int64, copy=True)
    y = to_binary_labels(original_y)
    del frame

    if not np.isfinite(x).all():
        raise ValueError("Validation features contain NaN or infinite values.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Validation data must contain both Benign and Attack labels.")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=engine.VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return feature_columns, loader, np.bincount(y, minlength=2).tolist()


def load_clients(data_root: Path, feature_columns: list[str]):
    engine.DATA_ROOT = data_root
    clients: dict[int, dict[str, Any]] = {}
    for cid in CLIENT_IDS:
        x, original_y, sample_count = engine.load_client_data(cid, feature_columns)
        binary_y = to_binary_labels(original_y)
        clients[cid] = {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "y": torch.from_numpy(binary_y),
            "samples": int(sample_count),
            "class_counts": np.bincount(binary_y, minlength=2).tolist(),
        }
        if len(clients[cid]["y"]) != int(sample_count):
            raise ValueError(f"Client {cid} sample count mismatch.")
        del x, original_y, binary_y
        gc.collect()
    return clients


def train_local(
    global_state: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    local_epochs: int,
    learning_rate: float,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=engine.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(seed)),
    )

    model = IntrusionDetectionMLP(input_features=78, num_classes=2).to(engine.DEVICE)
    model.load_state_dict(global_state, strict=True)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(int(local_epochs)):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(engine.DEVICE)
            batch_y = batch_y.to(engine.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    del loader, model, optimizer, criterion
    gc.collect()
    return state


def evaluate(model: torch.nn.Module, loader: DataLoader) -> dict[str, Any]:
    model.eval()
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []

    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(engine.DEVICE))
            pred = logits.argmax(dim=1)
            true_parts.append(y.cpu().numpy().astype(np.int64))
            pred_parts.append(pred.cpu().numpy().astype(np.int64))

    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    labels = np.array([0, 1], dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    benign_count = int(support[0])
    attack_count = int(support[1])
    false_alarm_count = int(matrix[0, 1])
    attack_to_benign_count = int(matrix[1, 0])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "validation_records": int(len(y_true)),
        "per_class_metrics": {
            BINARY_CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "predicted_count": int(matrix[:, i].sum()),
            }
            for i in range(2)
        },
        "benign_false_alarm_count": false_alarm_count,
        "benign_false_alarm_rate": float(false_alarm_count / benign_count) if benign_count else None,
        "attack_to_benign_count": attack_to_benign_count,
        "attack_to_benign_rate": float(attack_to_benign_count / attack_count) if attack_count else None,
        "confusion_matrix_rows_true_columns_predicted": matrix.astype(int).tolist(),
    }


def run(args: argparse.Namespace) -> Path:
    if args.rounds < 1 or args.local_epochs < 1 or args.learning_rate <= 0:
        raise ValueError("Rounds, local epochs and learning rate must be positive.")

    tag = re.sub(r"[^A-Za-z0-9_-]", "", args.output_tag)
    if not tag:
        raise ValueError("--output-tag must contain letters, digits, '-' or '_'.")

    data_root = Path(args.partition_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Partition directory not found: {data_root}")

    output_dir = PROJECT_ROOT / "results" / f"binary_fedavg_s{args.training_seed}_{tag}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output folder already has files: {output_dir}. Choose another tag.")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.training_seed)
    feature_columns, validation_loader, validation_counts = load_validation(data_root)
    clients = load_clients(data_root, feature_columns)
    global_class_counts = np.sum(
        np.asarray([clients[cid]["class_counts"] for cid in CLIENT_IDS], dtype=np.int64),
        axis=0,
    ).tolist()

    model = IntrusionDetectionMLP(input_features=78, num_classes=2).to(engine.DEVICE)
    history: list[dict[str, Any]] = []
    checkpoint_path = None

    print("=" * 72)
    print("BINARY CLEAN FEDAVG — PAPER TASK ALIGNMENT")
    print("=" * 72)
    print("Binary mapping: Label_id 0 -> Benign (0); Label_id 1..6 -> Attack (1)")
    print("Clients:", len(CLIENT_IDS))
    print("Rounds:", args.rounds)
    print("Local epochs:", args.local_epochs)
    print("Learning rate:", args.learning_rate)
    print("Training binary class counts:", global_class_counts)
    print("Validation binary class counts:", validation_counts)
    print("Independent test set: NOT USED")

    for round_no in range(1, int(args.rounds) + 1):
        local_states: dict[int, dict[str, torch.Tensor]] = {}
        sample_sizes: dict[int, int] = {}

        for cid in CLIENT_IDS:
            client_seed = int(args.training_seed) + round_no * 1000 + cid
            local_states[cid] = train_local(
                model.state_dict(),
                clients[cid]["x"],
                clients[cid]["y"],
                client_seed,
                args.local_epochs,
                args.learning_rate,
            )
            sample_sizes[cid] = int(clients[cid]["samples"])

        aggregated, weights = aggregate_accepted_models(
            client_states=local_states,
            sample_sizes=sample_sizes,
            accepted_client_ids=CLIENT_IDS,
            minimum_accepted_clients=2,
        )
        model.load_state_dict(aggregated, strict=True)
        metrics = evaluate(model, validation_loader)
        val_loss = float(engine.evaluate_validation_loss(model, validation_loader))

        history.append({
            "round": int(round_no),
            "validation_loss": val_loss,
            **{f"validation_{k}": v for k, v in metrics.items()},
        })

        checkpoint_path = output_dir / f"binary_round_{round_no:02d}.pt"
        torch.save(
            {
                "round": int(round_no),
                "model_state_dict": model.state_dict(),
                "model_config": {"input_features": 78, "num_classes": 2},
                "label_mapping": {"Benign": 0, "Attack": 1},
                "validation_metrics": metrics,
                "validation_loss": val_loss,
                "training_seed": int(args.training_seed),
                "partition_root": str(data_root),
            },
            checkpoint_path,
        )
        print(
            f"Round {round_no:02d}/{args.rounds}: "
            f"accuracy={metrics['accuracy']:.6f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.6f}, "
            f"macro_f1={metrics['macro_f1']:.6f}, "
            f"weighted_f1={metrics['weighted_f1']:.6f}"
        )
        del local_states
        gc.collect()

    final_metrics = evaluate(model, validation_loader)
    summary = {
        "experiment": "binary_clean_fedavg",
        "paper_alignment": (
            "Binary IDS classification as explicitly stated in Yang et al. Section 4.1; "
            "all non-Benign CIC-IDS-2017 classes are collapsed into Attack."
        ),
        "not_exact_reproduction": (
            "The paper does not fully disclose the model architecture and all preprocessing details. "
            "This run uses the project's 78-128-64-2 MLP, current partition, and validation split."
        ),
        "dataset": "CIC-IDS-2017",
        "training_seed": int(args.training_seed),
        "partition_root": str(data_root),
        "number_of_clients": len(CLIENT_IDS),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(engine.BATCH_SIZE),
        "number_of_features": len(feature_columns),
        "number_of_classes": 2,
        "label_mapping": {"Benign": 0, "Attack": 1},
        "global_training_class_counts": {
            BINARY_CLASS_NAMES[i]: int(global_class_counts[i]) for i in range(2)
        },
        "validation_class_counts": {
            BINARY_CLASS_NAMES[i]: int(validation_counts[i]) for i in range(2)
        },
        "validation_only_model_development": True,
        "independent_test_set_used": False,
        "round_results": history,
        "final_validation_metrics": final_metrics,
        "final_checkpoint": checkpoint_path.name if checkpoint_path else None,
    }
    (output_dir / "binary_fedavg_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(output_dir / "binary_fedavg_history.csv", index=False)
    print("Saved summary:", output_dir / "binary_fedavg_summary.json")
    print("Saved checkpoint:", checkpoint_path)
    return output_dir


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
