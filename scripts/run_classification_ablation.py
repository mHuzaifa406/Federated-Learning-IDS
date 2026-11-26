"""Clean classification ablation for the federated IDS.

Purpose:
- Compare the current unweighted FedAvg training objective with training-only
  class-weighting and/or mild local class-aware sampling.
- Keep model architecture, client partition, optimizer, batch size and seeds
  matched across modes.
- Use only the server validation split for model-development decisions.
- Never read or evaluate the independent test split.

Modes:
  baseline        unweighted CrossEntropy + shuffled local batches
  weighted_loss   capped inverse-square-root class-weighted CrossEntropy
  sampler         mild local WeightedRandomSampler + ordinary CrossEntropy
  combined        class-weighted CrossEntropy + mild local sampler

This is a classifier-training ablation, not a poisoning detector. Run this
first on the paper-style stratified/IID-like partition. Re-run the selected
configuration on the non-IID partition before making any robustness claim.
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
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import scripts.federated_engine as engine
from models.ids_model import IntrusionDetectionMLP
from trfa.aggregation import aggregate_accepted_models

CLIENT_IDS = list(range(1, 11))
CLASS_IDS = np.arange(7, dtype=np.int64)
CLASS_NAMES = [
    "Benign",
    "Bot",
    "DoS",
    "Infiltration",
    "PortScan",
    "Web Attack",
    "Brute Force",
]
VALID_MODES = ("baseline", "weighted_loss", "sampler", "combined")
CLASS_WEIGHT_CAP = 10.0
SAMPLER_POWER = 0.25
SAMPLER_WEIGHT_CAP = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=VALID_MODES)
    parser.add_argument("--training-seed", type=int, default=46)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--partition-root",
        default=str(PROJECT_ROOT / "data" / "processed" / "federated_v2"),
        help="Use federated_v2 first for the paper-style stratified/IID-like comparison.",
    )
    parser.add_argument("--output-tag", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(engine.CPU_THREADS)


def compute_global_class_weights(
    clients: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, list[int], list[float]]:
    counts = np.zeros(7, dtype=np.int64)
    for client in clients.values():
        counts += np.bincount(client["y"].numpy(), minlength=7)

    if np.any(counts <= 0):
        raise ValueError(
            f"At least one class is absent from the combined training data: {counts.tolist()}"
        )

    raw = np.sqrt(counts.max() / counts.astype(np.float64))
    capped = np.minimum(raw, CLASS_WEIGHT_CAP)
    normalized = capped / capped.mean()
    return normalized.astype(np.float32), counts.tolist(), normalized.tolist()


def make_local_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    client_seed: int,
) -> DataLoader:
    dataset = TensorDataset(x, y)
    generator = torch.Generator()
    generator.manual_seed(int(client_seed))

    use_sampler = mode in ("sampler", "combined")
    if not use_sampler:
        return DataLoader(
            dataset,
            batch_size=engine.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

    labels = y.numpy()
    counts = np.bincount(labels, minlength=7)
    present = counts > 0
    factors = np.ones(7, dtype=np.float64)
    if np.any(present):
        factors[present] = np.minimum(
            (counts[present].max() / counts[present]) ** SAMPLER_POWER,
            SAMPLER_WEIGHT_CAP,
        )
    record_weights = torch.as_tensor(factors[labels], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=record_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=engine.BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        generator=generator,
    )


def train_one_client(
    global_state: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    global_class_weights: np.ndarray,
    local_epochs: int,
    learning_rate: float,
    client_seed: int,
) -> tuple[dict[str, torch.Tensor], float]:
    torch.manual_seed(int(client_seed))
    loader = make_local_loader(x, y, mode, client_seed)

    local_model = IntrusionDetectionMLP(
        input_features=78,
        num_classes=7,
    ).to(engine.DEVICE)
    local_model.load_state_dict(global_state, strict=True)
    local_model.train()

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=float(learning_rate),
    )
    weighted = mode in ("weighted_loss", "combined")
    weight_tensor = (
        torch.as_tensor(global_class_weights, dtype=torch.float32, device=engine.DEVICE)
        if weighted
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    loss_sum = 0.0
    sample_sum = 0
    for _ in range(int(local_epochs)):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(engine.DEVICE)
            batch_y = batch_y.to(engine.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = local_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            n = int(len(batch_y))
            loss_sum += float(loss.detach().item()) * n
            sample_sum += n

    if sample_sum == 0:
        raise RuntimeError("A client completed zero local-training samples.")

    state = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }
    mean_loss = loss_sum / sample_sum
    del loader, local_model, optimizer, criterion, weight_tensor
    gc.collect()
    return state, float(mean_loss)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
) -> dict[str, Any]:
    model.eval()
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []

    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(engine.DEVICE))
            pred = torch.argmax(logits, dim=1)
            true_parts.append(y.cpu().numpy().astype(np.int64))
            pred_parts.append(pred.cpu().numpy().astype(np.int64))

    if not true_parts:
        raise ValueError("Validation loader returned no records.")

    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)

    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
            "predicted_count": int(matrix[:, idx].sum()),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=CLASS_IDS, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, labels=CLASS_IDS, average="weighted", zero_division=0
        )),
        "validation_records": int(len(y_true)),
        "class_counts": {
            CLASS_NAMES[i]: int(np.sum(y_true == i)) for i in range(7)
        },
        "per_class_metrics": per_class,
        "confusion_matrix_rows_true_columns_predicted": matrix.astype(int).tolist(),
    }


def load_clients_and_validation(data_root: Path):
    if not data_root.is_dir():
        raise FileNotFoundError(f"Partition directory not found: {data_root}")

    engine.DATA_ROOT = data_root
    engine.VALIDATION_PATH = (
        data_root / "server_validation" / "validation.parquet"
    )

    validation_x, validation_y, feature_columns = engine.load_validation_data()
    if len(feature_columns) != 78:
        raise ValueError(f"Expected 78 features, found {len(feature_columns)}.")

    validation_loader = engine.create_validation_loader(
        np.asarray(validation_x, dtype=np.float32),
        np.asarray(validation_y, dtype=np.int64),
    )

    clients: dict[int, dict[str, Any]] = {}
    for cid in CLIENT_IDS:
        x, y, count = engine.load_client_data(cid, feature_columns)
        clients[cid] = {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "y": torch.from_numpy(np.asarray(y, dtype=np.int64)),
            "samples": int(count),
        }
        if len(clients[cid]["y"]) != int(count):
            raise ValueError(f"Client {cid} sample count does not match its labels.")
        del x, y
        gc.collect()

    return clients, validation_loader, feature_columns


def run(args: argparse.Namespace) -> Path:
    if args.rounds < 1 or args.local_epochs < 1 or args.learning_rate <= 0:
        raise ValueError("Rounds, local epochs and learning rate must be positive.")

    tag = re.sub(r"[^A-Za-z0-9_-]", "", args.output_tag)
    if not tag:
        raise ValueError("--output-tag must contain letters, digits, '-' or '_'.")

    data_root = Path(args.partition_root).resolve()
    output_dir = (
        PROJECT_ROOT
        / "results"
        / f"classification_ablation_{args.mode}_s{args.training_seed}_{tag}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}. Use another --output-tag."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.training_seed)
    clients, validation_loader, feature_columns = load_clients_and_validation(data_root)
    class_weights, class_counts, normalized_weights = compute_global_class_weights(clients)

    if args.mode not in ("weighted_loss", "combined"):
        # The class-weight vector is still recorded for audit, but not used.
        print("Class-weighted loss: disabled")
    else:
        print("Class-weighted loss weights:", normalized_weights)

    print("Partition:", data_root)
    print("Mode:", args.mode)
    print("Training seed:", args.training_seed)
    print("Rounds:", args.rounds)
    print("Local epochs:", args.local_epochs)
    print("Learning rate:", args.learning_rate)
    print("Training-only class counts:", class_counts)
    print("Independent test set: NOT USED")

    global_model = IntrusionDetectionMLP(
        input_features=len(feature_columns),
        num_classes=7,
    ).to(engine.DEVICE)
    global_history: list[dict[str, Any]] = []
    checkpoint_path = None

    for round_no in range(1, args.rounds + 1):
        local_states: dict[int, dict[str, torch.Tensor]] = {}
        sample_sizes: dict[int, int] = {}
        client_losses: dict[str, float] = {}

        for cid in CLIENT_IDS:
            client_seed = int(args.training_seed) + round_no * 1000 + cid
            state, local_loss = train_one_client(
                global_model.state_dict(),
                clients[cid]["x"],
                clients[cid]["y"],
                args.mode,
                class_weights,
                args.local_epochs,
                args.learning_rate,
                client_seed,
            )
            local_states[cid] = state
            sample_sizes[cid] = int(clients[cid]["samples"])
            client_losses[str(cid)] = local_loss

        aggregated, aggregation_weights = aggregate_accepted_models(
            client_states=local_states,
            sample_sizes=sample_sizes,
            accepted_client_ids=CLIENT_IDS,
            minimum_accepted_clients=2,
        )
        global_model.load_state_dict(aggregated, strict=True)
        metrics = evaluate_model(global_model, validation_loader)
        validation_loss = float(
            engine.evaluate_validation_loss(global_model, validation_loader)
        )

        row = {
            "round": int(round_no),
            "validation_loss": validation_loss,
            **{f"validation_{k}": v for k, v in metrics.items()
               if k not in ("per_class_metrics", "class_counts",
                            "confusion_matrix_rows_true_columns_predicted")},
            "accepted_clients": CLIENT_IDS.copy(),
            "excluded_clients": [],
        }
        global_history.append(row)

        checkpoint_path = output_dir / f"phase11_round_{round_no:02d}.pt"
        torch.save(
            {
                "round": int(round_no),
                "model_state_dict": global_model.state_dict(),
                "validation_metrics": metrics,
                "validation_loss": validation_loss,
                "training_mode": args.mode,
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

    final_metrics = evaluate_model(global_model, validation_loader)
    summary = {
        "experiment": "clean_classification_ablation",
        "dataset": "CIC-IDS-2017",
        "mode": args.mode,
        "training_seed": int(args.training_seed),
        "partition_root": str(data_root),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "learning_rate": float(args.learning_rate),
        "clients": len(CLIENT_IDS),
        "features": len(feature_columns),
        "classes": CLASS_NAMES,
        "global_training_class_counts": class_counts,
        "class_weights_method": "capped_inverse_square_root_frequency",
        "class_weights_cap": CLASS_WEIGHT_CAP,
        "normalized_class_weights": normalized_weights,
        "sampler_method": "local_capped_power_law_weighted_random_sampler",
        "sampler_power": SAMPLER_POWER,
        "sampler_weight_cap": SAMPLER_WEIGHT_CAP,
        "validation_only_model_development": True,
        "independent_test_set_used": False,
        "round_results": global_history,
        "final_validation_metrics": final_metrics,
        "final_checkpoint": checkpoint_path.name if checkpoint_path else None,
    }
    (output_dir / "classification_ablation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(global_history).to_csv(
        output_dir / "classification_ablation_history.csv",
        index=False,
    )
    print("Saved:", output_dir / "classification_ablation_summary.json")
    print("Saved final checkpoint:", checkpoint_path)
    return output_dir


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
