"""Run parameterized Update Similarity V2 diagnostics for Phase 5 scenario validation.

Examples (from project root):
  python scripts\\run_similarity_v2_noniid.py --condition clean
  python scripts\\run_similarity_v2_noniid.py --condition poison-client07

Screening is disabled intentionally. This experiment measures signals; it does
not claim malicious-client detection or robust aggregation.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import scripts.federated_engine as engine
from models.ids_model import IntrusionDetectionMLP
from trfa.trust_metrics import flatten_model_update
from trfa.trust_similarity_v2 import calculate_update_similarity_v2
from trfa.aggregation import aggregate_accepted_models
from scripts.phase5_attack_utils import poison_eligible_labels

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "federated_noniid_alpha05_seed43"
OUTPUT_ROOT = PROJECT_ROOT / "results"
CLIENT_IDS = list(range(1, 11))
TARGET_LABEL = 0  # Benign


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5 parameterized Similarity V2 diagnostic.")
    parser.add_argument("--condition", required=True, choices=("clean", "poison"))
    parser.add_argument("--malicious-client", type=int, default=7)
    parser.add_argument("--poison-fraction", type=float, default=0.040566715)
    parser.add_argument("--training-seed", type=int, default=43)
    parser.add_argument("--poison-seed", type=int, default=2027)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--partition-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-tag", required=True)
    return parser.parse_args()


def set_seed(seed: int):
    engine.set_random_seed()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(engine.CPU_THREADS)


def load_metadata(data_root: Path):
    metadata_path = data_root / "metadata.json"
    if not data_root.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Non-IID data/metadata not found: {data_root}")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not math.isclose(float(metadata.get("dirichlet_alpha", -1)), 0.5):
        raise ValueError("Expected a class-wise Dirichlet partition with alpha 0.5.")
    return metadata


def poison_labels(y, client_id: int, poison_fraction: float, poison_seed: int):
    """Flip the requested fraction of eligible non-Benign labels to Benign."""
    if not 0.0 < float(poison_fraction) < 1.0:
        raise ValueError("--poison-fraction must be greater than 0 and less than 1.")
    return poison_eligible_labels(y, client_id, poison_fraction, poison_seed, TARGET_LABEL)


def evaluate(model, loader):
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(engine.DEVICE))
            p = torch.argmax(logits, dim=1)
            true.extend(y.detach().cpu().numpy().astype(int).tolist())
            pred.extend(p.detach().cpu().numpy().astype(int).tolist())
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
    }


def run(args):
    data_root = Path(args.partition_root).resolve()
    metadata = load_metadata(data_root)
    if args.malicious_client not in CLIENT_IDS:
        raise ValueError("--malicious-client must be an integer from 1 through 10.")
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1.")
    tag = "".join(c for c in args.output_tag if c.isalnum() or c in "-_")
    if not tag:
        raise ValueError("--output-tag must contain only letters, numbers, '-' or '_'.")
    poisoned_condition = args.condition == "poison"
    poison_condition = f"poison-client{args.malicious_client:02d}"
    scenario = (f"{poison_condition}_p{args.poison_fraction:.4f}_s{args.training_seed}_ps{args.poison_seed}"
                if poisoned_condition else f"clean_s{args.training_seed}")
    out = OUTPUT_ROOT / f"phase5_similarity_{scenario}_{tag}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {out}. Use a new --output-tag.")
    out.mkdir(parents=True, exist_ok=True)

    engine.DATA_ROOT = data_root
    set_seed(args.training_seed)
    print("\nPHASE 5 UPDATE SIMILARITY V2 DIAGNOSTIC")
    print("Condition:", "poison" if poisoned_condition else "clean")
    print("Malicious client:", args.malicious_client if poisoned_condition else "none")
    print("Poison fraction of eligible non-Benign records:", args.poison_fraction if poisoned_condition else 0)
    print("Poison seed:", args.poison_seed if poisoned_condition else "not applicable")
    print("Partition:", data_root)
    print("Partition alpha/seed:", metadata.get("dirichlet_alpha"), metadata.get("random_seed"))
    print("Training seed:", args.training_seed)
    print("Rounds:", args.rounds)
    print("Screening: DISABLED; all clients will be aggregated")

    global_model = IntrusionDetectionMLP().to(engine.DEVICE)
    validation_X, validation_y, feature_columns = engine.load_validation_data()
    validation_X = np.array(validation_X, dtype=np.float32, copy=True)
    validation_y = np.array(validation_y, dtype=np.int64, copy=True)
    validation_loader = engine.create_validation_loader(validation_X, validation_y)

    client_rows, round_rows = [], []
    poison_count = eligible_count = client_count = 0
    for round_no in range(1, args.rounds + 1):
        print(f"\nRound {round_no}/{args.rounds}")
        states, updates, sizes = {}, {}, {}
        round_poisoned = 0
        for cid in CLIENT_IDS:
            X, y, n = engine.load_client_data(cid, feature_columns)
            X = np.array(X, dtype=np.float32, copy=True)
            y = np.array(y, dtype=np.int64, copy=True)
            if poisoned_condition and cid == args.malicious_client:
                y, round_poisoned, eligible_count = poison_labels(y, cid, args.poison_fraction, args.poison_seed)
                client_count = int(n)
            local_model, _ = engine.train_local_model(global_model, X, y)
            local_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
            updates[cid] = flatten_model_update(local_state, global_model.state_dict())
            states[cid] = local_state
            sizes[cid] = int(n)
            del X, y, local_model
            gc.collect()

        poison_count = max(poison_count, round_poisoned)
        signal_rows = calculate_update_similarity_v2(updates)
        for cid in CLIENT_IDS:
            client_rows.append({
                "Round": round_no, "Client": cid,
                "Is_Malicious": bool(poisoned_condition and cid == args.malicious_client),
                **signal_rows[cid], "Client_Samples": sizes[cid],
                "Poisoned_Records": int(round_poisoned if cid == args.malicious_client and poisoned_condition else 0),
            })

        aggregate, weights = aggregate_accepted_models(states, sizes, CLIENT_IDS, minimum_accepted_clients=2)
        global_model.load_state_dict(aggregate)
        metrics = evaluate(global_model, validation_loader)
        val_loss = float(engine.evaluate_validation_loss(global_model, validation_loader))
        round_rows.append({"Round": round_no, "Validation_Loss": val_loss, **metrics,
                           "Accepted_Clients": len(CLIENT_IDS), "Aggregation_Weight_Sum": float(sum(weights.values())),
                           "Poisoned_Records": int(round_poisoned)})
        torch.save({"round": round_no, "condition": args.condition, "malicious_client": args.malicious_client if poisoned_condition else None,
                    "model_state_dict": global_model.state_dict(), "validation_metrics": metrics, "validation_loss": val_loss},
                   out / f"similarity_v2_round_{round_no:02d}.pt")
        print(f"accuracy={metrics['accuracy']:.6f}; macro_f1={metrics['macro_f1']:.6f}")
        del states, updates
        gc.collect()

    scores = pd.DataFrame(client_rows)
    rounds_df = pd.DataFrame(round_rows)
    rank_specs = {
        "Direction_Rank_Lowest_First": ("pairwise_direction_score", True),
        "Magnitude_Rank_Lowest_First": ("magnitude_agreement_score", True),
        "Reference_Distance_Rank_Highest_First": ("relative_reference_distance", False),
    }
    for rank_col, (source_col, ascending) in rank_specs.items():
        scores[rank_col] = scores.groupby("Round")[source_col].rank(method="average", ascending=ascending)
    summary = scores.groupby(["Client", "Is_Malicious"], as_index=False).agg(
        Pairwise_Direction_Mean=("pairwise_direction_score", "mean"),
        Pairwise_Direction_Std=("pairwise_direction_score", "std"),
        Magnitude_Agreement_Mean=("magnitude_agreement_score", "mean"),
        Relative_Magnitude_Mean=("relative_magnitude", "mean"),
        Reference_Distance_Agreement_Mean=("reference_distance_agreement_score", "mean"),
        Relative_Reference_Distance_Mean=("relative_reference_distance", "mean"),
        Mean_Direction_Rank=("Direction_Rank_Lowest_First", "mean"),
        Mean_Magnitude_Rank=("Magnitude_Rank_Lowest_First", "mean"),
    )
    scores.to_csv(out / "similarity_v2_client_round_scores.csv", index=False)
    rounds_df.to_csv(out / "similarity_v2_round_metrics.csv", index=False)
    summary.to_csv(out / "similarity_v2_client_summary.csv", index=False)
    meta = {
        "experiment": "phase5_update_similarity_v2_noniid_diagnostic", "condition": args.condition,
        "dataset": "CIC-IDS-2017", "partition_method": "class-wise Dirichlet",
        "dirichlet_alpha": float(metadata["dirichlet_alpha"]), "partition_seed": int(metadata["random_seed"]),
        "training_seed": int(args.training_seed), "clients": len(CLIENT_IDS), "rounds": int(args.rounds),
        "malicious_client": int(args.malicious_client) if poisoned_condition else None,
        "target_label": TARGET_LABEL if poisoned_condition else None,
        "poison_fraction_of_eligible_nonbenign": float(args.poison_fraction) if poisoned_condition else 0.0,
        "eligible_nonbenign_records": int(eligible_count) if poisoned_condition else 0,
        "client_records": int(client_count) if poisoned_condition else 0,
        "poisoned_records_per_round": int(poison_count),
        "actual_fraction_of_eligible": float(poison_count / eligible_count) if eligible_count else 0.0,
        "actual_fraction_of_client": float(poison_count / client_count) if client_count else 0.0,
        "poison_seed": int(args.poison_seed) if poisoned_condition else None,
        "screening_enabled": False, "independent_test_used": False,
        "interpretation": "Similarity diagnostics only; screening disabled; not a defense claim.",
    }
    (out / "similarity_v2_experiment_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nSIMILARITY V2 DIAGNOSTIC COMPLETED")
    print("Condition:", args.condition)
    print("Malicious client:", args.malicious_client if poisoned_condition else "none")
    print("Poisoned records per round:", poison_count)
    print("Actual fraction of eligible:", f"{(poison_count / eligible_count):.6%}" if eligible_count else "0")
    print("Final validation metrics:", round_rows[-1])
    print("Client summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("Output directory:", out)
    print("Screening enabled: NO; independent test used: NO")
    del validation_X, validation_y, validation_loader, global_model
    gc.collect()
    return out


if __name__ == "__main__":
    args = parse_args()
    run(args)
