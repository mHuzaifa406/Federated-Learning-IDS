"""Run diagnostic-only Update Similarity V2 on clean or poisoned non-IID FL.

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

DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "federated_noniid_alpha05_seed43"
OUTPUT_ROOT = PROJECT_ROOT / "results"
CLIENT_IDS = list(range(1, 11))
ROUNDS = 10
SEED = 43
MALICIOUS_CLIENT = 7
TARGET_LABEL = 0
POISON_SEED = 2027
POISON_CLIENT_FRACTION = 8676 / 213858


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=("clean", "poison-client07"))
    parser.add_argument("--output-tag", default="")
    return parser.parse_args()


def set_seed():
    engine.set_random_seed()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(engine.CPU_THREADS)


def load_metadata():
    metadata_path = DATA_ROOT / "metadata.json"
    if not DATA_ROOT.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Non-IID data/metadata not found: {DATA_ROOT}")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not math.isclose(float(metadata.get("dirichlet_alpha", -1)), 0.5):
        raise ValueError("Expected Dirichlet alpha 0.5.")
    if int(metadata.get("random_seed", -1)) != 43:
        raise ValueError("Expected non-IID partition seed 43.")
    return metadata


def poison_labels(y):
    poisoned = np.array(y, dtype=np.int64, copy=True)
    eligible = np.flatnonzero(poisoned != TARGET_LABEL)
    count = int(np.floor(len(poisoned) * POISON_CLIENT_FRACTION))
    if count > len(eligible):
        raise ValueError("Insufficient eligible labels for poisoning.")
    rng = np.random.default_rng(POISON_SEED + MALICIOUS_CLIENT)
    selected = rng.choice(eligible, size=count, replace=False)
    poisoned[selected] = TARGET_LABEL
    actual = int(np.count_nonzero(poisoned != y))
    if actual != count:
        raise RuntimeError(f"Expected {count} poisoned labels, got {actual}.")
    return poisoned, actual


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


def run(condition, tag=""):
    metadata = load_metadata()
    poisoned_condition = condition == "poison-client07"
    folder = f"similarity_v2_{'poison_client07' if poisoned_condition else 'clean'}_noniid_seed43"
    if tag:
        safe = "".join(c for c in tag if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError("--output-tag must contain letters, numbers, '-' or '_'.")
        folder += f"_{safe}"
    out = OUTPUT_ROOT / folder
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {out}. Use --output-tag.")
    out.mkdir(parents=True, exist_ok=True)

    engine.DATA_ROOT = DATA_ROOT
    set_seed()
    print("\nUPDATE SIMILARITY V2 NON-IID DIAGNOSTIC")
    print("Condition:", condition)
    print("Partition alpha:", metadata["dirichlet_alpha"])
    print("Partition seed:", metadata["random_seed"])
    print("Training seed:", SEED)
    print("Rounds:", ROUNDS)
    print("Screening: DISABLED; all clients will be aggregated")

    global_model = IntrusionDetectionMLP().to(engine.DEVICE)
    validation_X, validation_y, feature_columns = engine.load_validation_data()
    validation_X = np.array(validation_X, dtype=np.float32, copy=True)
    validation_y = np.array(validation_y, dtype=np.int64, copy=True)
    validation_loader = engine.create_validation_loader(validation_X, validation_y)

    client_rows, round_rows = [], []
    poisoned_count = 0
    for round_no in range(1, ROUNDS + 1):
        print(f"\nRound {round_no}/{ROUNDS}")
        states, updates, sizes = {}, {}, {}
        round_poisoned = 0
        for cid in CLIENT_IDS:
            X, y, n = engine.load_client_data(cid, feature_columns)
            X = np.array(X, dtype=np.float32, copy=True)
            y = np.array(y, dtype=np.int64, copy=True)
            if poisoned_condition and cid == MALICIOUS_CLIENT:
                y, round_poisoned = poison_labels(y)
            local_model, _ = engine.train_local_model(global_model, X, y)
            local_state = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
            updates[cid] = flatten_model_update(local_state, global_model.state_dict())
            states[cid] = local_state
            sizes[cid] = int(n)
            del X, y, local_model
            gc.collect()

        poisoned_count = max(poisoned_count, round_poisoned)
        signal_rows = calculate_update_similarity_v2(updates)
        for cid in CLIENT_IDS:
            client_rows.append({
                "Round": round_no,
                "Client": cid,
                "Is_Malicious": bool(poisoned_condition and cid == MALICIOUS_CLIENT),
                **signal_rows[cid],
                "Client_Samples": sizes[cid],
                "Poisoned_Records": int(round_poisoned if cid == MALICIOUS_CLIENT and poisoned_condition else 0),
            })

        # Diagnostic only: retain all clients so the global training trajectory
        # is not affected by a candidate threshold during signal calibration.
        aggregate, weights = aggregate_accepted_models(states, sizes, CLIENT_IDS, minimum_accepted_clients=2)
        global_model.load_state_dict(aggregate)
        metrics = evaluate(global_model, validation_loader)
        val_loss = float(engine.evaluate_validation_loss(global_model, validation_loader))
        round_rows.append({"Round": round_no, "Validation_Loss": val_loss, **metrics,
                           "Accepted_Clients": len(CLIENT_IDS), "Aggregation_Weight_Sum": float(sum(weights.values())),
                           "Poisoned_Records": int(round_poisoned)})
        torch.save({"round": round_no, "condition": condition,
                    "model_state_dict": global_model.state_dict(),
                    "validation_metrics": metrics, "validation_loss": val_loss},
                   out / f"similarity_v2_round_{round_no:02d}.pt")
        print(f"accuracy={metrics['accuracy']:.6f}; macro_f1={metrics['macro_f1']:.6f}")
        del states, updates
        gc.collect()

    scores = pd.DataFrame(client_rows)
    rounds_df = pd.DataFrame(round_rows)
    scores.to_csv(out / "similarity_v2_client_round_scores.csv", index=False)
    rounds_df.to_csv(out / "similarity_v2_round_metrics.csv", index=False)

    # Within-round ranks are calculated across clients, not across a single
    # client's repeated rounds. Lower direction/magnitude agreement ranks first.
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
    summary.to_csv(out / "similarity_v2_client_summary.csv", index=False)
    meta = {
        "experiment": "update_similarity_v2_noniid_diagnostic", "condition": condition,
        "dataset": "CIC-IDS-2017", "partition_method": "class-wise Dirichlet",
        "dirichlet_alpha": 0.5, "partition_seed": 43, "training_seed": SEED,
        "clients": len(CLIENT_IDS), "rounds": ROUNDS,
        "malicious_client": MALICIOUS_CLIENT if poisoned_condition else None,
        "target_label": TARGET_LABEL if poisoned_condition else None,
        "poisoned_records_per_round": poisoned_count,
        "poison_fraction_of_client": poisoned_count / int(sizes[MALICIOUS_CLIENT]) if poisoned_condition else 0.0,
        "screening_enabled": False, "independent_test_used": False,
        "interpretation": "Similarity diagnostics only; no detection or defense claim.",
    }
    with (out / "similarity_v2_experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSIMILARITY V2 DIAGNOSTIC COMPLETED")
    print("Condition:", condition)
    print("Poisoned records per round:", poisoned_count)
    print("Final validation metrics:", round_rows[-1])
    print("Client summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("Output directory:", out)
    print("Screening enabled: NO")
    print("Independent test used: NO")
    del validation_X, validation_y, validation_loader, global_model
    gc.collect()
    return out

if __name__ == "__main__":
    args = parse_args()
    run(args.condition, args.output_tag)
