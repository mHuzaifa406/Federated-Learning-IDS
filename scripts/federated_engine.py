
"""
TRFA Federated Training Engine
------------------------------
Executes one federated communication round using:

1. Local client training
2. Update Similarity
3. Validation-based Training Consistency
4. Historical Trust
5. Trust-based client screening
6. Sample-size-weighted aggregation
"""

from pathlib import Path
import sys
import copy
import json
import gc

# Make the project root available before importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from models.ids_model import IntrusionDetectionMLP

from trfa.trust_metrics import (
    flatten_model_update,
    calculate_reference_update,
    calculate_update_similarity,
    calculate_training_consistency,
    HistoricalTrustTracker,
    calculate_combined_trust,
    screen_clients,
)

from trfa.aggregation import aggregate_accepted_models


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "federated_v2"

VALIDATION_PATH = (
    DATA_ROOT / "server_validation" / "validation.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "trfa_round_tests"

DEVICE = torch.device("cpu")

NUM_CLIENTS = 10
NUM_FEATURES = 78
NUM_CLASSES = 7

BATCH_SIZE = 512
VALIDATION_BATCH_SIZE = 2048

LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1

# These values match the TRFA configuration tested earlier.
ALPHA = 1.0 / 3.0
BETA = 1.0 / 3.0
GAMMA = 1.0 / 3.0

TRUST_THRESHOLD = 0.50
HISTORY_RETENTION = 0.90
INITIAL_HISTORICAL_TRUST = 0.50

RANDOM_SEED = 42
CPU_THREADS = 4


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_random_seed():
    """Set random seeds for repeatable execution."""

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(CPU_THREADS)


# ============================================================
# 3. DATA LOADING
# ============================================================

def get_feature_columns(dataframe):
    """Identify the 78 model input features."""

    label_columns = {
        "Label_id",
        "Label",
        "Label_7class"
    }

    return [
        column for column in dataframe.columns
        if column not in label_columns
    ]


def load_validation_data():
    """Load the clean server-validation dataset."""

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_PATH}"
        )

    dataframe = pd.read_parquet(VALIDATION_PATH)

    if "Label_id" not in dataframe.columns:
        raise ValueError(
            "Label_id is missing from the validation dataset."
        )

    feature_columns = get_feature_columns(dataframe)

    if len(feature_columns) != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} validation features, "
            f"found {len(feature_columns)}."
        )

    X = dataframe[feature_columns].to_numpy(dtype=np.float32)
    y = dataframe["Label_id"].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise ValueError(
            "Validation dataset contains non-finite feature values."
        )

    if len(y) == 0:
        raise ValueError("Validation dataset is empty.")

    if y.min() < 0 or y.max() >= NUM_CLASSES:
        raise ValueError("Validation dataset contains invalid labels.")

    print("Validation records:", f"{len(y):,}")
    print("Validation features:", len(feature_columns))

    del dataframe
    gc.collect()

    return X, y, feature_columns


def load_client_data(client_id, feature_columns):
    """Load one federated client's training data."""

    client_name = f"client_{client_id:02d}"
    client_dir = DATA_ROOT / client_name

    if not client_dir.exists():
        raise FileNotFoundError(
            f"Client directory not found: {client_dir}"
        )

    parquet_files = sorted(client_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found for {client_name}."
        )

    frames = [
        pd.read_parquet(file)
        for file in parquet_files
    ]

    dataframe = pd.concat(frames, ignore_index=True)

    if "Label_id" not in dataframe.columns:
        raise ValueError(
            f"Label_id is missing for {client_name}."
        )

    actual_features = get_feature_columns(dataframe)

    if actual_features != feature_columns:
        raise ValueError(
            f"Feature names or ordering do not match for {client_name}."
        )

    X = dataframe[feature_columns].to_numpy(dtype=np.float32)
    y = dataframe["Label_id"].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite feature values found for {client_name}."
        )

    if len(y) == 0:
        raise ValueError(f"{client_name} has no training records.")

    if y.min() < 0 or y.max() >= NUM_CLASSES:
        raise ValueError(
            f"Invalid class labels found for {client_name}."
        )

    sample_count = len(y)

    del dataframe, frames
    gc.collect()

    return X, y, sample_count


# ============================================================
# 4. DATA LOADERS
# ============================================================

def create_training_loader(X, y):
    """Create the local client's training loader."""

    features = torch.from_numpy(X)
    labels = torch.from_numpy(y)

    dataset = TensorDataset(features, labels)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


def create_validation_loader(X, y):
    """Create the common clean validation loader."""

    features = torch.from_numpy(X)
    labels = torch.from_numpy(y)

    dataset = TensorDataset(features, labels)

    return DataLoader(
        dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


# ============================================================
# 5. LOCAL CLIENT TRAINING
# ============================================================

def train_local_model(global_model, X, y):
    """
    Train a local model initialized from the current global model.

    Returns:
        local_model
        average_local_training_loss
    """

    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.train()

    training_loader = create_training_loader(X, y)

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    for epoch in range(LOCAL_EPOCHS):

        for batch_X, batch_y in training_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs = local_model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            batch_size = len(batch_y)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    average_loss = total_loss / max(total_samples, 1)

    return local_model, average_loss


# ============================================================
# 6. VALIDATION-BASED MODEL EVALUATION
# ============================================================

def evaluate_validation_loss(model, validation_loader):
    """
    Evaluate a model using the same clean validation records.

    The returned value is the average cross-entropy loss
    over all validation records.
    """

    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch_X, batch_y in validation_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            total_samples += len(batch_y)

    return total_loss / max(total_samples, 1)


# ============================================================
# 7. ONE FEDERATED COMMUNICATION ROUND
# ============================================================

def run_one_round():

    print("\n" + "=" * 65)
    print("TRFA FEDERATED TRAINING - COMMUNICATION ROUND 1")
    print("=" * 65)

    set_random_seed()

    # --------------------------------------------------------
    # Initialize global model
    # --------------------------------------------------------

    print("\n[1/7] Initializing global model...")

    global_model = IntrusionDetectionMLP().to(DEVICE)

    global_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    print("Model initialized successfully.")
    print("Execution device:", DEVICE)

    # --------------------------------------------------------
    # Load common validation data
    # --------------------------------------------------------

    print("\n[2/7] Loading clean server-validation data...")

    validation_X, validation_y, feature_columns = (
        load_validation_data()
    )

    validation_loader = create_validation_loader(
        validation_X,
        validation_y
    )

    baseline_validation_loss = evaluate_validation_loss(
        global_model,
        validation_loader
    )

    print(
        "Initial global validation loss:",
        f"{baseline_validation_loss:.6f}"
    )

    # --------------------------------------------------------
    # Initialize historical trust
    # --------------------------------------------------------

    client_ids = list(range(1, NUM_CLIENTS + 1))

    historical_tracker = HistoricalTrustTracker(
        client_ids=client_ids,
        retention=HISTORY_RETENTION,
        initial_trust=INITIAL_HISTORICAL_TRUST
    )

    previous_historical_scores = (
        historical_tracker.get_scores(client_ids)
    )

    # --------------------------------------------------------
    # Train local models
    # --------------------------------------------------------

    print("\n[3/7] Training local client models...")

    client_states = {}
    client_updates = {}
    client_sizes = {}
    client_training_losses = {}
    candidate_validation_losses = {}

    for client_id in client_ids:

        print(
            f"\nTraining client {client_id:02d}/{NUM_CLIENTS}..."
        )

        X, y, sample_count = load_client_data(
            client_id,
            feature_columns
        )

        local_model, training_loss = train_local_model(
            global_model,
            X,
            y
        )

        # Save an independent copy of the complete local state.
        local_state = {
            key: value.detach().cpu().clone()
            for key, value in local_model.state_dict().items()
        }

        # Evaluate the local candidate on the same clean
        # server-validation dataset used for every client.
        candidate_loss = evaluate_validation_loss(
            local_model,
            validation_loader
        )

        # Convert the model update into a flattened vector
        # for update-similarity calculation.
        update_vector = flatten_model_update(
            local_state,
            global_state
        )

        client_states[client_id] = local_state
        client_updates[client_id] = update_vector
        client_sizes[client_id] = int(sample_count)

        client_training_losses[client_id] = float(training_loss)
        candidate_validation_losses[client_id] = float(candidate_loss)

        print("  Training samples:", f"{sample_count:,}")
        print("  Local training loss:", f"{training_loss:.6f}")
        print("  Candidate validation loss:", f"{candidate_loss:.6f}")

        del X, y, local_model
        gc.collect()

    # --------------------------------------------------------
    # Calculate the three trust signals
    # --------------------------------------------------------

    print("\n[4/7] Calculating TRFA trust signals...")

    # Signal 1: Update similarity
    reference_update = calculate_reference_update(
        client_updates
    )

    similarity_scores = calculate_update_similarity(
        client_updates,
        reference_update
    )

    # Signal 2: Training consistency
    consistency_scores = calculate_training_consistency(
        baseline_validation_loss,
        candidate_validation_losses
    )

    # Signal 3: Historical trust
    historical_scores = historical_tracker.get_scores(
        client_ids
    )

    # Combine all three trust signals.
    combined_trust_scores = calculate_combined_trust(
        similarity_scores=similarity_scores,
        consistency_scores=consistency_scores,
        historical_scores=historical_scores,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA
    )

    print("\nTrust score details:")

    print(
        "\n{:<12} {:<13} {:<13} {:<13} {:<13}".format(
            "Client",
            "Similarity",
            "Consistency",
            "Historical",
            "Combined"
        )
    )

    for client_id in client_ids:

        print(
            "{:<12} {:<13.4f} {:<13.4f} {:<13.4f} {:<13.4f}".format(
                f"Client {client_id:02d}",
                similarity_scores[client_id],
                consistency_scores[client_id],
                historical_scores[client_id],
                combined_trust_scores[client_id]
            )
        )

    # --------------------------------------------------------
    # Screen clients
    # --------------------------------------------------------

    print("\n[5/7] Applying TRFA trust-based screening...")

    accepted_clients, rejected_clients = screen_clients(
        combined_trust_scores,
        threshold=TRUST_THRESHOLD
    )

    print("Trust threshold:", TRUST_THRESHOLD)
    print("Accepted clients:", accepted_clients)
    print("Excluded clients:", rejected_clients)

    if len(accepted_clients) < 2:
        raise RuntimeError(
            "Fewer than two clients passed TRFA screening. "
            "Aggregation was stopped."
        )

    # --------------------------------------------------------
    # Aggregate accepted client model states
    # --------------------------------------------------------

    print("\n[6/7] Aggregating accepted client models...")

    aggregated_state, aggregation_weights = (
        aggregate_accepted_models(
            client_states=client_states,
            sample_sizes=client_sizes,
            accepted_client_ids=accepted_clients,
            minimum_accepted_clients=2
        )
    )

    # The aggregation module returns the complete new global
    # model state. Do not add this state to the previous model.
    global_model.load_state_dict(aggregated_state)

    print("\nAggregation weights:")

    for client_id, weight in aggregation_weights.items():

        print(
            f"Client {client_id:02d}: {weight:.6f}"
        )

    print(
        "Total aggregation weight:",
        f"{sum(aggregation_weights.values()):.6f}"
    )

    # --------------------------------------------------------
    # Update historical trust after current-round evaluation
    # --------------------------------------------------------

    historical_tracker.update(combined_trust_scores)

    updated_historical_scores = (
        historical_tracker.get_scores(client_ids)
    )

    # --------------------------------------------------------
    # Save results and checkpoint
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        OUTPUT_DIR / "round_01_global_model.pt"
    )

    checkpoint = {
        "round": 1,
        "model_state_dict": global_model.state_dict(),
        "accepted_clients": accepted_clients,
        "rejected_clients": rejected_clients,
        "aggregation_weights": aggregation_weights,
        "trust_scores": combined_trust_scores,
        "historical_trust_after_update": updated_historical_scores,
        "client_sizes": client_sizes,
        "client_training_losses": client_training_losses,
        "candidate_validation_losses": candidate_validation_losses,
        "baseline_validation_loss": baseline_validation_loss,
        "trust_threshold": TRUST_THRESHOLD
    }

    torch.save(checkpoint, checkpoint_path)

    summary = {
        "round": 1,
        "device": str(DEVICE),
        "clients_trained": len(client_ids),
        "accepted_clients": accepted_clients,
        "rejected_clients": rejected_clients,
        "trust_threshold": TRUST_THRESHOLD,
        "baseline_validation_loss": baseline_validation_loss,
        "client_sizes": client_sizes,
        "client_training_losses": client_training_losses,
        "candidate_validation_losses": candidate_validation_losses,
        "similarity_scores": similarity_scores,
        "consistency_scores": consistency_scores,
        "historical_scores_before_update": previous_historical_scores,
        "combined_trust_scores": combined_trust_scores,
        "historical_scores_after_update": updated_historical_scores,
        "aggregation_weights": aggregation_weights
    }

    summary_path = (
        OUTPUT_DIR / "round_01_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------------
    # Final round report
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ROUND 1 COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("Clients trained:", len(client_ids))
    print("Accepted clients:", len(accepted_clients))
    print("Excluded clients:", len(rejected_clients))

    print("Checkpoint:", checkpoint_path)
    print("Summary:", summary_path)

    print("\nNo test dataset was used for trust calculation or screening.")
    print("The independent test dataset remains untouched.")

    del validation_X, validation_y, validation_loader
    gc.collect()


# ============================================================
# 8. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_one_round()
