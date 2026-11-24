
"""
Create a separate non-IID federated training partition.

Method:
Class-wise Dirichlet partitioning.

Source:
Existing 10-client IID training dataset.

Important:
- Does not modify existing datasets.
- Does not use validation or test data.
- Every source training record is assigned exactly once.
"""

from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "federated_v2"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "federated_noniid_alpha05_seed43"
)

NUM_CLIENTS = 10
DIRICHLET_ALPHA = 0.5
RANDOM_SEED = 43

EXPECTED_FEATURES = 78
EXPECTED_CLIENTS = 10


# ============================================================
# LOAD ORIGINAL TRAINING CLIENTS
# ============================================================

def load_original_training_data():

    print("\nLoading original client training records...")

    client_directories = [
        SOURCE_ROOT / f"client_{i:02d}"
        for i in range(1, NUM_CLIENTS + 1)
    ]

    frames = []

    for client_dir in client_directories:

        if not client_dir.exists():
            raise FileNotFoundError(
                f"Missing source client directory: {client_dir}"
            )

        parquet_files = sorted(
            client_dir.glob("*.parquet")
        )

        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found in {client_dir}"
            )

        for parquet_file in parquet_files:

            print("Reading:", parquet_file.name)

            dataframe = pd.read_parquet(parquet_file)

            if "Label_id" not in dataframe.columns:
                raise ValueError(
                    f"Label_id missing from {parquet_file}"
                )

            frames.append(dataframe)

    print("\nCombining source training records...")

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    del frames
    gc.collect()

    feature_columns = [
        column for column in combined.columns
        if column not in {
            "Label_id",
            "Label",
            "Label_7class"
        }
    ]

    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"found {len(feature_columns)}"
        )

    labels = combined["Label_id"].to_numpy(
        dtype=np.int64
    )

    if not np.isfinite(
        combined[feature_columns].to_numpy()
    ).all():
        raise ValueError(
            "Non-finite feature values found."
        )

    if labels.min() < 0 or labels.max() >= 7:
        raise ValueError("Unexpected label IDs found.")

    print("Total training records:", f"{len(combined):,}")
    print("Number of features:", len(feature_columns))
    print("Number of classes:", len(np.unique(labels)))

    return combined, labels, feature_columns


# ============================================================
# DIRICHLET PARTITIONING
# ============================================================

def create_dirichlet_assignments(labels):

    print("\nCreating non-IID client assignments...")

    rng = np.random.default_rng(RANDOM_SEED)

    client_indices = [
        [] for _ in range(NUM_CLIENTS)
    ]

    original_class_counts = {}

    for class_id in np.unique(labels):

        class_indices = np.flatnonzero(
            labels == class_id
        )

        rng.shuffle(class_indices)

        # Generate different client proportions for
        # this particular class.
        proportions = rng.dirichlet(
            np.full(
                NUM_CLIENTS,
                DIRICHLET_ALPHA
            )
        )

        # Allocate every record exactly once.
        counts = rng.multinomial(
            len(class_indices),
            proportions
        )

        original_class_counts[str(int(class_id))] = int(
            len(class_indices)
        )

        offset = 0

        for client_index, count in enumerate(counts):

            next_offset = offset + int(count)

            selected = class_indices[
                offset:next_offset
            ]

            client_indices[client_index].extend(
                selected.tolist()
            )

            offset = next_offset

        if offset != len(class_indices):
            raise RuntimeError(
                f"Records were not fully allocated for class {class_id}"
            )

    # Shuffle each client's assigned record order.
    for indices in client_indices:
        rng.shuffle(indices)

    # Verify that all source records were assigned exactly once.
    all_indices = np.concatenate([
        np.asarray(indices, dtype=np.int64)
        for indices in client_indices
    ])

    if len(all_indices) != len(labels):
        raise RuntimeError(
            "Total assigned records do not match source records."
        )

    if len(np.unique(all_indices)) != len(labels):
        raise RuntimeError(
            "Duplicate or missing record assignments detected."
        )

    print("All records assigned exactly once.")

    return client_indices, original_class_counts


# ============================================================
# SAVE NON-IID CLIENT DATASETS
# ============================================================

def save_noniid_clients(
    dataframe,
    client_indices,
    feature_columns,
    original_class_counts
):

    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_ROOT}\n"
            "To avoid overwriting data, rename or remove the "
            "previous experiment directory before rerunning."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False
    )

    client_sizes = {}
    client_class_counts = {}

    total_saved = 0

    print("\nSaving non-IID client datasets...")

    for client_index, indices in enumerate(
        client_indices,
        start=1
    ):

        client_name = f"client_{client_index:02d}"

        client_dir = OUTPUT_ROOT / client_name
        client_dir.mkdir(parents=True)

        selected_indices = np.asarray(
            indices,
            dtype=np.int64
        )

        if len(selected_indices) == 0:
            raise RuntimeError(
                f"{client_name} received no records."
            )

        client_dataframe = (
            dataframe.iloc[selected_indices]
            .reset_index(drop=True)
        )

        # Preserve the same feature and label schema.
        expected_columns = list(dataframe.columns)

        if list(client_dataframe.columns) != expected_columns:
            raise ValueError(
                f"Schema mismatch for {client_name}"
            )

        output_file = (
            client_dir / f"{client_name}.parquet"
        )

        client_dataframe.to_parquet(
            output_file,
            index=False
        )

        counts = (
            client_dataframe["Label_id"]
            .value_counts()
            .sort_index()
        )

        class_distribution = {
            str(int(class_id)): int(count)
            for class_id, count in counts.items()
        }

        client_sizes[client_name] = len(
            client_dataframe
        )

        client_class_counts[client_name] = (
            class_distribution
        )

        total_saved += len(client_dataframe)

        print(
            f"{client_name}: "
            f"{len(client_dataframe):,} records | "
            f"{len(class_distribution)} represented classes"
        )

        del client_dataframe
        gc.collect()

    if total_saved != len(dataframe):
        raise RuntimeError(
            "Saved record count does not match source record count."
        )

    # Verify class counts are preserved.
    reconstructed_counts = {}

    for class_id in original_class_counts:

        reconstructed_counts[class_id] = sum(
            client_class_counts[client][class_id]
            if class_id in client_class_counts[client]
            else 0
            for client in client_class_counts
        )

    if reconstructed_counts != original_class_counts:
        raise RuntimeError(
            "Class distributions do not account for all source records."
        )

    # Save experiment metadata.
    metadata = {
        "dataset": "CIC-IDS-2017",
        "partition_method": "Class-wise Dirichlet distribution",
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "random_seed": RANDOM_SEED,
        "number_of_clients": NUM_CLIENTS,
        "number_of_features": len(feature_columns),
        "total_training_records": int(total_saved),
        "original_class_counts": original_class_counts,
        "client_sizes": client_sizes,
        "client_class_counts": client_class_counts,
        "validation_dataset_modified": False,
        "test_dataset_modified": False,
        "source": "data/processed/federated_v2"
    }

    metadata_path = OUTPUT_ROOT / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nPARTITIONING SUMMARY")
    print("=" * 60)

    print("Partition method: Class-wise Dirichlet")
    print("Dirichlet alpha:", DIRICHLET_ALPHA)
    print("Random seed:", RANDOM_SEED)
    print("Clients:", NUM_CLIENTS)
    print("Total saved records:", f"{total_saved:,}")

    print("\nMetadata saved:", metadata_path)

    print("\nNON-IID PARTITION CREATED SUCCESSFULLY")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("NON-IID FEDERATED DATASET CREATION")
    print("=" * 65)

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Original training dataset not found: {SOURCE_ROOT}"
        )

    dataframe, labels, feature_columns = (
        load_original_training_data()
    )

    client_indices, original_class_counts = (
        create_dirichlet_assignments(labels)
    )

    save_noniid_clients(
        dataframe,
        client_indices,
        feature_columns,
        original_class_counts
    )

    del dataframe, labels, client_indices
    gc.collect()


if __name__ == "__main__":
    main()
