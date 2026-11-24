
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "federated_noniid_alpha05_seed43"
)

METADATA_PATH = DATA_ROOT / "metadata.json"

NUM_CLIENTS = 10
EXPECTED_FEATURES = 78
EXPECTED_TOTAL = 2_149_187


def main():

    print("\nNON-IID PARTITION VERIFICATION")
    print("=" * 75)

    if not METADATA_PATH.exists():
        raise FileNotFoundError("Partition metadata is missing.")

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    client_sizes = {}
    actual_class_counts = {}
    all_rows = 0

    for client_number in range(1, NUM_CLIENTS + 1):

        client_name = f"client_{client_number:02d}"
        client_dir = DATA_ROOT / client_name

        parquet_files = sorted(client_dir.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found for {client_name}"
            )

        frames = [
            pd.read_parquet(file)
            for file in parquet_files
        ]

        data = pd.concat(
            frames,
            ignore_index=True
        )

        if "Label_id" not in data.columns:
            raise ValueError(
                f"Label_id missing in {client_name}"
            )

        feature_columns = [
            column for column in data.columns
            if column not in {
                "Label_id",
                "Label",
                "Label_7class"
            }
        ]

        if len(feature_columns) != EXPECTED_FEATURES:
            raise ValueError(
                f"Unexpected feature count in {client_name}"
            )

        if not data.columns.equals(frames[0].columns):
            raise ValueError(
                f"Schema mismatch in {client_name}"
            )

        if data.empty:
            raise ValueError(
                f"{client_name} has no training records."
            )

        counts = (
            data["Label_id"]
            .value_counts()
            .sort_index()
        )

        actual_class_counts[client_name] = {
            str(int(label)): int(count)
            for label, count in counts.items()
        }

        client_sizes[client_name] = len(data)
        all_rows += len(data)

        print(
            f"{client_name}: "
            f"{len(data):,} records | "
            f"{len(counts)} represented classes"
        )

        del data, frames

    # --------------------------------------------------------
    # Verify total row count
    # --------------------------------------------------------

    print("\nTOTAL RECORD VERIFICATION")

    print("Actual records:", f"{all_rows:,}")
    print("Expected records:", f"{EXPECTED_TOTAL:,}")

    if all_rows != EXPECTED_TOTAL:
        raise AssertionError(
            "Total records do not match the expected total."
        )

    # --------------------------------------------------------
    # Verify original class counts are preserved
    # --------------------------------------------------------

    reconstructed = {}

    for class_id in metadata["original_class_counts"]:

        reconstructed[class_id] = sum(
            actual_class_counts[client].get(class_id, 0)
            for client in actual_class_counts
        )

    if reconstructed != metadata["original_class_counts"]:
        raise AssertionError(
            "Original class counts are not preserved."
        )

    # --------------------------------------------------------
    # Compare actual distribution with metadata
    # --------------------------------------------------------

    if client_sizes != metadata["client_sizes"]:
        raise AssertionError(
            "Saved client sizes do not match metadata."
        )

    if actual_class_counts != metadata["client_class_counts"]:
        raise AssertionError(
            "Saved client class distributions do not match metadata."
        )

    print("\nCLASS DISTRIBUTION BY CLIENT")
    print("-" * 75)

    class_ids = sorted(
        metadata["original_class_counts"].keys(),
        key=int
    )

    for client_name in client_sizes:

        total = client_sizes[client_name]

        print(f"\n{client_name} ({total:,} records)")

        for class_id in class_ids:

            count = actual_class_counts[
                client_name
            ].get(class_id, 0)

            percentage = count / total * 100

            print(
                f"  Class {class_id}: "
                f"{count:>8,} ({percentage:>7.3f}%)"
            )

    # --------------------------------------------------------
    # Calculate class distribution variation
    # --------------------------------------------------------

    print("\nNON-IID DISTRIBUTION ANALYSIS")
    print("-" * 75)

    distribution_rows = []

    for client_name in client_sizes:

        total = client_sizes[client_name]

        distribution_rows.append([
            actual_class_counts[client_name].get(
                class_id, 0
            ) / total
            for class_id in class_ids
        ])

    distribution = pd.DataFrame(
        distribution_rows,
        index=list(client_sizes.keys()),
        columns=class_ids
    )

    for class_id in class_ids:

        values = distribution[class_id]

        print(
            f"Class {class_id}: "
            f"minimum={values.min():.4f} | "
            f"maximum={values.max():.4f} | "
            f"variation={values.max() - values.min():.4f}"
        )

    print("\nPartition method:", metadata["partition_method"])
    print("Dirichlet alpha:", metadata["dirichlet_alpha"])
    print("Random seed:", metadata["random_seed"])

    print("\nVALIDATION RESULTS")
    print("-" * 75)
    print("Total records verified:", f"{all_rows:,}")
    print("Ten client datasets verified: YES")
    print("Feature schema verified: YES")
    print("Original class totals preserved: YES")
    print("Metadata matches saved data: YES")

    print("\nNON-IID PARTITION VERIFICATION PASSED")


if __name__ == "__main__":
    main()
