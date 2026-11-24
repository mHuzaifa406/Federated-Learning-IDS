
"""
TRFA Dataset Validation
-----------------------
Checks:
1. Client, validation, and test file availability.
2. Expected Parquet row counts.
3. Feature names and feature ordering.
4. Numeric feature and label types.
5. Missing and invalid class labels.
6. Validation and test class distributions.

Only the Label_id column is read in batches for the label audit.
The full dataset is never loaded into memory.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.json"

LABEL_COLUMN = "Label_id"
EXTRA_LABEL_COLUMNS = {"Label", "Label_7class"}
BATCH_SIZE = 250_000


def load_json(path):
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(relative_path):
    """Resolve a configured path relative to the project root."""
    return PROJECT_ROOT / Path(relative_path)


# ---------------------------------------------------------
# 2. PARQUET FILE AUDIT
# ---------------------------------------------------------

def audit_parquet_file(file_path, reference_features, valid_labels):
    """
    Check one Parquet file's schema and labels.

    Returns:
        row_count, feature_names, class_distribution, errors
    """

    errors = []

    if not file_path.exists():
        return 0, [], Counter(), [f"Missing file: {file_path}"]

    parquet_file = pq.ParquetFile(file_path)
    row_count = parquet_file.metadata.num_rows
    schema = parquet_file.schema_arrow
    column_names = schema.names

    # Check required label column.
    if LABEL_COLUMN not in column_names:
        return row_count, [], Counter(), [
            f"{file_path.name}: Missing {LABEL_COLUMN}"
        ]

    # Remove label-related columns to identify model features.
    feature_names = [
        name for name in column_names
        if name != LABEL_COLUMN and name not in EXTRA_LABEL_COLUMNS
    ]

    if len(feature_names) != 78:
        errors.append(
            f"{file_path.name}: Expected 78 features, "
            f"found {len(feature_names)}"
        )

    # Check feature order against the first client file.
    if reference_features is not None and feature_names != reference_features:
        errors.append(
            f"{file_path.name}: Feature names or feature order differ"
        )

    # Check that all model features are numeric.
    for feature in feature_names:
        feature_type = schema.field(feature).type

        if not (
            pa.types.is_integer(feature_type)
            or pa.types.is_floating(feature_type)
        ):
            errors.append(
                f"{file_path.name}: Non-numeric feature "
                f"'{feature}' ({feature_type})"
            )

    # Check that Label_id is an integer column.
    label_type = schema.field(LABEL_COLUMN).type

    if not pa.types.is_integer(label_type):
        errors.append(
            f"{file_path.name}: Label_id must be an integer, "
            f"found {label_type}"
        )
        return row_count, feature_names, Counter(), errors

    # Read labels in batches to keep memory usage low.
    class_counts = Counter()

    for batch in parquet_file.iter_batches(
        columns=[LABEL_COLUMN],
        batch_size=BATCH_SIZE
    ):
        label_array = batch.column(0)

        if label_array.null_count > 0:
            errors.append(
                f"{file_path.name}: Found null class labels"
            )
            continue

        labels = label_array.to_numpy(zero_copy_only=False)

        if not np.isin(labels, list(valid_labels)).all():
            invalid_values = np.unique(
                labels[~np.isin(labels, list(valid_labels))]
            )
            errors.append(
                f"{file_path.name}: Invalid class IDs "
                f"{invalid_values.tolist()}"
            )
            continue

        values, counts = np.unique(labels, return_counts=True)

        for value, count in zip(values, counts):
            class_counts[int(value)] += int(count)

    return row_count, feature_names, class_counts, errors


# ---------------------------------------------------------
# 3. MAIN VALIDATION
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("TRFA DATASET VALIDATION")
    print("=" * 65)

    config = load_json(CONFIG_PATH)

    data_config = config["data"]
    paths = config["paths"]
    expected = data_config["expected_records"]

    valid_labels = {
        int(class_id)
        for class_id in data_config["classes"]
    }

    # Resolve dataset directories.
    federated_dir = resolve_path(paths["federated_data_dir"])
    validation_file = resolve_path(paths["validation_file"])
    test_dir = resolve_path(paths["test_dir"])

    split_metadata_path = resolve_path(paths["split_metadata"])

    federated_metadata_path = federated_dir / "metadata.json"

    split_metadata = load_json(split_metadata_path)
    federated_metadata = load_json(federated_metadata_path)

    errors = []

    # Discover files.
    client_files = sorted(
        federated_dir.glob("client_*/client_data.parquet")
    )

    test_files = sorted(test_dir.glob("*.parquet"))

    print("\n[1] FILE AVAILABILITY")

    print("Client files:", len(client_files))
    print("Test files:", len(test_files))
    print("Validation exists:", validation_file.exists())

    if len(client_files) != data_config["number_of_clients"]:
        errors.append("Incorrect number of client files")

    if len(test_files) != data_config["test_file_count"]:
        errors.append("Incorrect number of test files")

    if not validation_file.exists():
        errors.append("Validation file is missing")

    # Audit all client files.
    print("\n[2] CLIENT TRAINING DATA")

    total_client_rows = 0
    total_client_labels = Counter()
    reference_features = None

    for file_path in client_files:

        rows, features, labels, file_errors = audit_parquet_file(
            file_path,
            reference_features,
            valid_labels
        )

        if reference_features is None and features:
            reference_features = features

        total_client_rows += rows
        total_client_labels.update(labels)
        errors.extend(file_errors)

        print(f"{file_path.parent.name}: {rows:,} rows")

    print("Total client rows:", f"{total_client_rows:,}")

    if total_client_rows != expected["client_training_total"]:
        errors.append("Client training total does not match configuration")

    # Audit validation data.
    print("\n[3] VALIDATION DATA")

    validation_rows = 0
    validation_labels = Counter()

    if validation_file.exists():

        (
            validation_rows,
            validation_features,
            validation_labels,
            file_errors
        ) = audit_parquet_file(
            validation_file,
            reference_features,
            valid_labels
        )

        errors.extend(file_errors)

    print("Validation rows:", f"{validation_rows:,}")

    if validation_rows != expected["validation"]:
        errors.append("Validation row count does not match configuration")

    # Audit test files.
    print("\n[4] INDEPENDENT TEST DATA")

    total_test_rows = 0
    total_test_labels = Counter()

    for file_path in test_files:

        rows, features, labels, file_errors = audit_parquet_file(
            file_path,
            reference_features,
            valid_labels
        )

        total_test_rows += rows
        total_test_labels.update(labels)
        errors.extend(file_errors)

        print(f"{file_path.name}: {rows:,} rows")

    print("Total test rows:", f"{total_test_rows:,}")

    if total_test_rows != expected["test_total"]:
        errors.append("Test row count does not match configuration")

    # Check that the complete record count reconciles.
    complete_total = (
        total_client_rows + validation_rows + total_test_rows
    )

    print("\n[5] COMPLETE DATASET ACCOUNTING")

    print("Client training:", f"{total_client_rows:,}")
    print("Validation:", f"{validation_rows:,}")
    print("Independent test:", f"{total_test_rows:,}")
    print("Combined total:", f"{complete_total:,}")

    if complete_total != expected["complete_dataset_total"]:
        errors.append("Complete dataset total does not match configuration")

    # Check validation class distribution against metadata.
    print("\n[6] CLASS DISTRIBUTION CHECKS")

    expected_validation_labels = {
        int(k): int(v)
        for k, v in federated_metadata[
            "validation_class_distribution"
        ].items()
    }

    actual_validation_labels = dict(validation_labels)

    print("Validation labels:", actual_validation_labels)

    if actual_validation_labels != expected_validation_labels:
        errors.append(
            "Validation class distribution differs from metadata"
        )

    # Compare test class distribution with split metadata.
    class_map = {
        int(k): v
        for k, v in data_config["classes"].items()
    }

    actual_test_by_name = {
        class_map[class_id]: count
        for class_id, count in total_test_labels.items()
    }

    expected_test_by_name = {
        name: int(count)
        for name, count in split_metadata[
            "class_testing_counts"
        ].items()
    }

    print("Test labels by class:", actual_test_by_name)

    if actual_test_by_name != expected_test_by_name:
        errors.append(
            "Test class distribution differs from split metadata"
        )

    # Final result.
    print("\n" + "=" * 65)

    if errors:
        print("DATASET VALIDATION FAILED")
        print("=" * 65)

        for number, error in enumerate(errors, start=1):
            print(f"{number}. {error}")

        sys.exit(1)

    print("DATASET VALIDATION PASSED")
    print("All expected files, row counts, feature schemas,")
    print("and class distributions have been verified.")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nVALIDATION ERROR:", str(exc))
        sys.exit(1)
