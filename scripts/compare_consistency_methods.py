
import json
import math
import numpy as np
import pandas as pd

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "results"
    / "scenario_H_noniid_trust_calibration"
    / "per_class_candidate_diagnostic.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "federated_noniid_alpha05_seed43"
    / "metadata.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "scenario_H_noniid_trust_calibration"
    / "consistency_method_comparison.csv"
)

MIN_VALIDATION_SUPPORT = 500


def sigmoid_score(improvement, scale):

    if scale <= 0:
        return 0.5

    z = np.clip(improvement / scale, -60, 60)

    return float(1.0 / (1.0 + math.exp(-z)))


def main():

    print("\nCONSISTENCY METHOD COMPARISON")
    print("=" * 85)

    diagnostic = pd.read_csv(DIAGNOSTIC_FILE)

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    # Validation class counts.
    validation_counts = (
        diagnostic.groupby("Class_ID")["Validation_Count"]
        .first()
    )

    supported_classes = validation_counts[
        validation_counts >= MIN_VALIDATION_SUPPORT
    ].index.tolist()

    print("\nClasses with sufficient validation support:")
    print(supported_classes)

    print(
        "Minimum validation support:",
        MIN_VALIDATION_SUPPORT
    )

    results = []

    # --------------------------------------------------------
    # Calculate three alternative improvement measures
    # --------------------------------------------------------

    for client_id, group in diagnostic.groupby("Client"):

        group = group.sort_values("Class_ID")

        improvements = group[
            "Validation_Loss_Improvement"
        ].to_numpy(dtype=float)

        counts = group[
            "Validation_Count"
        ].to_numpy(dtype=float)

        client_shares = group[
            "Client_Class_Share"
        ].to_numpy(dtype=float)

        # Method 1:
        # Overall validation-record-weighted improvement.
        global_improvement = float(
            np.average(
                improvements,
                weights=counts
            )
        )

        # Method 2:
        # Macro improvement over sufficiently represented
        # validation classes.
        supported_mask = (
            group["Validation_Count"].to_numpy()
            >= MIN_VALIDATION_SUPPORT
        )

        supported_improvement = float(
            np.mean(improvements[supported_mask])
        )

        # Method 3:
        # Weight validation-class improvements according
        # to the client's observed training class distribution.
        distribution_improvement = float(
            np.sum(improvements * client_shares)
            / np.sum(client_shares)
        )

        results.append({
            "Client": int(client_id),
            "Global_Improvement": global_improvement,
            "Supported_Class_Improvement": supported_improvement,
            "Distribution_Weighted_Improvement": distribution_improvement
        })

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Convert improvements into comparable 0-1 scores.
    # Use the median absolute improvement as scale,
    # matching the existing consistency function's
    # scaling approach.
    # --------------------------------------------------------

    methods = {
        "Global": "Global_Improvement",
        "Supported_Class": "Supported_Class_Improvement",
        "Distribution_Weighted": "Distribution_Weighted_Improvement"
    }

    for method_name, column in methods.items():

        values = result_df[column].to_numpy(dtype=float)

        nonzero = np.abs(values[np.abs(values) > 1e-12])

        scale = (
            float(np.median(nonzero))
            if len(nonzero) > 0
            else 1.0
        )

        score_column = method_name + "_Consistency"

        result_df[score_column] = [
            sigmoid_score(value, scale)
            for value in values
        ]

    # --------------------------------------------------------
    # Attach original 10-round consistency for context.
    # These are not same-round comparisons.
    # --------------------------------------------------------

    history_file = (
        PROJECT_ROOT
        / "results"
        / "scenario_H_noniid_trust_calibration"
        / "clean_noniid_trfa_trust_history.csv"
    )

    history = pd.read_csv(history_file)

    original_means = (
        history.groupby("Client")["Training_Consistency"]
        .mean()
        .rename("Original_10_Round_Mean_Consistency")
    )

    result_df = result_df.merge(
        original_means,
        on="Client",
        how="left"
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    output_columns = [
        "Client",
        "Global_Improvement",
        "Global_Consistency",
        "Supported_Class_Improvement",
        "Supported_Class_Consistency",
        "Distribution_Weighted_Improvement",
        "Distribution_Weighted_Consistency",
        "Original_10_Round_Mean_Consistency"
    ]

    print("\nCLIENT-BY-CLIENT COMPARISON")
    print("-" * 85)

    print(
        result_df[output_columns].round(6).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary of alternative consistency scores
    # --------------------------------------------------------

    print("\nCONSISTENCY SCORE SUMMARY")
    print("-" * 85)

    for column in [
        "Global_Consistency",
        "Supported_Class_Consistency",
        "Distribution_Weighted_Consistency"
    ]:

        print(
            f"{column}: "
            f"mean={result_df[column].mean():.6f} | "
            f"minimum={result_df[column].min():.6f} | "
            f"maximum={result_df[column].max():.6f}"
        )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nComparison saved:", OUTPUT_FILE)

    print("\nIMPORTANT:")
    print(
        "This is a one-round diagnostic comparison, "
        "not a validated replacement for TRFA."
    )

    print(
        "Distribution-weighted consistency uses client label "
        "distributions, which may be manipulated by malicious clients."
    )

    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
