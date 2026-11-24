
import json
import numpy as np
import pandas as pd

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "federated_noniid_alpha05_seed43"
    / "metadata.json"
)

TRUST_PATH = (
    PROJECT_ROOT
    / "results"
    / "scenario_H_noniid_trust_calibration"
    / "clean_noniid_trfa_trust_history.csv"
)


# ============================================================
# JENSEN-SHANNON DIVERGENCE
# ============================================================

def calculate_js_divergence(p, q):

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / p.sum()
    q = q / q.sum()

    midpoint = 0.5 * (p + q)

    def kl_divergence(a, b):

        mask = a > 0

        return np.sum(
            a[mask] * np.log2(
                a[mask] / b[mask]
            )
        )

    divergence = (
        0.5 * kl_divergence(p, midpoint)
        + 0.5 * kl_divergence(q, midpoint)
    )

    return float(divergence)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("\nNON-IID DISTRIBUTION VS TRUST ANALYSIS")
    print("=" * 75)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    trust_data = pd.read_csv(TRUST_PATH)

    global_counts = metadata["original_class_counts"]

    class_ids = sorted(
        global_counts.keys(),
        key=int
    )

    global_distribution = np.array(
        [
            global_counts[class_id]
            for class_id in class_ids
        ],
        dtype=np.float64
    )

    global_distribution /= global_distribution.sum()

    # --------------------------------------------------------
    # Calculate average trust signals per client
    # --------------------------------------------------------

    client_trust = (
        trust_data.groupby("Client")
        .agg(
            Mean_Similarity=("Update_Similarity", "mean"),
            Mean_Consistency=("Training_Consistency", "mean"),
            Mean_Historical_Trust=(
                "Historical_Trust_Before",
                "mean"
            ),
            Mean_Combined_Trust=("Combined_Trust", "mean"),
            Mean_Candidate_Validation_Loss=(
                "Candidate_Validation_Loss",
                "mean"
            )
        )
    )

    results = []

    # --------------------------------------------------------
    # Compare each client distribution with global distribution
    # --------------------------------------------------------

    for client_name, counts in metadata[
        "client_class_counts"
    ].items():

        client_counts = np.array(
            [
                counts.get(class_id, 0)
                for class_id in class_ids
            ],
            dtype=np.float64
        )

        client_distribution = (
            client_counts / client_counts.sum()
        )

        js_divergence = calculate_js_divergence(
            client_distribution,
            global_distribution
        )

        client_number = int(
            client_name.split("_")[1]
        )

        trust_row = client_trust.loc[client_number]

        results.append({
            "Client": client_number,
            "Training_Records": metadata[
                "client_sizes"
            ][client_name],
            "JS_Divergence": js_divergence,
            "Mean_Similarity": trust_row["Mean_Similarity"],
            "Mean_Consistency": trust_row["Mean_Consistency"],
            "Mean_Historical_Trust": (
                trust_row["Mean_Historical_Trust"]
            ),
            "Mean_Combined_Trust": (
                trust_row["Mean_Combined_Trust"]
            ),
            "Mean_Candidate_Validation_Loss": (
                trust_row["Mean_Candidate_Validation_Loss"]
            )
        })

    result_df = pd.DataFrame(results).sort_values(
        "JS_Divergence",
        ascending=False
    )

    # --------------------------------------------------------
    # Print comparison table
    # --------------------------------------------------------

    print("\nCLIENT DISTRIBUTION AND TRUST COMPARISON")
    print("-" * 110)

    print(
        result_df.to_string(
            index=False,
            formatters={
                "JS_Divergence": "{:.6f}".format,
                "Mean_Similarity": "{:.6f}".format,
                "Mean_Consistency": "{:.6f}".format,
                "Mean_Historical_Trust": "{:.6f}".format,
                "Mean_Combined_Trust": "{:.6f}".format,
                "Mean_Candidate_Validation_Loss": "{:.6f}".format
            }
        )
    )

    # --------------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------------

    print("\nCORRELATION ANALYSIS")
    print("-" * 75)

    variables = [
        "JS_Divergence",
        "Training_Records",
        "Mean_Similarity",
        "Mean_Consistency",
        "Mean_Combined_Trust",
        "Mean_Candidate_Validation_Loss"
    ]

    print(
        result_df[variables]
        .corr(method="spearman")
        .round(4)
        .to_string()
    )

    print("\nINTERPRETATION GUIDE")
    print("-" * 75)

    print(
        "Positive JS/consistency correlation: "
        "greater divergence is associated with higher consistency."
    )

    print(
        "Negative JS/consistency correlation: "
        "greater divergence is associated with lower consistency."
    )

    print(
        "Negative JS/combined-trust correlation: "
        "greater divergence is associated with lower trust."
    )

    print(
        "These are exploratory correlations across only "
        "10 clients, not causal evidence."
    )

    # --------------------------------------------------------
    # Save diagnostic results
    # --------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "results"
        / "scenario_H_noniid_trust_calibration"
        / "distribution_vs_trust_analysis.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    print("\nAnalysis saved:", output_path)
    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
