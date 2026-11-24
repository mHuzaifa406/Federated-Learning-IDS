
import pandas as pd
from pathlib import Path

RESULT_DIR = (
    Path("results")
    / "scenario_H_noniid_trust_calibration"
)

FILE = RESULT_DIR / "clean_noniid_trfa_trust_history.csv"

THRESHOLD = 0.7422773619

data = pd.read_csv(FILE)

signals = [
    "Update_Similarity",
    "Training_Consistency",
    "Historical_Trust_Before",
    "Combined_Trust"
]

print("\nNON-IID TRUST SIGNAL CONTRIBUTION ANALYSIS")
print("=" * 75)

# ------------------------------------------------------------
# 1. Overall signal statistics
# ------------------------------------------------------------

print("\nOVERALL SIGNAL STATISTICS")
print("-" * 75)

print(
    data[signals].agg(
        ["mean", "std", "min", "max"]
    ).round(6).to_string()
)

# ------------------------------------------------------------
# 2. Per-client average signal scores
# ------------------------------------------------------------

print("\nAVERAGE TRUST SIGNALS BY CLIENT")
print("-" * 75)

client_summary = (
    data.groupby("Client")[signals]
    .mean()
    .round(6)
)

print(client_summary.to_string())

# ------------------------------------------------------------
# 3. Which signals are weakest in each round?
# ------------------------------------------------------------

print("\nROUND-BY-ROUND SIGNAL MEANS")
print("-" * 75)

round_summary = (
    data.groupby("Round")[
        signals
    ].mean().round(6)
)

print(round_summary.to_string())

# ------------------------------------------------------------
# 4. Retrospective component comparisons
# ------------------------------------------------------------

print("\nCOMPONENT-ONLY RETROSPECTIVE COMPARISON")
print("-" * 75)

for signal in [
    "Update_Similarity",
    "Training_Consistency",
    "Historical_Trust_Before",
    "Combined_Trust"
]:

    below = int(
        (data[signal] < THRESHOLD).sum()
    )

    print(
        f"{signal:<28} "
        f"Below threshold: {below}/100 "
        f"({below:.1f}%)"
    )

# ------------------------------------------------------------
# 5. Identify weakest signal for each client
# ------------------------------------------------------------

print("\nWEAKEST AVERAGE SIGNAL BY CLIENT")
print("-" * 75)

component_columns = [
    "Update_Similarity",
    "Training_Consistency",
    "Historical_Trust_Before"
]

for client_id, group in data.groupby("Client"):

    means = group[component_columns].mean()

    weakest_signal = means.idxmin()
    weakest_score = means.min()

    print(
        f"Client {int(client_id):02d} | "
        f"Weakest signal: {weakest_signal} | "
        f"Mean score: {weakest_score:.6f}"
    )

print("\nANALYSIS COMPLETED")
