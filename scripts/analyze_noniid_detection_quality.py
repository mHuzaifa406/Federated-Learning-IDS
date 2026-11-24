
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

RESULT_DIR = (
    Path("results")
    / "scenario_I_noniid_poison_trust_calibration"
)

FILE = (
    RESULT_DIR
    / "poisoned_noniid_trfa_trust_history.csv"
)

data = pd.read_csv(FILE)

print("\nNON-IID DETECTION QUALITY ANALYSIS")
print("=" * 85)

# Malicious clients have lower trust scores.
# Negating trust makes lower trust correspond to a higher
# anomaly score for ROC-AUC and average precision.
labels = data["Malicious"].astype(int)

anomaly_scores = -data["Combined_Trust"]

# --------------------------------------------------------
# 1. Overall exploratory separation
# --------------------------------------------------------

print("\nOVERALL EXPLORATORY SEPARATION")
print("-" * 85)

auc = roc_auc_score(labels, anomaly_scores)

average_precision = average_precision_score(
    labels,
    anomaly_scores
)

print(f"ROC-AUC: {auc:.6f}")
print(f"Average precision: {average_precision:.6f}")

# --------------------------------------------------------
# 2. Per-round separation
# --------------------------------------------------------

print("\nPER-ROUND ROC-AUC AND ATTACKER RANK")
print("-" * 85)

for round_number, group in data.groupby("Round"):

    round_labels = group["Malicious"].astype(int)

    round_scores = -group["Combined_Trust"]

    round_auc = roc_auc_score(
        round_labels,
        round_scores
    )

    attacker = group[group["Malicious"] == True].iloc[0]

    # Rank 1 means lowest trust and highest suspicion.
    sorted_clients = group.sort_values(
        "Combined_Trust",
        ascending=True
    )

    attacker_rank = int(
        sorted_clients["Client"].tolist().index(
            int(attacker["Client"])
        ) + 1
    )

    print(
        f"Round {int(round_number):02d} | "
        f"ROC-AUC={round_auc:.4f} | "
        f"Attacker trust={attacker['Combined_Trust']:.6f} | "
        f"Attacker suspiciousness rank={attacker_rank}/10"
    )

# --------------------------------------------------------
# 3. Threshold comparison
# --------------------------------------------------------

print("\nTHRESHOLD COMPARISON")
print("-" * 85)

print(
    f"{'Threshold':<14}"
    f"{'Detection':<18}"
    f"{'Honest false flags':<22}"
    f"{'False-positive rate':<20}"
)

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.7422773619
]

malicious = data[data["Malicious"] == True]
honest = data[data["Malicious"] == False]

for threshold in thresholds:

    detected = int(
        (malicious["Combined_Trust"] < threshold).sum()
    )

    false_flags = int(
        (honest["Combined_Trust"] < threshold).sum()
    )

    detection_rate = detected / len(malicious)

    false_positive_rate = (
        false_flags / len(honest)
    )

    print(
        f"{threshold:<14.6f}"
        f"{detection_rate * 100:<18.2f}%"
        f"{false_flags:<22}"
        f"{false_positive_rate * 100:.2f}%"
    )

# --------------------------------------------------------
# 4. Per-round trust gap
# --------------------------------------------------------

print("\nPER-ROUND TRUST GAP")
print("-" * 85)

for round_number, group in data.groupby("Round"):

    attacker_score = group.loc[
        group["Malicious"] == True,
        "Combined_Trust"
    ].iloc[0]

    honest_scores = group.loc[
        group["Malicious"] == False,
        "Combined_Trust"
    ]

    gap = (
        honest_scores.median()
        - attacker_score
    )

    print(
        f"Round {int(round_number):02d} | "
        f"Honest median={honest_scores.median():.6f} | "
        f"Attacker={attacker_score:.6f} | "
        f"Gap={gap:+.6f}"
    )

# --------------------------------------------------------
# 5. Save results
# --------------------------------------------------------

output_path = (
    RESULT_DIR
    / "noniid_detection_quality_analysis.txt"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"Overall ROC-AUC: {auc:.6f}\n"
    )

    file.write(
        f"Average precision: {average_precision:.6f}\n"
    )

    file.write(
        "Note: Client-round observations are repeated "
        "measurements, not independent samples.\n"
    )

print("\nAnalysis saved:", output_path)

print("\nIMPORTANT:")
print(
    "Overall ROC-AUC and average precision are exploratory "
    "because the same clients are observed repeatedly "
    "across communication rounds."
)

print("\nANALYSIS COMPLETED")
