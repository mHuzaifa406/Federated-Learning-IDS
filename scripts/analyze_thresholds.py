
import pandas as pd
from pathlib import Path

# Load the existing TRFA trust history.
file_path = (
    Path("results")
    / "scenario_C_poisoned_trfa"
    / "trfa_client_trust_history.csv"
)

data = pd.read_csv(file_path)

print("\nTHRESHOLD SENSITIVITY ANALYSIS")
print("-" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Malicious detected':<22}"
    f"{'Honest false excluded':<24}"
    f"{'Detection rate':<15}"
)

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:

    malicious = data[data["Malicious"] == True]
    honest = data[data["Malicious"] == False]

    detected = int(
        (malicious["Combined_Trust"] < threshold).sum()
    )

    false_excluded = int(
        (honest["Combined_Trust"] < threshold).sum()
    )

    detection_rate = detected / len(malicious) * 100

    print(
        f"{threshold:<12.2f}"
        f"{detected}/10{'':<17}"
        f"{false_excluded}/90{'':<18}"
        f"{detection_rate:.2f}%"
    )

print("\nROUND-BY-ROUND TRUST SEPARATION")
print("-" * 75)

for round_number, group in data.groupby("Round"):

    malicious_score = group.loc[
        group["Malicious"] == True,
        "Combined_Trust"
    ].iloc[0]

    honest_scores = group.loc[
        group["Malicious"] == False,
        "Combined_Trust"
    ]

    print(
        f"Round {round_number:02d} | "
        f"Malicious: {malicious_score:.6f} | "
        f"Minimum honest: {honest_scores.min():.6f} | "
        f"Maximum honest: {honest_scores.max():.6f}"
    )

print("\nANALYSIS COMPLETED")
