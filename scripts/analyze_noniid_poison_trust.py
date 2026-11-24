
import pandas as pd
from pathlib import Path

RESULT_DIR = (
    Path("results")
    / "scenario_I_noniid_poison_trust_calibration"
)

FILE = (
    RESULT_DIR
    / "poisoned_noniid_trfa_trust_history.csv"
)

THRESHOLD = 0.7422773619

data = pd.read_csv(FILE)

print("\nNON-IID POISONED TRUST ANALYSIS")
print("=" * 85)

required_columns = [
    "Round",
    "Client",
    "Malicious",
    "Update_Similarity",
    "Training_Consistency",
    "Historical_Trust_Before",
    "Combined_Trust"
]

missing = [
    column for column in required_columns
    if column not in data.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

malicious = data[data["Malicious"] == True]
honest = data[data["Malicious"] == False]

if malicious.empty or honest.empty:
    raise ValueError(
        "Malicious or honest client records are missing."
    )

print("Clients:", data["Client"].nunique())
print("Communication rounds:", data["Round"].nunique())
print("Candidate threshold:", THRESHOLD)

# --------------------------------------------------------
# 1. Malicious client trajectory
# --------------------------------------------------------

print("\nCLIENT 10: ALL COMMUNICATION ROUNDS")
print("-" * 85)

for _, row in malicious.sort_values("Round").iterrows():

    print(
        f"Round {int(row['Round']):02d} | "
        f"S={row['Update_Similarity']:.6f} | "
        f"C={row['Training_Consistency']:.6f} | "
        f"H={row['Historical_Trust_Before']:.6f} | "
        f"T={row['Combined_Trust']:.6f}"
    )

# --------------------------------------------------------
# 2. Round 1, 5 and 10 comparisons
# --------------------------------------------------------

for round_number in [1, 5, 10]:

    group = data[data["Round"] == round_number].copy()

    print(
        f"\nALL CLIENTS: ROUND {round_number}"
    )

    print("-" * 85)

    columns = [
        "Client",
        "Malicious",
        "Update_Similarity",
        "Training_Consistency",
        "Historical_Trust_Before",
        "Combined_Trust"
    ]

    print(
        group[columns].sort_values("Client").to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

# --------------------------------------------------------
# 3. Retrospective detection analysis
# --------------------------------------------------------

detected = (
    malicious["Combined_Trust"] < THRESHOLD
)

false_excluded = (
    honest["Combined_Trust"] < THRESHOLD
)

print("\nRETROSPECTIVE THRESHOLD ANALYSIS")
print("-" * 85)

print(
    "Malicious client-rounds below threshold:",
    f"{int(detected.sum())}/{len(malicious)}"
)

print(
    "Honest client-rounds below threshold:",
    f"{int(false_excluded.sum())}/{len(honest)}"
)

print(
    "Malicious detection proportion:",
    f"{detected.mean() * 100:.2f}%"
)

print(
    "Honest false-flag proportion:",
    f"{false_excluded.mean() * 100:.2f}%"
)

# --------------------------------------------------------
# 4. Trust separation by round
# --------------------------------------------------------

print("\nROUND-BY-ROUND TRUST SEPARATION")
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

    minimum_honest = honest_scores.min()
    maximum_honest = honest_scores.max()

    print(
        f"Round {int(round_number):02d} | "
        f"Malicious={attacker_score:.6f} | "
        f"Honest min={minimum_honest:.6f} | "
        f"Honest max={maximum_honest:.6f} | "
        f"Attacker below threshold={attacker_score < THRESHOLD}"
    )

# --------------------------------------------------------
# 5. Save a retrospective summary
# --------------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Malicious client-rounds",
        "Honest client-rounds",
        "Malicious trust mean",
        "Honest trust mean",
        "Maximum malicious trust",
        "Minimum honest trust",
        "Malicious client-rounds below threshold",
        "Honest client-rounds below threshold"
    ],
    "Value": [
        len(malicious),
        len(honest),
        malicious["Combined_Trust"].mean(),
        honest["Combined_Trust"].mean(),
        malicious["Combined_Trust"].max(),
        honest["Combined_Trust"].min(),
        int(detected.sum()),
        int(false_excluded.sum())
    ]
})

output_path = RESULT_DIR / "noniid_poison_trust_analysis.csv"

summary.to_csv(output_path, index=False)

print("\nAnalysis saved:", output_path)

print("\nIMPORTANT:")
print(
    "These are retrospective threshold results from an "
    "experiment in which all clients participated. "
    "They are not observed results from screened aggregation."
)

print("\nANALYSIS COMPLETED")
