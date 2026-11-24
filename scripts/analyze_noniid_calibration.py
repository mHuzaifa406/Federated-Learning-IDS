
import pandas as pd
from pathlib import Path

RESULT_DIR = (
    Path("results")
    / "scenario_H_noniid_trust_calibration"
)

FILE = RESULT_DIR / "clean_noniid_trfa_trust_history.csv"

CANDIDATE_THRESHOLD = 0.7422773619

data = pd.read_csv(FILE)

print("\nNON-IID HONEST CLIENT TRUST CALIBRATION")
print("=" * 75)

print("Clients:", data["Client"].nunique())
print("Communication rounds:", data["Round"].nunique())
print("Candidate threshold:", CANDIDATE_THRESHOLD)

# All clients in this experiment are honest.
print("\nOVERALL TRUST DISTRIBUTION")
print("-" * 75)

for column in [
    "Update_Similarity",
    "Training_Consistency",
    "Historical_Trust_Before",
    "Combined_Trust"
]:
    print(
        f"{column}: "
        f"min={data[column].min():.6f} | "
        f"mean={data[column].mean():.6f} | "
        f"max={data[column].max():.6f}"
    )

# Retrospectively assess the candidate threshold.
# This does not simulate retraining after client exclusion.
data["Below_Candidate_Threshold"] = (
    data["Combined_Trust"] < CANDIDATE_THRESHOLD
)

flagged = data[data["Below_Candidate_Threshold"]]

print("\nRETROSPECTIVE CANDIDATE THRESHOLD ANALYSIS")
print("-" * 75)

print(
    "Honest client-rounds below threshold:",
    len(flagged),
    "/",
    len(data)
)

print(
    "Retrospective honest exclusion rate:",
    f"{len(flagged) / len(data) * 100:.2f}%"
)

print("\nROUND-BY-ROUND TRUST DISTRIBUTION")
print("-" * 75)

for round_number, group in data.groupby("Round"):

    below = int(
        (group["Combined_Trust"] < CANDIDATE_THRESHOLD).sum()
    )

    print(
        f"Round {round_number:02d} | "
        f"Minimum={group['Combined_Trust'].min():.6f} | "
        f"Maximum={group['Combined_Trust'].max():.6f} | "
        f"Below threshold={below}/10"
    )

print("\nCLIENT-BY-CLIENT ANALYSIS")
print("-" * 75)

for client_id, group in data.groupby("Client"):

    below = int(
        (group["Combined_Trust"] < CANDIDATE_THRESHOLD).sum()
    )

    print(
        f"Client {int(client_id):02d} | "
        f"Mean trust={group['Combined_Trust'].mean():.6f} | "
        f"Minimum={group['Combined_Trust'].min():.6f} | "
        f"Below threshold={below}/10"
    )

print("\nANALYSIS COMPLETED")
