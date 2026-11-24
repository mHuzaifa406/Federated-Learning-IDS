
import json
import pandas as pd
from pathlib import Path

RESULT_DIR = (
    Path("results") / "scenario_E_trfa_seed43"
)

summary_path = RESULT_DIR / "trfa_experiment_summary.json"
history_path = RESULT_DIR / "trfa_client_trust_history.csv"

with open(summary_path, "r", encoding="utf-8") as file:
    summary = json.load(file)

data = pd.read_csv(history_path)

threshold = summary["trfa_configuration"]["threshold"]

malicious = data[data["Malicious"] == True]
honest = data[data["Malicious"] == False]

print("\nSEED 43 EXPERIMENT VERIFICATION")
print("=" * 65)

print("Stored scenario label:", summary.get("scenario"))
print("Trust threshold:", threshold)
print("Poisoning details:")
print(json.dumps(summary["poisoning_statistics"], indent=2))

print("\nROUND-BY-ROUND TRUST SEPARATION")
print("-" * 65)

for round_number, group in data.groupby("Round"):

    attacker = group[group["Malicious"] == True].iloc[0]
    honest_group = group[group["Malicious"] == False]

    attacker_score = attacker["Combined_Trust"]
    minimum_honest = honest_group["Combined_Trust"].min()

    attacker_rejected = bool(attacker["Excluded"])
    honest_rejected = int(honest_group["Excluded"].sum())

    print(
        f"Round {round_number:02d} | "
        f"Attacker trust={attacker_score:.8f} | "
        f"Minimum honest={minimum_honest:.8f} | "
        f"Attacker rejected={attacker_rejected} | "
        f"Honest excluded={honest_rejected}"
    )

detection_count = int(malicious["Excluded"].sum())
false_exclusions = int(honest["Excluded"].sum())

print("\nFINAL VERIFICATION")
print("-" * 65)
print(f"Malicious detections: {detection_count}/10")
print(f"Honest false exclusions: {false_exclusions}/90")
print(f"Malicious detection rate: {detection_count / 10 * 100:.2f}%")
print(f"Honest false-exclusion rate: {false_exclusions / 90 * 100:.2f}%")

print("\nMaximum malicious trust:", f"{malicious['Combined_Trust'].max():.8f}")
print("Minimum honest trust:", f"{honest['Combined_Trust'].min():.8f}")

print("\nANALYSIS COMPLETED")
