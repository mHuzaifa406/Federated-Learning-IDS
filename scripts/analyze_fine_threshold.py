
import pandas as pd
from pathlib import Path

file_path = (
    Path("results")
    / "scenario_C_poisoned_trfa"
    / "trfa_client_trust_history.csv"
)

data = pd.read_csv(file_path)

malicious = data[data["Malicious"] == True]
honest = data[data["Malicious"] == False]

max_malicious = malicious["Combined_Trust"].max()
min_honest = honest["Combined_Trust"].min()

gap = min_honest - max_malicious
candidate_threshold = (max_malicious + min_honest) / 2

print("\nFINE-GRAINED TRFA THRESHOLD ANALYSIS")
print("-" * 55)

print(f"Maximum malicious trust: {max_malicious:.10f}")
print(f"Minimum honest trust:    {min_honest:.10f}")
print(f"Separation gap:          {gap:.10f}")
print(f"Candidate threshold:     {candidate_threshold:.10f}")

if gap > 0:

    print("\nTHRESHOLD CANDIDATE RESULTS")
    print("-" * 55)

    for threshold in [
        0.7410,
        0.7415,
        0.7420,
        candidate_threshold,
        0.7425,
        0.7430
    ]:

        detected = int(
            (malicious["Combined_Trust"] < threshold).sum()
        )

        false_excluded = int(
            (honest["Combined_Trust"] < threshold).sum()
        )

        print(
            f"Threshold: {threshold:.10f} | "
            f"Detected: {detected}/10 | "
            f"Honest excluded: {false_excluded}/90"
        )

else:
    print("\nNo complete separation exists in this experiment.")

print("\nANALYSIS COMPLETED")
