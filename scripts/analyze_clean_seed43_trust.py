
import pandas as pd
from pathlib import Path

file_path = (
    Path("results")
    / "scenario_F_clean_trfa_seed43_control"
    / "clean_trfa_trust_history.csv"
)

data = pd.read_csv(file_path)

threshold = 0.7422773619

print("\nSEED 43 CLEAN CLIENT TRUST ANALYSIS")
print("=" * 65)

print("Threshold:", f"{threshold:.10f}")

minimum = data.loc[data["Combined_Trust"].idxmin()]

print("\nLowest observed honest-client trust:")
print("Round:", int(minimum["Round"]))
print("Client:", int(minimum["Client"]))
print("Trust:", f"{minimum['Combined_Trust']:.10f}")
print(
    "Margin above threshold:",
    f"{minimum['Combined_Trust'] - threshold:.10f}"
)

print("\nROUND-BY-ROUND MINIMUM TRUST")
print("-" * 65)

for round_number, group in data.groupby("Round"):

    lowest = group.loc[group["Combined_Trust"].idxmin()]

    print(
        f"Round {round_number:02d} | "
        f"Minimum trust={lowest['Combined_Trust']:.8f} | "
        f"Client={int(lowest['Client'])} | "
        f"Accepted={bool(lowest['Accepted'])}"
    )

print("\nOverall minimum trust:", f"{data['Combined_Trust'].min():.10f}")
print("Overall maximum trust:", f"{data['Combined_Trust'].max():.10f}")
print("Total false exclusions:", int(data["False_Exclusion"].sum()))

print("\nANALYSIS COMPLETED")
