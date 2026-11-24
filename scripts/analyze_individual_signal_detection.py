
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

RESULT_DIR = (
    Path("results")
    / "scenario_I_noniid_poison_trust_calibration"
)

FILE = RESULT_DIR / "poisoned_noniid_trfa_trust_history.csv"

data = pd.read_csv(FILE)

signals = [
    "Update_Similarity",
    "Training_Consistency",
    "Historical_Trust_Before",
    "Combined_Trust"
]

print("\nINDIVIDUAL TRUST SIGNAL DETECTION ANALYSIS")
print("=" * 90)

labels = data["Malicious"].astype(int)

summary = []

# ------------------------------------------------------------
# 1. Pooled signal comparison
# ------------------------------------------------------------

print("\nOVERALL SIGNAL COMPARISON")
print("-" * 90)

for signal in signals:

    suspiciousness = -data[signal]

    auc = roc_auc_score(labels, suspiciousness)

    ap = average_precision_score(
        labels,
        suspiciousness
    )

    malicious_mean = data.loc[
        data["Malicious"] == True,
        signal
    ].mean()

    honest_mean = data.loc[
        data["Malicious"] == False,
        signal
    ].mean()

    summary.append({
        "Signal": signal,
        "ROC_AUC": auc,
        "Average_Precision": ap,
        "Honest_Mean": honest_mean,
        "Malicious_Mean": malicious_mean
    })

    print(f"\n{signal}")
    print(f"  ROC-AUC: {auc:.6f}")
    print(f"  Average precision: {ap:.6f}")
    print(f"  Honest mean: {honest_mean:.6f}")
    print(f"  Malicious mean: {malicious_mean:.6f}")

# ------------------------------------------------------------
# 2. Per-round signal ranking
# ------------------------------------------------------------

print("\n\nPER-ROUND ATTACKER RANKING")
print("-" * 90)

round_results = []

for round_number, group in data.groupby("Round"):

    print(f"\nROUND {int(round_number):02d}")

    malicious_row = group[
        group["Malicious"] == True
    ].iloc[0]

    honest = group[
        group["Malicious"] == False
    ]

    result = {"Round": int(round_number)}

    for signal in signals:

        # Lower trust = greater suspicion.
        attacker_rank = int(
            group[signal]
            .rank(method="min", ascending=True)
            .loc[malicious_row.name]
        )

        round_auc = roc_auc_score(
            group["Malicious"].astype(int),
            -group[signal]
        )

        result[signal + "_Rank"] = attacker_rank
        result[signal + "_AUC"] = round_auc

        print(
            f"  {signal:<28} "
            f"Trust={malicious_row[signal]:.6f} | "
            f"Rank={attacker_rank}/10 | "
            f"ROC-AUC={round_auc:.4f}"
        )

    round_results.append(result)

# ------------------------------------------------------------
# 3. Mean per-round AUC
# ------------------------------------------------------------

round_df = pd.DataFrame(round_results)

print("\n\nMEAN PER-ROUND ROC-AUC")
print("-" * 90)

for signal in signals:

    auc_column = signal + "_AUC"

    print(
        f"{signal:<28} "
        f"{round_df[auc_column].mean():.6f}"
    )

# ------------------------------------------------------------
# 4. Save detailed analysis
# ------------------------------------------------------------

summary_df = pd.DataFrame(summary)

summary_path = (
    RESULT_DIR / "individual_signal_detection_summary.csv"
)

round_path = (
    RESULT_DIR / "individual_signal_detection_by_round.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

round_df.to_csv(
    round_path,
    index=False
)

print("\nFILES SAVED")
print("Overall signal analysis:", summary_path)
print("Round analysis:", round_path)

print("\nIMPORTANT:")
print(
    "Pooled AUC is exploratory because the same clients "
    "are observed repeatedly across rounds."
)

print(
    "Per-round AUC evaluates ranking within each round "
    "across ten clients."
)

print("\nANALYSIS COMPLETED")
