
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("results")

CLEAN_FILE = (
    ROOT / "scenario_H_noniid_trust_calibration"
    / "clean_noniid_trfa_trust_history.csv"
)

POISONED_FILE = (
    ROOT / "scenario_K_noniid_client07_matched_poison"
    / "matched_poison_noniid_trfa_trust_history.csv"
)

OUTPUT_DIR = (
    ROOT / "scenario_K_noniid_client07_matched_poison"
)

THRESHOLD = 0.7422773619
ATTACKER = 7

SIGNALS = [
    "Update_Similarity",
    "Training_Consistency",
    "Historical_Trust_Before",
    "Combined_Trust"
]


def main():

    print("\nCLIENT 07 MATCHED ATTACK COMPARISON")
    print("=" * 90)

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(CLEAN_FILE)

    if not POISONED_FILE.exists():
        raise FileNotFoundError(POISONED_FILE)

    clean = pd.read_csv(CLEAN_FILE)
    poisoned = pd.read_csv(POISONED_FILE)

    # Verify that both experiments contain the same clients
    # and communication rounds.
    if set(clean["Client"]) != set(poisoned["Client"]):
        raise ValueError("Client IDs differ between experiments.")

    if set(clean["Round"]) != set(poisoned["Round"]):
        raise ValueError("Communication rounds differ.")

    # --------------------------------------------------------
    # 1. Compare Client 07's trust before and after poisoning
    # --------------------------------------------------------

    clean_attacker = clean[
        clean["Client"] == ATTACKER
    ].sort_values("Round")

    poisoned_attacker = poisoned[
        poisoned["Client"] == ATTACKER
    ].sort_values("Round")

    print("\nCLIENT 07: CLEAN VS MATCHED POISONING")
    print("-" * 90)

    comparison = pd.DataFrame({
        "Round": clean_attacker["Round"].to_numpy()
    })

    for signal in SIGNALS:

        clean_values = clean_attacker[signal].to_numpy()
        poisoned_values = poisoned_attacker[signal].to_numpy()

        comparison[f"Clean_{signal}"] = clean_values
        comparison[f"Poisoned_{signal}"] = poisoned_values

        comparison[f"Change_{signal}"] = (
            poisoned_values - clean_values
        )

    display_columns = [
        "Round",
        "Clean_Combined_Trust",
        "Poisoned_Combined_Trust",
        "Change_Combined_Trust"
    ]

    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nAVERAGE TRUST-SIGNAL CHANGES FOR CLIENT 07")
    print("-" * 90)

    for signal in SIGNALS:

        change = (
            comparison[f"Change_{signal}"].mean()
        )

        print(f"{signal:<28}: {change:+.6f}")

    # --------------------------------------------------------
    # 2. Examine the other nine honest clients
    # --------------------------------------------------------

    print("\nHONEST CLIENT TRUST CHANGES")
    print("-" * 90)

    clean_honest = clean[
        clean["Client"] != ATTACKER
    ]

    poisoned_honest = poisoned[
        poisoned["Client"] != ATTACKER
    ]

    merged = clean_honest.merge(
        poisoned_honest,
        on=["Round", "Client"],
        suffixes=("_Clean", "_Poisoned")
    )

    honest_rows = []

    for client_id in sorted(merged["Client"].unique()):

        group = merged[
            merged["Client"] == client_id
        ]

        mean_change = (
            group["Combined_Trust_Poisoned"]
            - group["Combined_Trust_Clean"]
        ).mean()

        honest_rows.append({
            "Client": int(client_id),
            "Mean_Trust_Change": float(mean_change)
        })

    honest_comparison = pd.DataFrame(honest_rows)

    print(
        honest_comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:+.6f}"
        )
    )

    # --------------------------------------------------------
    # 3. Retrospective threshold analysis in Scenario K
    # --------------------------------------------------------

    print("\nRETROSPECTIVE THRESHOLD ANALYSIS")
    print("-" * 90)

    malicious = poisoned[
        poisoned["Client"] == ATTACKER
    ]

    honest = poisoned[
        poisoned["Client"] != ATTACKER
    ]

    detected = (
        malicious["Combined_Trust"] < THRESHOLD
    )

    false_flags = (
        honest["Combined_Trust"] < THRESHOLD
    )

    print(
        "Attacker client-rounds flagged:",
        f"{int(detected.sum())}/{len(malicious)}"
    )

    print(
        "Honest client-rounds flagged:",
        f"{int(false_flags.sum())}/{len(honest)}"
    )

    print(
        "Attacker flag proportion:",
        f"{detected.mean() * 100:.2f}%"
    )

    print(
        "Honest false-flag proportion:",
        f"{false_flags.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 4. Per-round attacker ranking and ROC-AUC
    # --------------------------------------------------------

    print("\nATTACKER RANKING IN SCENARIO K")
    print("-" * 90)

    round_results = []

    for round_number, group in poisoned.groupby("Round"):

        labels = (
            group["Client"] == ATTACKER
        ).astype(int)

        scores = -group["Combined_Trust"]

        auc = roc_auc_score(
            labels,
            scores
        )

        attacker = group[
            group["Client"] == ATTACKER
        ].iloc[0]

        sorted_group = group.sort_values(
            "Combined_Trust",
            ascending=True
        )

        rank = int(
            sorted_group["Client"].tolist().index(
                ATTACKER
            ) + 1
        )

        round_results.append({
            "Round": int(round_number),
            "Attacker_Trust": float(
                attacker["Combined_Trust"]
            ),
            "Attacker_Rank": rank,
            "ROC_AUC": float(auc),
            "Flagged": bool(
                attacker["Combined_Trust"] < THRESHOLD
            )
        })

        print(
            f"Round {int(round_number):02d} | "
            f"Trust={attacker['Combined_Trust']:.6f} | "
            f"Rank={rank}/10 | "
            f"ROC-AUC={auc:.4f}"
        )

    # --------------------------------------------------------
    # 5. Overall exploratory separation
    # --------------------------------------------------------

    labels = poisoned["Client"].eq(ATTACKER).astype(int)

    auc = roc_auc_score(
        labels,
        -poisoned["Combined_Trust"]
    )

    average_precision = average_precision_score(
        labels,
        -poisoned["Combined_Trust"]
    )

    print("\nOVERALL EXPLORATORY SEPARATION")
    print("-" * 90)

    print(f"ROC-AUC: {auc:.6f}")
    print(f"Average precision: {average_precision:.6f}")

    # --------------------------------------------------------
    # 6. Save comparison outputs
    # --------------------------------------------------------

    comparison.to_csv(
        OUTPUT_DIR / "client07_clean_vs_poisoned_trust.csv",
        index=False
    )

    honest_comparison.to_csv(
        OUTPUT_DIR / "honest_client_trust_changes.csv",
        index=False
    )

    pd.DataFrame(round_results).to_csv(
        OUTPUT_DIR / "client07_attacker_ranking.csv",
        index=False
    )

    print("\nFILES SAVED")
    print("Client 07 comparison: client07_clean_vs_poisoned_trust.csv")
    print("Honest-client comparison: honest_client_trust_changes.csv")
    print("Attacker ranking: client07_attacker_ranking.csv")

    print("\nIMPORTANT:")
    print(
        "Threshold results are retrospective. "
        "Both experiments used unscreened aggregation."
    )

    print(
        "ROC-AUC and average precision pool repeated "
        "client-round observations and are exploratory."
    )

    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
