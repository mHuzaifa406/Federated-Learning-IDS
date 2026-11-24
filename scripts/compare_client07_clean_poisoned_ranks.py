
import pandas as pd
from pathlib import Path

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

ATTACKER = 7

SIGNALS = [
    "Update_Similarity",
    "Training_Consistency",
    "Historical_Trust_Before",
    "Combined_Trust"
]


def main():

    print("\nCLIENT 07: CLEAN VS POISONED RANK COMPARISON")
    print("=" * 100)

    clean = pd.read_csv(CLEAN_FILE)
    poisoned = pd.read_csv(POISONED_FILE)

    results = []

    for round_number in sorted(clean["Round"].unique()):

        clean_round = clean[
            clean["Round"] == round_number
        ]

        poisoned_round = poisoned[
            poisoned["Round"] == round_number
        ]

        clean_attacker = clean_round[
            clean_round["Client"] == ATTACKER
        ].iloc[0]

        poisoned_attacker = poisoned_round[
            poisoned_round["Client"] == ATTACKER
        ].iloc[0]

        print(f"\nROUND {int(round_number):02d}")
        print("-" * 100)

        for signal in SIGNALS:

            clean_rank = int(
                clean_round[signal]
                .rank(method="min", ascending=True)
                .loc[clean_attacker.name]
            )

            poisoned_rank = int(
                poisoned_round[signal]
                .rank(method="min", ascending=True)
                .loc[poisoned_attacker.name]
            )

            clean_score = float(
                clean_attacker[signal]
            )

            poisoned_score = float(
                poisoned_attacker[signal]
            )

            change = poisoned_score - clean_score

            print(
                f"{signal:<28} "
                f"Clean score={clean_score:.6f} | "
                f"Poisoned score={poisoned_score:.6f} | "
                f"Change={change:+.6f} | "
                f"Clean rank={clean_rank}/10 | "
                f"Poisoned rank={poisoned_rank}/10"
            )

            results.append({
                "Round": int(round_number),
                "Client": ATTACKER,
                "Signal": signal,
                "Clean_Score": clean_score,
                "Poisoned_Score": poisoned_score,
                "Score_Change": change,
                "Clean_Rank": clean_rank,
                "Poisoned_Rank": poisoned_rank,
                "Rank_Change": poisoned_rank - clean_rank
            })

    # --------------------------------------------------------
    # Summary across all rounds
    # --------------------------------------------------------

    result_df = pd.DataFrame(results)

    print("\nAVERAGE RANK COMPARISON")
    print("-" * 100)

    for signal in SIGNALS:

        group = result_df[
            result_df["Signal"] == signal
        ]

        print(
            f"{signal:<28} "
            f"Clean average rank="
            f"{group['Clean_Rank'].mean():.2f}/10 | "
            f"Poisoned average rank="
            f"{group['Poisoned_Rank'].mean():.2f}/10"
        )

    print("\nAVERAGE TRUST-SCORE CHANGES")
    print("-" * 100)

    for signal in SIGNALS:

        group = result_df[
            result_df["Signal"] == signal
        ]

        print(
            f"{signal:<28} "
            f"{group['Score_Change'].mean():+.6f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "client07_clean_poisoned_rank_comparison.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    print("\nComparison saved:", output_path)

    print("\nANALYSIS COMPLETED")

    print(
        "Note: Both trajectories use unscreened aggregation. "
        "This is a retrospective paired comparison."
    )


if __name__ == "__main__":
    main()
