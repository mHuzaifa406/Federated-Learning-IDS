"""Compare completed clean and poisoned CAVC calibration CSV files.

Run after both analyze_cavc_noniid.py conditions finish:
    python scripts/compare_cavc_conditions.py --seed 43
"""

from pathlib import Path
import argparse
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    seed = args.seed

    clean_dir = PROJECT_ROOT / "results" / f"scenario_noniid_cavc_clean_seed{seed}"
    poison_dir = PROJECT_ROOT / "results" / f"scenario_noniid_cavc_poison_client07_seed{seed}"
    clean_path = clean_dir / "cavc_client_round_scores.csv"
    poison_path = poison_dir / "cavc_client_round_scores.csv"

    for path in (clean_path, poison_path):
        if not path.exists():
            raise FileNotFoundError(f"Required result file not found: {path}")

    clean = pd.read_csv(clean_path)
    poison = pd.read_csv(poison_path)
    clean = clean[clean["Min_Class_Support"] == 100].copy()
    poison = poison[poison["Min_Class_Support"] == 100].copy()

    client7_clean = clean[clean["Client"] == 7][["Round", "CAVC_Score"]].rename(
        columns={"CAVC_Score": "Client07_Clean_CAVC"}
    )
    client7_poison = poison[poison["Client"] == 7][["Round", "CAVC_Score"]].rename(
        columns={"CAVC_Score": "Client07_Poisoned_CAVC"}
    )
    clean_others = clean[clean["Client"] != 7].groupby("Round")["CAVC_Score"].mean().rename(
        "Clean_Other_Honest_Mean"
    )
    poison_others = poison[poison["Client"] != 7].groupby("Round")["CAVC_Score"].mean().rename(
        "Poison_Other_Honest_Mean"
    )

    comparison = (
        client7_clean.merge(client7_poison, on="Round", validate="one_to_one")
        .merge(clean_others, on="Round", validate="one_to_one")
        .merge(poison_others, on="Round", validate="one_to_one")
    )
    comparison["Clean_Client07_Gap_vs_Honest"] = (
        comparison["Client07_Clean_CAVC"] - comparison["Clean_Other_Honest_Mean"]
    )
    comparison["Poisoned_Client07_Gap_vs_Honest"] = (
        comparison["Client07_Poisoned_CAVC"] - comparison["Poison_Other_Honest_Mean"]
    )
    comparison["Client07_Poisoning_Score_Change"] = (
        comparison["Client07_Poisoned_CAVC"] - comparison["Client07_Clean_CAVC"]
    )

    out_path = PROJECT_ROOT / "results" / f"cavc_clean_vs_poison_client07_seed{seed}.csv"
    comparison.to_csv(out_path, index=False)

    print("CAVC CLEAN VS POISONED COMPARISON (minimum class support = 100)")
    print("=" * 95)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nMean Client 07 clean score:", f"{comparison['Client07_Clean_CAVC'].mean():.6f}")
    print("Mean Client 07 poisoned score:", f"{comparison['Client07_Poisoned_CAVC'].mean():.6f}")
    print("Mean clean honest-client score:", f"{comparison['Clean_Other_Honest_Mean'].mean():.6f}")
    print("Mean poisoned-run honest-client score:", f"{comparison['Poison_Other_Honest_Mean'].mean():.6f}")
    print("Mean Client 07 score change after poisoning:", f"{comparison['Client07_Poisoning_Score_Change'].mean():.6f}")
    print("\nSaved comparison:", out_path)
    print("Note: repeated client-round observations are descriptive, not independent samples.")


if __name__ == "__main__":
    main()
