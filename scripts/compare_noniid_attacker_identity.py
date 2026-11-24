
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path("results")

SCENARIOS = {
    "Clean": (
        ROOT / "scenario_H_noniid_trust_calibration"
        / "clean_noniid_trfa_trust_history.csv"
    ),
    "Client10_Attack": (
        ROOT / "scenario_I_noniid_poison_trust_calibration"
        / "poisoned_noniid_trfa_trust_history.csv"
    ),
    "Client07_Attack": (
        ROOT / "scenario_J_noniid_client07_poison"
        / "poisoned_noniid_trfa_trust_history.csv"
    )
}

def load_data():

    frames = {}

    for name, path in SCENARIOS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing experiment file: {path}"
            )

        data = pd.read_csv(path)

        required = [
            "Round",
            "Client",
            "Update_Similarity",
            "Training_Consistency",
            "Historical_Trust_Before",
            "Combined_Trust"
        ]

        missing = [
            column for column in required
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"{name} is missing columns: {missing}"
            )

        frames[name] = data

    return frames


def print_client_comparison(frames, client_id):

    clean = frames["Clean"]

    attack = frames[
        "Client07_Attack"
        if client_id == 7
        else "Client10_Attack"
    ]

    clean_client = clean[
        clean["Client"] == client_id
    ].sort_values("Round")

    attack_client = attack[
        attack["Client"] == client_id
    ].sort_values("Round")

    comparison = pd.DataFrame({
        "Round": clean_client["Round"].to_numpy(),
        "Clean_Trust": clean_client[
            "Combined_Trust"
        ].to_numpy(),
        "Poisoned_Trust": attack_client[
            "Combined_Trust"
        ].to_numpy()
    })

    comparison["Trust_Change"] = (
        comparison["Poisoned_Trust"]
        - comparison["Clean_Trust"]
    )

    print(
        f"\nCLIENT {client_id:02d}: "
        "CLEAN VS POISONED"
    )

    print("-" * 85)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print(
        "\nAverage trust change:",
        f"{comparison['Trust_Change'].mean():+.6f}"
    )


def print_attacker_detection(frames, scenario_name, attacker_id):

    data = frames[scenario_name]

    malicious = data[
        data["Client"] == attacker_id
    ]

    honest = data[
        data["Client"] != attacker_id
    ]

    print(
        f"\nATTACKER RANKING: {scenario_name}"
    )

    print("-" * 85)

    for round_number, group in data.groupby("Round"):

        attacker = group[
            group["Client"] == attacker_id
        ].iloc[0]

        honest_scores = group[
            group["Client"] != attacker_id
        ]["Combined_Trust"]

        rank = int(
            group["Combined_Trust"]
            .rank(method="min", ascending=True)
            .loc[attacker.name]
        )

        auc = roc_auc_score(
            group["Client"].eq(attacker_id).astype(int),
            -group["Combined_Trust"]
        )

        print(
            f"Round {int(round_number):02d} | "
            f"Attacker trust={attacker['Combined_Trust']:.6f} | "
            f"Honest median={honest_scores.median():.6f} | "
            f"Rank={rank}/10 | "
            f"ROC-AUC={auc:.4f}"
        )

    # This is a retrospective diagnostic, not a screened run.
    print(
        "\nMean attacker trust:",
        f"{malicious['Combined_Trust'].mean():.6f}"
    )

    print(
        "Mean honest trust:",
        f"{honest['Combined_Trust'].mean():.6f}"
    )


def main():

    print("\nNON-IID ATTACKER IDENTITY COMPARISON")
    print("=" * 85)

    frames = load_data()

    # Compare Client 07 before and after becoming malicious.
    print_client_comparison(frames, 7)

    # Compare Client 10 before and after becoming malicious.
    print_client_comparison(frames, 10)

    print_attacker_detection(
        frames,
        "Client07_Attack",
        7
    )

    print_attacker_detection(
        frames,
        "Client10_Attack",
        10
    )

    output_dir = (
        ROOT / "scenario_J_noniid_client07_poison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nCOMPARISON COMPLETED")
    print("Original experiment files were not modified.")

    print(
        "\nNote: Trust comparisons are retrospective. "
        "They do not represent screened aggregation."
    )


if __name__ == "__main__":
    main()
