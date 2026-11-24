"""Run one complete Phase 5 attack scenario (diagnostics only).

Runs Similarity V2, Historical Trust V2 and frozen-checkpoint CAVC V2 for one
selected attacker/intensity, then performs offline fusion. Existing files are
never overwritten. The clean controls are reused and must match the requested
training seed/partition.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    p = argparse.ArgumentParser(description="Run one complete Phase 5 non-IID attack scenario.")
    p.add_argument("--malicious-client", required=True, type=int, choices=range(1, 11))
    p.add_argument("--poison-fraction", required=True, type=float, help="Share of eligible non-Benign labels flipped to Benign.")
    p.add_argument("--training-seed", type=int, default=43)
    p.add_argument("--poison-seed", type=int, default=2027)
    p.add_argument("--rounds", type=int, default=10, choices=(10,), help="Phase 5 fusion currently requires 10 rounds.")
    p.add_argument("--partition-root", default=str(ROOT / "data" / "processed" / "federated_noniid_alpha05_seed43"))
    p.add_argument("--clean-historical", default=str(RESULTS / "historical_trust_v2_clean_noniid_seed43_matched_clean_control_01" / "historical_v2_client_round_trust.csv"))
    p.add_argument("--clean-similarity", default=str(RESULTS / "similarity_v2_clean_noniid_seed43" / "similarity_v2_client_round_scores.csv"))
    p.add_argument("--global-checkpoint", default=str(RESULTS / "historical_trust_v2_clean_noniid_seed43_matched_clean_control_01" / "historical_v2_round_10.pt"))
    p.add_argument("--output-tag", required=True)
    args = p.parse_args()
    if not 0 < args.poison_fraction < 1:
        raise ValueError("--poison-fraction must be between 0 and 1.")
    tag = "".join(ch for ch in args.output_tag if ch.isalnum() or ch in "-_")
    if not tag:
        raise ValueError("--output-tag must contain safe letters, digits, '-' or '_'.")
    cid = args.malicious_client
    frac_tag = f"{args.poison_fraction:.4f}"
    poison_tag = f"c{cid:02d}_p{int(round(args.poison_fraction*100)):02d}_s{args.training_seed}_ps{args.poison_seed}_{tag}"
    poison_folder_token = f"poison-client{cid:02d}_p{frac_tag}_s{args.training_seed}_ps{args.poison_seed}_{tag}"

    clean_hist = Path(args.clean_historical).resolve()
    clean_sim = Path(args.clean_similarity).resolve()
    checkpoint = Path(args.global_checkpoint).resolve()
    for f in (clean_hist, clean_sim, checkpoint):
        if not f.exists():
            raise FileNotFoundError(f"Required matched clean control/checkpoint is missing: {f}")

    common = ["--condition", "poison", "--malicious-client", str(cid),
              "--poison-fraction", str(args.poison_fraction), "--training-seed", str(args.training_seed),
              "--poison-seed", str(args.poison_seed), "--rounds", str(args.rounds),
              "--partition-root", str(Path(args.partition_root).resolve()), "--output-tag", tag]

    commands = [
        [sys.executable, str(ROOT / "scripts" / "run_phase5_similarity.py"), *common],
        [sys.executable, str(ROOT / "scripts" / "run_phase5_historical.py"), *common],
        [sys.executable, str(ROOT / "scripts" / "run_phase5_cavc.py"),
         "--malicious-client", str(cid), "--poison-fraction", str(args.poison_fraction),
         "--seed", str(args.training_seed), "--poison-seed", str(args.poison_seed),
         "--partition-root", str(Path(args.partition_root).resolve()),
         "--global-checkpoint", str(checkpoint), "--output-tag", poison_tag],
    ]
    for command in commands:
        print("\nRUNNING:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)

    sim_poison = RESULTS / f"phase5_similarity_{poison_folder_token}" / "similarity_v2_client_round_scores.csv"
    hist_poison = RESULTS / f"phase5_historical_v2_{poison_folder_token}" / "historical_v2_client_round_trust.csv"
    cavc_dir = RESULTS / f"phase5_cavc_v2_paired_poison_client{cid:02d}_p{frac_tag}_{poison_tag}"
    cavc_csv = cavc_dir / "cavc_v2_client_summary_min_support_100.csv"
    fusion_command = [sys.executable, str(ROOT / "scripts" / "run_phase5_fusion.py"),
        "--malicious-client", str(cid), "--training-seed", str(args.training_seed),
        "--poison-fraction", str(args.poison_fraction), "--poison-seed", str(args.poison_seed),
        "--cavc-csv", str(cavc_csv), "--historical-clean", str(clean_hist),
        "--historical-poison", str(hist_poison), "--similarity-clean", str(clean_sim),
        "--similarity-poison", str(sim_poison), "--output-tag", tag]
    print("\nRUNNING:", " ".join(f'"{x}"' if " " in x else x for x in fusion_command))
    subprocess.run(fusion_command, cwd=ROOT, check=True)
    print("\nPHASE 5 SCENARIO COMPLETED:", poison_tag)
    print("Screening: DISABLED; aggregation changes: NONE; independent test: NOT USED")


if __name__ == "__main__":
    main()
