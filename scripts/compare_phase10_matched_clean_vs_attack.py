"""Compare BCTCD from matched all-honest and delayed-onset Phase 9 runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase10_matched_control_core import build_matched_round_comparison, compare_pre_attack_trajectories, parse_bool


def parse_args():
    p = argparse.ArgumentParser(description="Paired clean-control versus delayed-attack BCTCD comparison.")
    p.add_argument("--clean-phase9-dir", required=True)
    p.add_argument("--attack-phase9-dir", required=True)
    p.add_argument("--clean-phase10-dir", required=True)
    p.add_argument("--attack-phase10-dir", required=True)
    p.add_argument("--target-client", required=True, type=int, choices=range(1, 11))
    p.add_argument("--attack-start-round", required=True, type=int, choices=(4, 6))
    p.add_argument("--output-tag", required=True)
    return p.parse_args()


def resolve(value: str) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()


def read_meta(folder: Path, filename: str) -> dict:
    path = folder / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(args):
    clean9, attack9, clean10, attack10 = map(resolve, [
        args.clean_phase9_dir, args.attack_phase9_dir, args.clean_phase10_dir, args.attack_phase10_dir,
    ])
    files = {
        "clean Phase 9": clean9 / "phase9_client_round_scores.csv",
        "attack Phase 9": attack9 / "phase9_client_round_scores.csv",
        "clean Phase 10": clean10 / "phase10_bctcd_clean_client_round_scores.csv",
        "attack Phase 10": attack10 / "phase10_bctcd_client_round_scores.csv",
    }
    for label, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} file: {path}")
    c9, a9, c10, a10 = (pd.read_csv(files[k]) for k in files)

    # Confirm source clean run is genuinely all-honest; identify the designated probe.
    if parse_bool(c9["Is_Malicious"], "clean Is_Malicious").any():
        raise ValueError("Clean Phase 9 source contains a malicious-client flag.")
    if parse_bool(c9["Attack_Active"], "clean Attack_Active").any():
        raise ValueError("Clean Phase 9 source contains attack-active rows.")
    if "Target_Client" not in c9.columns:
        raise ValueError("Clean Phase 9 source is missing Target_Client.")
    target_flags = parse_bool(c9["Target_Client"], "clean Target_Client")
    if set(pd.to_numeric(c9.loc[target_flags, "Client"], errors="raise").astype(int).unique()) != {args.target_client}:
        raise ValueError("Clean Phase 9 probe-client flag does not match --target-client.")
    attack_malicious = parse_bool(a9["Is_Malicious"], "attack Is_Malicious")
    observed_targets = set(pd.to_numeric(a9.loc[attack_malicious, "Client"], errors="raise").astype(int).unique())
    if observed_targets != {args.target_client}:
        raise ValueError(f"Attack run malicious-client ID mismatch: {sorted(observed_targets)}")
    if int(pd.to_numeric(a9["Attack_Start_Round"], errors="raise").nunique()) != 1:
        raise ValueError("Attack Phase 9 metadata has inconsistent attack-start rounds.")
    actual_start = int(pd.to_numeric(a9["Attack_Start_Round"], errors="raise").iloc[0])
    if actual_start != args.attack_start_round:
        raise ValueError(f"Attack start mismatch: expected {args.attack_start_round}, found {actual_start}.")

    # Verify the clean and poisoned FL trajectories match exactly before attack onset.
    pre_match = compare_pre_attack_trajectories(c9, a9, args.attack_start_round)

    clean_meta = read_meta(clean9, "phase9_metadata.json")
    attack_meta = read_meta(attack9, "phase9_metadata.json")
    for key in ("training_seed", "poison_seed", "partition_seed", "dirichlet_alpha"):
        cv, av = clean_meta.get(key), attack_meta.get(key)
        if cv is not None and av is not None and cv != av:
            raise ValueError(f"Matched-run metadata mismatch for {key}: clean={cv}, attack={av}.")

    # Validate the BCTCD exports themselves: the clean control has no malicious
    # or active rows; the attack run has exactly one persistent target label and
    # activity beginning on the requested round.
    if parse_bool(c10["Is_Malicious"], "clean Phase 10 Is_Malicious").any():
        raise ValueError("Clean Phase 10 output must mark every client honest.")
    if parse_bool(c10["Attack_Active"], "clean Phase 10 Attack_Active").any():
        raise ValueError("Clean Phase 10 output must have no active attack rows.")
    a10_malicious = parse_bool(a10["Is_Malicious"], "attack Phase 10 Is_Malicious")
    a10_active = parse_bool(a10["Attack_Active"], "attack Phase 10 Attack_Active")
    a10_target_ids = set(pd.to_numeric(a10.loc[a10_malicious, "Client"], errors="raise").astype(int).unique())
    if a10_target_ids != {args.target_client}:
        raise ValueError(f"Attack Phase 10 target mismatch: {sorted(a10_target_ids)}")
    expected_active = (pd.to_numeric(a10["Round"], errors="raise").astype(int) >= args.attack_start_round) & (pd.to_numeric(a10["Client"], errors="raise").astype(int) == args.target_client)
    if not np.array_equal(a10_active.to_numpy(dtype=bool), expected_active.to_numpy(dtype=bool)):
        raise ValueError("Attack Phase 10 activity flags do not match the requested client/onset.")

    paired = build_matched_round_comparison(c10, a10, args.target_client, args.attack_start_round)
    output_dir = PROJECT_ROOT / "results" / f"phase10_matched_clean_attack_{args.output_tag}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains results: {output_dir}. Use a new --output-tag.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "phase10_matched_round_client_comparison.csv", index=False)

    probe = paired[paired["Is_Probe_Client"]].copy()
    monitored = probe[probe["Round"] > 3]
    pre_onset = probe[probe["Round"] < args.attack_start_round]
    active = probe[probe["Round"] >= args.attack_start_round]
    clean_probe_top3 = int(parse_bool(probe["BCTCD_CUSUM_Top3_clean"], "clean CUSUM Top3").sum())
    attack_probe_pre_top3 = int(parse_bool(pre_onset["BCTCD_CUSUM_Top3_attack"], "attack pre-onset CUSUM Top3").sum())
    attack_probe_active_top3 = int(parse_bool(active["BCTCD_CUSUM_Top3_attack"], "attack active CUSUM Top3").sum())
    deviation_clean_top3 = int(parse_bool(probe["BCTCD_Deviation_Top3_clean"], "clean deviation Top3").sum())
    deviation_attack_active_top3 = int(parse_bool(active["BCTCD_Deviation_Top3_attack"], "attack active deviation Top3").sum())

    # Count how often honest clients occupy top-three positions in each model run.
    honest_attack_rows = paired[~paired["Is_Probe_Client"]]
    honest_top3_summary = honest_attack_rows.groupby("Round", as_index=False).agg(
        Honest_Clients_In_Attack_CUSUM_Top3=("BCTCD_CUSUM_Top3_attack", "sum"),
        Honest_Clients_In_Clean_CUSUM_Top3=("BCTCD_CUSUM_Top3_clean", "sum"),
        Honest_Clients_In_Attack_Deviation_Top3=("BCTCD_Deviation_Top3_attack", "sum"),
        Honest_Clients_In_Clean_Deviation_Top3=("BCTCD_Deviation_Top3_clean", "sum"),
    )
    honest_top3_summary.to_csv(output_dir / "phase10_matched_honest_top3_counts.csv", index=False)

    summary = {
        "phase": "Phase 10 matched clean-versus-delayed-attack comparison",
        "target_client": args.target_client,
        "attack_start_round": args.attack_start_round,
        "matched_metadata": {k: {"clean": clean_meta.get(k), "attack": attack_meta.get(k)} for k in ("training_seed", "poison_seed", "partition_seed", "dirichlet_alpha")},
        "pre_attack_trajectory_match": pre_match,
        "probe_clean_cusum_top3_rounds_all_monitoring": clean_probe_top3,
        "probe_clean_monitoring_rounds": int(len(probe)),
        "probe_clean_deviation_top3_rounds_all_monitoring": deviation_clean_top3,
        "probe_attack_cusum_top3_before_onset": attack_probe_pre_top3,
        "probe_attack_clean_monitoring_rounds_before_onset": int(len(pre_onset)),
        "probe_attack_cusum_top3_after_onset": attack_probe_active_top3,
        "probe_attack_active_rounds": int(len(active)),
        "probe_attack_deviation_top3_after_onset": deviation_attack_active_top3,
        "mean_target_cusum_poison_minus_clean_after_onset": float(active["BCTCD_CUSUM_Score_Poison_Minus_Clean"].mean()),
        "mean_target_deviation_poison_minus_clean_after_onset": float(active["BCTCD_Round_Deviation_Poison_Minus_Clean"].mean()),
        "screening_enabled": False,
        "aggregation_modified": False,
        "independent_test_used": False,
        "note": "Top-3 metrics are cohort ranks, not calibrated false-positive rates. The clean control is used to measure how often the probe client is unusual when honest.",
    }
    (output_dir / "phase10_matched_comparison_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "PHASE 10 — MATCHED CLEAN CONTROL VS DELAYED-ONSET ATTACK",
        "========================================================",
        f"Probe/malicious client: {args.target_client}",
        f"Attack starts: Round {args.attack_start_round}",
        f"Pre-attack client-round pairs compared: {pre_match['client_round_pairs_compared']}",
        f"Pre-attack max absolute difference: {pre_match['max_absolute_difference']:.12g}",
        f"Pre-attack trajectories matched within {pre_match['tolerance']}: {pre_match['matched_within_tolerance']}",
        "",
        f"Probe clean-control CUSUM top-3: {clean_probe_top3}/{len(probe)} monitored rounds",
        f"Probe clean-control deviation top-3: {deviation_clean_top3}/{len(probe)} monitored rounds",
        f"Probe attacked-run CUSUM top-3 before onset: {attack_probe_pre_top3}/{len(pre_onset)} rounds",
        f"Probe attacked-run CUSUM top-3 after onset: {attack_probe_active_top3}/{len(active)} rounds",
        f"Probe attacked-run deviation top-3 after onset: {deviation_attack_active_top3}/{len(active)} rounds",
        f"Mean target CUSUM change (attack minus clean) after onset: {summary['mean_target_cusum_poison_minus_clean_after_onset']:.6f}",
        f"Mean target deviation change (attack minus clean) after onset: {summary['mean_target_deviation_poison_minus_clean_after_onset']:.6f}",
        "",
        "Top-3 is a cohort-relative rank, not a calibrated alarm threshold.",
        "No screening; no aggregation change; independent test set not used.",
    ]
    (output_dir / "phase10_matched_comparison_summary.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nPHASE 10 MATCHED CLEAN-VS-ATTACK COMPARISON COMPLETED")
    print("Output directory:", output_dir)
    print("Pre-attack trajectories matched:", pre_match["matched_within_tolerance"])
    print("Pre-attack maximum absolute difference:", f"{pre_match['max_absolute_difference']:.12g}")
    print("Clean-control probe CUSUM top-3:", f"{clean_probe_top3}/{len(probe)}")
    print("Attack probe CUSUM top-3 before onset:", f"{attack_probe_pre_top3}/{len(pre_onset)}")
    print("Attack probe CUSUM top-3 after onset:", f"{attack_probe_active_top3}/{len(active)}")
    print("Mean post-onset CUSUM change:", f"{summary['mean_target_cusum_poison_minus_clean_after_onset']:.6f}")
    print("Screening enabled: NO; aggregation modified: NO; independent test used: NO")
    return output_dir


if __name__ == "__main__":
    run(parse_args())
