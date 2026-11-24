"""Helpers for paired clean-versus-delayed-attack BCTCD evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError(f"{label} contains unrecognized boolean values.")
    return parsed.astype(bool)


def prepare_clean_replay_frame(raw: pd.DataFrame, target_client: int) -> pd.DataFrame:
    """Create a replay-only sentinel view of a genuinely all-honest control.

    The source frame is never mutated. A temporary Is_Malicious=True sentinel is
    set only for the probe client because the existing BCTCD core expects one
    target row per round. The returned scoring output must be relabelled honest.
    """
    required = {"Round", "Client", "Is_Malicious", "Attack_Active", "Attack_Start_Round", "Target_Client"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Clean-control CSV missing required columns: {missing}")
    out = raw.copy()
    out["Round"] = pd.to_numeric(out["Round"], errors="raise").astype(int)
    out["Client"] = pd.to_numeric(out["Client"], errors="raise").astype(int)
    out["Is_Malicious"] = parse_bool(out["Is_Malicious"], "Is_Malicious")
    out["Attack_Active"] = parse_bool(out["Attack_Active"], "Attack_Active")
    out["Target_Client"] = parse_bool(out["Target_Client"], "Target_Client")
    if out["Is_Malicious"].any():
        raise ValueError("Clean control is invalid: a client is marked malicious.")
    if out["Attack_Active"].any():
        raise ValueError("Clean control is invalid: attack-active rows were found.")
    if out.duplicated(["Round", "Client"]).any():
        raise ValueError("Clean control has duplicate (Round, Client) rows.")
    rounds = sorted(out["Round"].unique().tolist())
    clients = sorted(out["Client"].unique().tolist())
    if rounds != list(range(1, len(rounds) + 1)):
        raise ValueError(f"Clean-control rounds must be consecutive from 1; found {rounds}.")
    if target_client not in clients:
        raise ValueError(f"Probe client {target_client} is absent from clean control.")
    counts = out.groupby("Round")["Client"].nunique()
    if not (counts == len(clients)).all():
        raise ValueError("Each clean-control round must contain every client exactly once.")
    target_counts = out.groupby("Round")["Target_Client"].sum()
    if not (target_counts == 1).all():
        raise ValueError("Target_Client must identify exactly one probe row per round.")
    observed_target = set(out.loc[out["Target_Client"], "Client"].unique().tolist())
    if observed_target != {int(target_client)}:
        raise ValueError(f"Target_Client flags do not consistently identify client {target_client}.")

    # Replay-only sentinel: there is no attack; Attack_Start_Round=last+1 ensures
    # every row remains inactive in the core's validation logic.
    out["Is_Malicious"] = out["Client"].eq(int(target_client))
    out["Attack_Active"] = False
    out["Attack_Start_Round"] = int(max(rounds) + 1)
    return out


def compare_pre_attack_trajectories(
    clean_raw: pd.DataFrame,
    attack_raw: pd.DataFrame,
    attack_start_round: int,
    tolerance: float = 1e-7,
) -> dict:
    """Compare common numeric per-client signals before attack onset."""
    keys = ["Round", "Client"]
    clean = clean_raw.copy()
    attack = attack_raw.copy()
    for name, frame in (("clean", clean), ("attack", attack)):
        missing = set(keys) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} Phase 9 file missing keys: {sorted(missing)}")
        frame["Round"] = pd.to_numeric(frame["Round"], errors="raise").astype(int)
        frame["Client"] = pd.to_numeric(frame["Client"], errors="raise").astype(int)
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} Phase 9 file has duplicate (Round, Client) rows.")
    warm_rounds = list(range(1, int(attack_start_round)))
    c = clean[clean["Round"].isin(warm_rounds)]
    a = attack[attack["Round"].isin(warm_rounds)]
    merged = c.merge(a, on=keys, suffixes=("_clean", "_attack"), how="outer", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("Clean and attack runs do not have identical client-round keys during warm-up.")

    excluded = {
        "Round", "Client", "Is_Malicious", "Target_Client", "Control_Condition",
        "Attack_Active", "Attack_Start_Round", "Condition", "Poisoned_Records",
    }
    common_numeric: list[str] = []
    for col in clean.columns:
        if col in excluded or col not in attack.columns:
            continue
        if pd.api.types.is_numeric_dtype(clean[col]) and pd.api.types.is_numeric_dtype(attack[col]):
            common_numeric.append(col)
    if not common_numeric:
        raise ValueError("No common numeric columns available for warm-up trajectory comparison.")
    per_column = {}
    global_max = 0.0
    for col in common_numeric:
        left = pd.to_numeric(merged[f"{col}_clean"], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(merged[f"{col}_attack"], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            raise ValueError(f"Non-finite values encountered in warm-up comparison column {col}.")
        diff = np.abs(left - right)
        mx = float(diff.max(initial=0.0))
        per_column[col] = mx
        global_max = max(global_max, mx)
    return {
        "warmup_rounds_compared": warm_rounds,
        "client_round_pairs_compared": int(len(merged)),
        "numeric_columns_compared": common_numeric,
        "max_absolute_difference": global_max,
        "tolerance": float(tolerance),
        "matched_within_tolerance": bool(global_max <= tolerance),
        "max_difference_by_column": per_column,
    }


def build_matched_round_comparison(
    clean_scores: pd.DataFrame,
    attack_scores: pd.DataFrame,
    target_client: int,
    attack_start_round: int,
) -> pd.DataFrame:
    """Pair clean-control and poisoned BCTCD rows by round and client."""
    required_score = {
        "Round", "Client", "BCTCD_Round_Deviation", "BCTCD_CUSUM_Score",
        "BCTCD_Deviation_Rank", "BCTCD_CUSUM_Rank",
        "BCTCD_Deviation_Top3", "BCTCD_CUSUM_Top3",
    }
    for label, frame in (("clean", clean_scores), ("attack", attack_scores)):
        missing = sorted(required_score - set(frame.columns))
        if missing:
            raise ValueError(f"{label} BCTCD score file missing columns: {missing}")
    keys = ["Round", "Client"]
    clean = clean_scores.copy()
    attack = attack_scores.copy()
    for frame in (clean, attack):
        frame["Round"] = pd.to_numeric(frame["Round"], errors="raise").astype(int)
        frame["Client"] = pd.to_numeric(frame["Client"], errors="raise").astype(int)
    if clean.duplicated(keys).any() or attack.duplicated(keys).any():
        raise ValueError("Duplicate (Round, Client) records in BCTCD score files.")
    keep = keys + [
        "BCTCD_Round_Deviation", "BCTCD_CUSUM_Score", "BCTCD_Deviation_Rank",
        "BCTCD_CUSUM_Rank", "BCTCD_Deviation_Top3", "BCTCD_CUSUM_Top3",
    ]
    merged = clean[keep].merge(attack[keep], on=keys, how="outer", suffixes=("_clean", "_attack"),
                               validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("Clean and attack BCTCD files must have matching monitored client-round keys.")
    merged = merged.drop(columns="_merge")
    merged["Is_Probe_Client"] = merged["Client"].eq(int(target_client))
    merged["Attack_Active"] = merged["Round"] >= int(attack_start_round)
    for score in ("BCTCD_Round_Deviation", "BCTCD_CUSUM_Score"):
        merged[f"{score}_Poison_Minus_Clean"] = merged[f"{score}_attack"] - merged[f"{score}_clean"]
    return merged.sort_values(["Round", "Client"]).reset_index(drop=True)
