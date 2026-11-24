from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CLIENTS = list(range(1, 11))
ROUNDS = list(range(1, 11))
EXPECTED_SCENARIOS = 12


def _member_map(zf: zipfile.ZipFile) -> dict[str, str]:
    """Normalize Windows-style ZIP member separators to POSIX paths."""
    return {info.filename.replace('\\', '/'): info.filename for info in zf.infolist() if not info.is_dir()}


def _read_json(zf: zipfile.ZipFile, member: str) -> dict[str, Any]:
    return json.loads(zf.read(member).decode('utf-8-sig'))


def _read_csv(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(member)))


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f'{label} is missing required columns: {missing}')


def _validate_rounds(df: pd.DataFrame, label: str, condition: str) -> None:
    _require_columns(df, ['Round', 'Client', 'Is_Malicious'], label)
    if len(df) != 100:
        raise ValueError(f'{label} must have 100 rows (10 clients x 10 rounds); found {len(df)}')
    df['Round'] = pd.to_numeric(df['Round'], errors='raise').astype(int)
    df['Client'] = pd.to_numeric(df['Client'], errors='raise').astype(int)
    if set(df['Client']) != set(CLIENTS) or set(df['Round']) != set(ROUNDS):
        raise ValueError(f'{label} must cover clients 1–10 and rounds 1–10 exactly')
    if df.duplicated(['Round', 'Client']).any():
        raise ValueError(f'{label} contains duplicate (Round, Client) rows')
    per_client = df.groupby('Client')['Round'].nunique()
    if not (per_client == 10).all():
        raise ValueError(f'{label} does not contain exactly 10 rounds per client')
    flags = df.groupby('Client')['Is_Malicious'].nunique()
    if (flags != 1).any():
        raise ValueError(f'{label} has inconsistent malicious flags')
    if condition == 'clean':
        if _as_bool_series(df['Is_Malicious']).any():
            raise ValueError(f'{label} clean condition unexpectedly marks a malicious client')
    else:
        marked = set(df.loc[_as_bool_series(df['Is_Malicious']), 'Client'].astype(int))
        if len(marked) != 1:
            raise ValueError(f'{label} poisoned condition must mark exactly one malicious client; found {marked}')


def _as_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    out = s.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        raise ValueError(f'Invalid boolean values: {s[out.isna()].astype(str).unique().tolist()}')
    return out.astype(bool)


def _percentile(values: pd.Series, high_is_suspicious: bool) -> pd.Series:
    vals = pd.to_numeric(values, errors='coerce')
    if vals.isna().any() or not np.isfinite(vals.to_numpy(dtype=float)).all():
        raise ValueError('Non-finite value encountered during percentile scoring')
    # Largest gets 1.0 when high_is_suspicious; smallest gets 1.0 otherwise.
    return vals.rank(method='average', ascending=high_is_suspicious, pct=True)


def _auc_rank(frame: pd.DataFrame, score_col: str, target_client: int) -> tuple[float, float, float, float]:
    y = (frame['Client'].astype(int).to_numpy() == int(target_client)).astype(int)
    s = frame[score_col].astype(float).to_numpy()
    auc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))
    ranked = frame.assign(_score=s).sort_values(['_score', 'Client'], ascending=[False, True])
    rank = float(np.flatnonzero(ranked['Client'].astype(int).to_numpy() == int(target_client))[0] + 1)
    target_score = float(frame.loc[frame['Client'].astype(int) == int(target_client), score_col].iloc[0])
    honest_mean = float(frame.loc[frame['Client'].astype(int) != int(target_client), score_col].mean())
    return auc, ap, rank, target_score - honest_mean


def _mean_by_client(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby('Client', as_index=False)[cols].mean(numeric_only=True)


def _volatility(df: pd.DataFrame, col: str, transform: str = 'identity') -> pd.Series:
    x = pd.to_numeric(df[col], errors='coerce').astype(float)
    if transform == 'log':
        x = np.log(np.clip(x, 1e-12, None))
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError(f'Non-finite values in volatility source {col}')
    tmp = df[['Client', 'Round']].copy()
    tmp['_x'] = x
    tmp = tmp.sort_values(['Client', 'Round'])
    tmp['_abs_step'] = tmp.groupby('Client')['_x'].diff().abs()
    return tmp.groupby('Client')['_abs_step'].mean()


def _build_scenario(zf: zipfile.ZipFile, members: dict[str, str], folder: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prefix = folder.rstrip('/') + '/'
    required_names = [
        'phase5_fusion_metadata.json', 'historical_clean_rounds.csv', 'historical_poison_rounds.csv',
        'similarity_clean_rounds.csv', 'similarity_poison_rounds.csv',
        'cavc_diagnostics/cavc_v2_client_summary_min_support_100.csv',
        'cavc_diagnostics/cavc_v2_component_scores.csv',
        'cavc_diagnostics/cavc_v2_experiment_metadata.json',
    ]
    actual = {}
    for rel in required_names:
        key = prefix + rel
        if key not in members:
            raise ValueError(f'{folder}: missing archive member {rel}')
        actual[rel] = members[key]

    fm = _read_json(zf, actual['phase5_fusion_metadata.json'])
    cm = _read_json(zf, actual['cavc_diagnostics/cavc_v2_experiment_metadata.json'])
    inputs = fm.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError(f'{folder}: phase5_fusion_metadata.json has no inputs object')
    target = int(fm['malicious_client'])
    seed = int(fm['training_seed'])
    poison_seed = int(fm['poison_seed'])
    fraction = float(fm['poison_fraction_of_eligible_nonbenign'])
    if int(fm.get('clients', -1)) != 10 or int(fm.get('rounds_averaged', -1)) != 10:
        raise ValueError(f'{folder}: fusion metadata does not confirm 10 clients and 10 rounds')
    if fm.get('screening_enabled') is not False or fm.get('independent_test_used') is not False:
        raise ValueError(f'{folder}: fusion metadata indicates screening or independent-test use')
    if int(cm.get('partition_seed', -1)) != 43 or not math.isclose(float(cm.get('dirichlet_alpha', -1)), 0.5, abs_tol=1e-9):
        raise ValueError(f'{folder}: CAVC partition settings do not match alpha=0.5, partition seed=43')
    if int(cm.get('number_of_clients', cm.get('clients', -1))) not in (-1, 10):
        raise ValueError(f'{folder}: CAVC metadata has unexpected client count')
    expected_count = int(math.floor(int(cm.get('eligible_nonbenign_records', 0)) * fraction))
    if int(cm.get('poisoned_records', -1)) != expected_count:
        raise ValueError(f'{folder}: poisoned record count does not match floor(eligible records x configured fraction)')
    if int(cm.get('malicious_client_records', -1)) <= 0 or int(cm.get('eligible_nonbenign_records', -1)) <= 0:
        raise ValueError(f'{folder}: CAVC metadata has invalid client/eligible record counts')
    if int(cm.get('eligible_nonbenign_records', 0)) > int(cm.get('malicious_client_records', 0)):
        raise ValueError(f'{folder}: eligible non-Benign records exceed client records')
    if int(cm.get('validation_records', -1)) != 113116:
        raise ValueError(f'{folder}: unexpected validation record count')
    if int(cm.get('malicious_client_id', -1)) != target:
        raise ValueError(f'{folder}: CAVC target client does not match fusion metadata')
    if int(cm.get('training_seed', -1)) != seed or int(cm.get('poison_seed', -1)) != poison_seed:
        raise ValueError(f'{folder}: CAVC seeds do not match fusion metadata')
    if not math.isclose(float(cm.get('configured_fraction_of_eligible_nonbenign', -1)), fraction, abs_tol=1e-9):
        raise ValueError(f'{folder}: CAVC poisoning fraction does not match fusion metadata')
    if int(cm.get('global_checkpoint_round', -1)) != 10:
        raise ValueError(f'{folder}: CAVC diagnostic did not use round-10 checkpoint')
    if cm.get('screening_enabled') is not False or cm.get('aggregation_performed') is not False or cm.get('independent_test_set_used') is not False:
        raise ValueError(f'{folder}: unexpected screening, aggregation, or independent-test use')

    hclean = _read_csv(zf, actual['historical_clean_rounds.csv'])
    hpoison = _read_csv(zf, actual['historical_poison_rounds.csv'])
    sclean = _read_csv(zf, actual['similarity_clean_rounds.csv'])
    spoison = _read_csv(zf, actual['similarity_poison_rounds.csv'])
    for df, label, cond in [(hclean, 'historical clean', 'clean'), (hpoison, 'historical poisoned', 'poison'),
                            (sclean, 'similarity clean', 'clean'), (spoison, 'similarity poisoned', 'poison')]:
        _validate_rounds(df, f'{folder}/{label}', cond)
    for df, label in [(hpoison, 'historical poisoned'), (spoison, 'similarity poisoned')]:
        marked = set(df.loc[_as_bool_series(df['Is_Malicious']), 'Client'].astype(int))
        if marked != {target}:
            raise ValueError(f'{folder}/{label}: malicious client flag does not match metadata target {target}')
    for df, label in [(hclean, 'historical clean'), (hpoison, 'historical poisoned'),
                      (sclean, 'similarity clean'), (spoison, 'similarity poisoned')]:
        if 'Poisoned_Records' in df.columns:
            values = pd.to_numeric(df['Poisoned_Records'], errors='raise')
            expected = 0 if label.endswith('clean') else int(cm['poisoned_records'])
            observed = int(values.max())
            if observed != expected:
                raise ValueError(f'{folder}/{label}: expected max Poisoned_Records={expected}, got {observed}')

    # The CAVC summary intentionally represents the selected target's paired
    # clean/poison candidate diagnostic at min_class_support=100.
    cavc_summary = _read_csv(zf, actual['cavc_diagnostics/cavc_v2_client_summary_min_support_100.csv'])
    _require_columns(cavc_summary, ['condition', 'client_id', 'is_malicious', 'attack_detection_contribution_score',
                                    'targeted_damage_penalty', 'benign_fpr_damage_penalty'], 'CAVC summary')
    if len(cavc_summary) != 20:
        raise ValueError(f'{folder}: CAVC summary must have 20 rows, found {len(cavc_summary)}')
    poison_condition = f'poison_client{target:02d}'
    cavc_poison = cavc_summary[cavc_summary['condition'].astype(str) == poison_condition].copy()
    if len(cavc_poison) != 10:
        # Also allow the common condition name without a zero-padded client id.
        cavc_poison = cavc_summary[cavc_summary['condition'].astype(str) == f'poison_client{target}'].copy()
    if len(cavc_poison) != 10:
        raise ValueError(f'{folder}: could not locate the 10 poisoned-condition CAVC rows')
    cavc_poison['Client'] = pd.to_numeric(cavc_poison['client_id'], errors='raise').astype(int)
    cavc_poison['Is_Malicious'] = _as_bool_series(cavc_poison['is_malicious'])
    if set(cavc_poison['Client']) != set(CLIENTS):
        raise ValueError(f'{folder}: CAVC does not contain clients 1–10')
    marked_cavc = set(cavc_poison.loc[cavc_poison['Is_Malicious'], 'Client'].astype(int))
    if marked_cavc != {target}:
        raise ValueError(f'{folder}: CAVC malicious-client labels {marked_cavc} do not match target {target}')

    # Average the round-wise historical and similarity data for static
    # component comparisons, preserving round-wise volatility as separate features.
    hcols = ['Update_Similarity', 'Temporal_Consistency', 'Historical_Trust_Before',
             'Current_Evidence', 'Combined_Trust', 'Historical_Trust_After']
    scols = ['pairwise_direction_score', 'direction_score_to_reference', 'magnitude_agreement_score',
             'relative_magnitude', 'reference_distance_agreement_score', 'relative_reference_distance']
    _require_columns(hpoison, ['Round', 'Client'] + hcols, 'historical poisoned')
    _require_columns(hclean, ['Round', 'Client'] + hcols, 'historical clean')
    _require_columns(spoison, ['Round', 'Client'] + scols, 'similarity poisoned')
    _require_columns(sclean, ['Round', 'Client'] + scols, 'similarity clean')

    hmean = _mean_by_client(hpoison, hcols)
    smean = _mean_by_client(spoison, scols)
    frame = hmean.merge(smean, on='Client', validate='one_to_one')
    frame = frame.merge(cavc_poison[['Client', 'attack_detection_contribution_score', 'targeted_damage_penalty',
                                     'benign_fpr_damage_penalty']], on='Client', validate='one_to_one')
    frame['Is_Malicious'] = (frame['Client'] == target)

    # Reproduce the three V1 groups for a direct comparator.
    frame['v1_sim_pair'] = _percentile(frame['pairwise_direction_score'], False)
    frame['v1_sim_refdir'] = _percentile(frame['direction_score_to_reference'], False)
    frame['v1_sim_mag_agree'] = _percentile(frame['magnitude_agreement_score'], False)
    frame['v1_sim_refdist_agree'] = _percentile(frame['reference_distance_agreement_score'], False)
    frame['v1_sim_relmag'] = _percentile(np.abs(np.log(np.clip(frame['relative_magnitude'], 1e-12, None))), True)
    frame['v1_sim_relrefdist'] = _percentile(np.abs(np.log(np.clip(frame['relative_reference_distance'], 1e-12, None))), True)
    frame['V1_Similarity_Group'] = frame[['v1_sim_pair', 'v1_sim_refdir', 'v1_sim_mag_agree',
                                         'v1_sim_refdist_agree', 'v1_sim_relmag', 'v1_sim_relrefdist']].mean(axis=1)
    frame['V1_Historical_Group'] = _percentile(frame['Combined_Trust'], False)
    frame['v1_cavc_attack'] = _percentile(frame['attack_detection_contribution_score'], False)
    frame['v1_cavc_target'] = _percentile(frame['targeted_damage_penalty'], True)
    frame['v1_cavc_benign'] = _percentile(frame['benign_fpr_damage_penalty'], True)
    frame['V1_CAVC_Group'] = frame[['v1_cavc_attack', 'v1_cavc_target', 'v1_cavc_benign']].mean(axis=1)
    frame['V1_Equal_Fusion'] = frame[['V1_Similarity_Group', 'V1_Historical_Group', 'V1_CAVC_Group']].mean(axis=1)

    # V1 non-overlapping diagnostic candidate:
    # 1) direction-only cross-client anomaly;
    # 2) within-client round-to-round instability;
    # CAVC remains independent and is not fused into CCTRT.
    frame['Direction_Anomaly_Group'] = frame[['v1_sim_pair', 'v1_sim_refdir']].mean(axis=1)
    vol_features = {
        'Vol_Log_Relative_Magnitude': _volatility(spoison, 'relative_magnitude', 'log'),
        'Vol_Magnitude_Agreement': _volatility(spoison, 'magnitude_agreement_score'),
        'Vol_Reference_Distance_Agreement': _volatility(spoison, 'reference_distance_agreement_score'),
        'Vol_Historical_Combined_Trust': _volatility(hpoison, 'Combined_Trust'),
    }
    for name, series in vol_features.items():
        frame = frame.merge(series.rename(name), left_on='Client', right_index=True, validate='one_to_one')
        frame[name + '_Percentile'] = _percentile(frame[name], True)
    vol_cols = [name + '_Percentile' for name in vol_features]
    frame['Temporal_Instability_Group'] = frame[vol_cols].mean(axis=1)
    frame['CCTRT_V1_Diagnostic_Score'] = 0.5 * frame['Direction_Anomaly_Group'] + 0.5 * frame['Temporal_Instability_Group']

    # Historical memory-only comparator: use the previous EMA state, not V1's
    # Combined_Trust, which already contains current-round similarity evidence.
    frame['Historical_Memory_Only_Score'] = _percentile(frame['Historical_Trust_Before'], False)

    # Paired clean-vs-poison clean-reference deviation is an oracle diagnostic.
    # It must never be treated as a deployable detection feature.
    hclean_mean = _mean_by_client(hclean, hcols).set_index('Client')
    hpoison_mean = _mean_by_client(hpoison, hcols).set_index('Client')
    sclean_mean = _mean_by_client(sclean, scols).set_index('Client')
    spoison_mean = _mean_by_client(spoison, scols).set_index('Client')
    paired_sources = {
        'Paired_Abs_Historical_Combined_Trust': (hpoison_mean['Combined_Trust'] - hclean_mean['Combined_Trust']).abs(),
        'Paired_Abs_Historical_Update_Similarity': (hpoison_mean['Update_Similarity'] - hclean_mean['Update_Similarity']).abs(),
        'Paired_Abs_Similarity_Pairwise_Direction': (spoison_mean['pairwise_direction_score'] - sclean_mean['pairwise_direction_score']).abs(),
        'Paired_Abs_Log_Relative_Magnitude': (np.log(np.clip(spoison_mean['relative_magnitude'], 1e-12, None)) -
                                             np.log(np.clip(sclean_mean['relative_magnitude'], 1e-12, None))).abs(),
        'Paired_Abs_Reference_Distance_Agreement': (spoison_mean['reference_distance_agreement_score'] -
                                                    sclean_mean['reference_distance_agreement_score']).abs(),
    }
    paired = pd.DataFrame({'Client': CLIENTS})
    for name, values in paired_sources.items():
        paired[name] = values.reindex(CLIENTS).to_numpy()
        paired[name + '_Percentile'] = _percentile(paired[name], True)
    paired_pct = [n + '_Percentile' for n in paired_sources]
    paired['Paired_Oracle_Deviation_Score'] = paired[paired_pct].mean(axis=1)
    paired['Is_Malicious'] = paired['Client'].eq(target)

    # Attach all round-level volatility inputs and flags for auditability.
    ts = spoison[['Round', 'Client', 'Is_Malicious', 'pairwise_direction_score',
                  'direction_score_to_reference', 'relative_magnitude', 'magnitude_agreement_score',
                  'relative_reference_distance', 'reference_distance_agreement_score']].merge(
        hpoison[['Round', 'Client', 'Temporal_Consistency', 'Historical_Trust_Before', 'Combined_Trust',
                 'Historical_Trust_After']], on=['Round', 'Client'], validate='one_to_one')
    ts['Log_Relative_Magnitude'] = np.log(np.clip(ts['relative_magnitude'], 1e-12, None))
    ts = ts.sort_values(['Client', 'Round'])
    for col in ['Log_Relative_Magnitude', 'magnitude_agreement_score',
                'reference_distance_agreement_score', 'Combined_Trust']:
        ts['Abs_Step_' + col] = ts.groupby('Client')[col].diff().abs()
    ts['Scenario'] = folder
    ts['Target_Client'] = target
    ts['Training_Seed'] = seed
    ts['Poison_Fraction'] = fraction

    # Per-scenario metrics; one positive client among ten.
    labels = frame['Client'].eq(target).astype(int)
    metrics = {
        'V1_Similarity_Group': frame['V1_Similarity_Group'],
        'V1_Historical_Group': frame['V1_Historical_Group'],
        'V1_CAVC_Group': frame['V1_CAVC_Group'],
        'V1_Equal_Fusion': frame['V1_Equal_Fusion'],
        'Historical_Memory_Only': frame['Historical_Memory_Only_Score'],
        'Direction_Anomaly_Group': frame['Direction_Anomaly_Group'],
        'Temporal_Instability_Group': frame['Temporal_Instability_Group'],
        'CCTRT_V1_Diagnostic': frame['CCTRT_V1_Diagnostic_Score'],
        'Paired_Oracle_Deviation': paired['Paired_Oracle_Deviation_Score'],
    }
    result = {
        'Scenario': folder,
        'Target_Client': target,
        'Training_Seed': seed,
        'Poison_Seed': poison_seed,
        'Poison_Fraction_Eligible': fraction,
        'Poisoned_Records': int(cm['poisoned_records']),
        'Eligible_NonBenign_Records': int(cm['eligible_nonbenign_records']),
        'CAVC_Checkpoint': cm.get('global_checkpoint'),
        'Validation_Records': int(cm.get('validation_records', -1)),
    }
    for name, series in metrics.items():
        temp = frame[['Client']].copy()
        if name == 'Paired_Oracle_Deviation':
            temp = paired[['Client']].copy()
        temp['score'] = np.asarray(series, dtype=float)
        auc, ap, rank, gap = _auc_rank(temp.rename(columns={'score': 'score'}), 'score', target)
        prefix_name = name.replace(' ', '_')
        result[prefix_name + '_ROC_AUC'] = auc
        result[prefix_name + '_AP'] = ap
        result[prefix_name + '_Rank'] = rank
        result[prefix_name + '_Score_Gap_vs_Honest_Mean'] = gap

    frame.insert(0, 'Scenario', folder)
    frame.insert(1, 'Training_Seed', seed)
    frame.insert(2, 'Poison_Fraction', fraction)
    frame['Fusion_Rank'] = frame['V1_Equal_Fusion'].rank(method='average', ascending=False)
    frame['CCTRT_Rank'] = frame['CCTRT_V1_Diagnostic_Score'].rank(method='average', ascending=False)
    frame['Temporal_Group_Rank'] = frame['Temporal_Instability_Group'].rank(method='average', ascending=False)
    frame['Direction_Group_Rank'] = frame['Direction_Anomaly_Group'].rank(method='average', ascending=False)
    paired.insert(0, 'Scenario', folder)
    paired.insert(1, 'Training_Seed', seed)
    paired.insert(2, 'Poison_Fraction', fraction)
    return frame, ts, paired, result


def run(input_zip: Path, output_dir: Path) -> None:
    if not input_zip.exists():
        raise FileNotFoundError(f'Input ZIP not found: {input_zip}')
    with zipfile.ZipFile(input_zip, 'r') as zf:
        members = _member_map(zf)
        partition_key = 'partition_metadata_seed43.json'
        if partition_key not in members:
            raise ValueError('Archive is missing partition_metadata_seed43.json')
        partition_meta = _read_json(zf, members[partition_key])
        if not math.isclose(float(partition_meta.get('dirichlet_alpha', -1)), 0.5, abs_tol=1e-9):
            raise ValueError('Partition metadata does not confirm Dirichlet alpha=0.5')
        if int(partition_meta.get('random_seed', -1)) != 43:
            raise ValueError('Partition metadata does not confirm partition seed=43')
        if int(partition_meta.get('number_of_clients', -1)) != 10:
            raise ValueError('Partition metadata does not confirm 10 clients')
        folders = sorted({p.split('/')[0] for p in members if p.startswith('phase5_fusion_') and '/' in p})
        if len(folders) != EXPECTED_SCENARIOS:
            raise ValueError(f'Expected {EXPECTED_SCENARIOS} phase5_fusion folders; found {len(folders)}')
        all_scores, all_timeseries, all_paired, summaries = [], [], [], []
        for folder in folders:
            scores, timeseries, paired, summary = _build_scenario(zf, members, folder)
            all_scores.append(scores)
            all_timeseries.append(timeseries)
            all_paired.append(paired)
            summaries.append(summary)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory already contains files: {output_dir}. Choose a new output directory/tag.')
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_df = pd.concat(all_scores, ignore_index=True)
    ts_df = pd.concat(all_timeseries, ignore_index=True)
    paired_df = pd.concat(all_paired, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    scores_df.to_csv(output_dir / 'phase7_client_scores.csv', index=False)
    ts_df.to_csv(output_dir / 'phase7_round_level_features.csv', index=False)
    paired_df.to_csv(output_dir / 'phase7_paired_oracle_diagnostics.csv', index=False)
    summary_df.to_csv(output_dir / 'phase7_scenario_metrics.csv', index=False)

    auc_columns = [c for c in summary_df.columns if c.endswith('_ROC_AUC')]
    aggregate_rows = []
    for col in auc_columns:
        vals = pd.to_numeric(summary_df[col], errors='raise')
        aggregate_rows.append({
            'Diagnostic': col.removesuffix('_ROC_AUC'),
            'Mean_ROC_AUC_12_Scenarios': float(vals.mean()),
            'Std_ROC_AUC_12_Scenarios': float(vals.std(ddof=1)),
            'Min_ROC_AUC': float(vals.min()),
            'Max_ROC_AUC': float(vals.max()),
            'Scenarios_Rank_Top3': int((summary_df[col.replace('_ROC_AUC', '_Rank')] <= 3).sum()),
        })
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values('Mean_ROC_AUC_12_Scenarios', ascending=False)
    aggregate_df.to_csv(output_dir / 'phase7_diagnostic_aggregate_comparison.csv', index=False)

    # Check repeated clean controls are identical within each training seed.
    integrity = []
    for seed in sorted(summary_df['Training_Seed'].unique()):
        ds = [d for d in folders if f'_s{seed}_' in d]
        for file_name in ['historical_clean_rounds.csv', 'similarity_clean_rounds.csv']:
            raw_hashes = []
            for folder in ds:
                member = folder + '/' + file_name
                raw_hashes.append(zlib_sha256_from_zip(input_zip, member))
            integrity.append(f'Seed {seed} {file_name}: {len(set(raw_hashes))} unique clean-control file hash(es) across {len(ds)} scenarios')

    # Validate CAVC checkpoint provenance: seed-43 and seed-44 scenarios use
    # their corresponding clean-control checkpoints, rather than mixing seeds.
    for _, row in summary_df.iterrows():
        ck = str(row['CAVC_Checkpoint']).lower()
        expected_seed = str(int(row['Training_Seed']))
        if f'clean_s{expected_seed}' not in ck and f'seed{expected_seed}' not in ck:
            raise ValueError(f"{row['Scenario']}: CAVC checkpoint path does not visibly match training seed {expected_seed}: {row['CAVC_Checkpoint']}")

    lines = [
        'PHASE 7 — CCTRT V1 DIAGNOSTIC REPORT',
        '====================================',
        '',
        f'Scenarios validated: {len(summary_df)}',
        f'Client-round rows (poisoned runs): {len(ts_df)}',
        f'Client-score rows: {len(scores_df)}',
        f"Partition: {partition_meta.get('dataset', 'unknown')}, {partition_meta.get('partition_method', 'unknown')}, alpha={partition_meta.get('dirichlet_alpha')}, partition seed={partition_meta.get('random_seed')}",
        'Training seeds: ' + ', '.join(map(str, sorted(summary_df['Training_Seed'].unique()))),
        '',
        'MAIN DIAGNOSTIC COMPARISON (mean AUC is descriptive across 12 scenario-specific ten-client evaluations)',
    ]
    for _, row in aggregate_df.iterrows():
        lines.append(f"- {row['Diagnostic']}: mean AUC={row['Mean_ROC_AUC_12_Scenarios']:.4f}; range={row['Min_ROC_AUC']:.4f}–{row['Max_ROC_AUC']:.4f}; top-3 ranks={int(row['Scenarios_Rank_Top3'])}/12")
    lines += [
        '',
        'INTERPRETATION AND LIMITATIONS',
        '- V1_Equal_Fusion is reconstructed from the archived raw inputs using the V1 direction conventions.',
        '- CCTRT_V1_Diagnostic is a fixed 50/50 combination of a direction-anomaly group and a within-client temporal-instability group.',
        '- Temporal instability is the mean absolute round-to-round change in log relative magnitude, magnitude agreement, reference-distance agreement, and Historical Combined_Trust.',
        '- The CCTRT diagnostic is retrospective over the 10 observed rounds; it is not an online screening implementation.',
        '- Paired_Oracle_Deviation compares poisoned-client behavior with the matched clean run for each same-seed scenario. It is an offline counterfactual diagnostic and is NOT deployable because a live server does not possess the same client-round clean counterfactual.',
        '- CAVC remains a separate candidate-model utility diagnostic and is not included in CCTRT V1.',
        '- ROC-AUC and AP are computed separately within each scenario from 10 clients and one positive client. The 12 scenario summaries are not 120 independent client observations.',
        '- Results cover three attacker identities, two poisoning strengths, two training seeds, and one fixed non-IID partition. They do not establish cross-partition or cross-dataset generalization.',
        '- No client screening, model aggregation, model training, checkpoint modification, or independent-test evaluation was performed.',
        '- No threshold or weights are recommended for deployment.',
        '',
        'CLEAN CONTROL FILE CHECKS',
        *integrity,
        '',
        'Recommended next validation: evaluate a prospective online version using only past rounds (no matched clean reference), include attack onset after an initial honest warm-up and from round 1, add more attacker identities and partition seeds, then evaluate false-positive rates before any screening experiment.',
    ]
    (output_dir / 'phase7_summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\nPHASE 7 CCTRT V1 DIAGNOSTIC COMPLETED')
    print('Scenarios validated:', len(summary_df))
    print('Output directory:', output_dir)
    print('\nDiagnostic comparison:')
    print(aggregate_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nScreening enabled: NO')
    print('Aggregation modified: NO')
    print('Independent test used: NO')


def zlib_sha256_from_zip(zip_path: Path, member_name: str) -> str:
    import hashlib
    with zipfile.ZipFile(zip_path, 'r') as zf:
        member_map = _member_map(zf)
        key = member_name
        if key not in member_map:
            raise ValueError(f'Missing clean control in ZIP: {member_name}')
        return hashlib.sha256(zf.read(member_map[key])).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Offline Phase 7 CCTRT V1 diagnostic; no training or screening.')
    parser.add_argument('--input-zip', default='phase7_raw_round_inputs.zip', help='ZIP produced by the Phase 7 collection script')
    parser.add_argument('--output-dir', default='results/phase7_cctrt_v1_diagnostic_s43_s44', help='New output folder; existing nonempty folder is not overwritten')
    args = parser.parse_args()
    run(Path(args.input_zip).resolve(), Path(args.output_dir).resolve())


if __name__ == '__main__':
    main()
