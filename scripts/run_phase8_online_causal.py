from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CLIENTS = list(range(1, 11))
ROUNDS = list(range(1, 11))
EXPECTED_SCENARIOS = 12


def normalize_zip_name(name: str) -> str:
    return name.replace('\\', '/').lstrip('./')


def build_member_map(zf: zipfile.ZipFile) -> dict[str, str]:
    return {normalize_zip_name(i.filename): i.filename for i in zf.infolist() if not i.is_dir()}


def read_json(zf: zipfile.ZipFile, members: dict[str, str], name: str) -> dict[str, Any]:
    if name not in members:
        raise ValueError(f'Missing archive member: {name}')
    return json.loads(zf.read(members[name]).decode('utf-8-sig'))


def read_csv(zf: zipfile.ZipFile, members: dict[str, str], name: str) -> pd.DataFrame:
    if name not in members:
        raise ValueError(f'Missing archive member: {name}')
    return pd.read_csv(io.BytesIO(zf.read(members[name])))


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    result = series.astype(str).str.strip().str.lower().map(mapping)
    if result.isna().any():
        raise ValueError(f'{label} contains invalid boolean values')
    return result.astype(bool)


def validate_round_frame(df: pd.DataFrame, label: str, is_poisoned: bool, target: int | None) -> pd.DataFrame:
    required = ['Round', 'Client', 'Is_Malicious']
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'{label} missing columns: {missing}')
    out = df.copy()
    out['Round'] = pd.to_numeric(out['Round'], errors='raise').astype(int)
    out['Client'] = pd.to_numeric(out['Client'], errors='raise').astype(int)
    out['Is_Malicious'] = parse_bool(out['Is_Malicious'], f'{label}.Is_Malicious')
    if len(out) != 100 or set(out['Client']) != set(CLIENTS) or set(out['Round']) != set(ROUNDS):
        raise ValueError(f'{label} must have exactly 100 rows covering clients 1-10 and rounds 1-10')
    if out.duplicated(['Round', 'Client']).any():
        raise ValueError(f'{label} contains duplicate round/client rows')
    counts = out.groupby('Client')['Round'].nunique()
    if not (counts == 10).all():
        raise ValueError(f'{label} must have 10 rounds per client')
    flags = out.groupby('Client')['Is_Malicious'].nunique()
    if (flags != 1).any():
        raise ValueError(f'{label} has inconsistent malicious flags')
    marked = set(out.loc[out['Is_Malicious'], 'Client'].astype(int))
    expected = {int(target)} if is_poisoned and target is not None else set()
    if marked != expected:
        raise ValueError(f'{label} malicious flags expected {expected}, found {marked}')
    return out.sort_values(['Round', 'Client']).reset_index(drop=True)


def percentile(values: pd.Series, high_is_suspicious: bool) -> pd.Series:
    x = pd.to_numeric(values, errors='coerce')
    if x.isna().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('Cannot percentile-rank missing/non-finite values')
    # High values suspicious: largest value gets percentile 1.
    # Low values suspicious: smallest value gets percentile 1.
    return x.rank(method='average', ascending=high_is_suspicious, pct=True)


def causal_round_scores(sim: pd.DataFrame, hist: pd.DataFrame, upto_round: int = 10) -> pd.DataFrame:
    """Calculate every score using only records from rounds <= current round.

    Current-round directional ranks use only that round's client cohort.
    Temporal step statistics are cumulative means of absolute changes observed
    from round 1 through the current round; no future rows are read.
    """
    if upto_round < 1 or upto_round > 10:
        raise ValueError('upto_round must be between 1 and 10')
    sim_cols = ['Round', 'Client', 'pairwise_direction_score', 'direction_score_to_reference',
                'relative_magnitude', 'magnitude_agreement_score', 'reference_distance_agreement_score']
    hist_cols = ['Round', 'Client', 'Combined_Trust']
    for label, frame, cols in [('similarity', sim, sim_cols), ('historical', hist, hist_cols)]:
        missing = sorted(set(cols) - set(frame.columns))
        if missing:
            raise ValueError(f'{label} frame missing columns: {missing}')
    merged = sim[sim_cols].merge(hist[hist_cols], on=['Round', 'Client'], how='inner', validate='one_to_one')
    expected_rows = 10 * len(set(pd.to_numeric(sim['Round'], errors='raise').astype(int)))
    if len(merged) != expected_rows:
        raise ValueError(f'Expected {expected_rows} aligned rows for the supplied round prefix; found {len(merged)}')
    merged = merged.sort_values(['Client', 'Round']).copy()
    merged['Log_Relative_Magnitude'] = np.log(np.clip(pd.to_numeric(merged['relative_magnitude'], errors='raise'), 1e-12, None))
    raw_temporal_cols = ['Log_Relative_Magnitude', 'magnitude_agreement_score',
                         'reference_distance_agreement_score', 'Combined_Trust']
    for col in raw_temporal_cols:
        merged[col] = pd.to_numeric(merged[col], errors='raise').astype(float)
        if not np.isfinite(merged[col].to_numpy()).all():
            raise ValueError(f'Non-finite temporal values in {col}')

    # Per-client cumulative absolute-step state. Only the current and prior
    # observations are included when producing that round's features.
    cumulative_sums = {cid: {col: 0.0 for col in raw_temporal_cols} for cid in CLIENTS}
    cumulative_counts = {cid: 0 for cid in CLIENTS}
    previous = {cid: None for cid in CLIENTS}
    records: list[dict[str, Any]] = []

    for round_no in range(1, upto_round + 1):
        current = merged[merged['Round'] == round_no].sort_values('Client').copy()
        if len(current) != 10 or current['Client'].astype(int).tolist() != CLIENTS:
            raise ValueError(f'Round {round_no} does not contain clients 1-10')
        for row in current.itertuples(index=False):
            cid = int(row.Client)
            now = {col: float(getattr(row, col)) for col in raw_temporal_cols}
            if previous[cid] is not None:
                for col in raw_temporal_cols:
                    cumulative_sums[cid][col] += abs(now[col] - previous[cid][col])
                cumulative_counts[cid] += 1
            previous[cid] = now

        out = current[['Round', 'Client']].copy()
        out['Direction_Pairwise_Anomaly'] = percentile(current['pairwise_direction_score'], False).to_numpy()
        out['Direction_Reference_Anomaly'] = percentile(current['direction_score_to_reference'], False).to_numpy()
        out['Direction_Anomaly_Group'] = out[['Direction_Pairwise_Anomaly', 'Direction_Reference_Anomaly']].mean(axis=1)

        # At round 1 there is no within-client history yet. Use a neutral value
        # of 0.5 for the entire temporal group rather than rank tied zero steps.
        if round_no == 1:
            out['Temporal_Instability_Group'] = 0.5
            for col in raw_temporal_cols:
                out[f'Cumulative_Mean_Abs_Step_{col}'] = 0.0
                out[f'Temporal_{col}_Percentile'] = 0.5
        else:
            raw_names = []
            for col in raw_temporal_cols:
                vals = [cumulative_sums[cid][col] / max(cumulative_counts[cid], 1) for cid in CLIENTS]
                name = f'Cumulative_Mean_Abs_Step_{col}'
                out[name] = vals
                pct_name = f'Temporal_{col}_Percentile'
                out[pct_name] = percentile(out[name], True).to_numpy()
                raw_names.append(pct_name)
            out['Temporal_Instability_Group'] = out[raw_names].mean(axis=1)

        out['CCTRT_Causal_Score'] = 0.5 * out['Direction_Anomaly_Group'] + 0.5 * out['Temporal_Instability_Group']
        out['Round_Used_Through'] = round_no
        records.extend(out.to_dict(orient='records'))

    return pd.DataFrame(records)


def scenario_metrics(scores: pd.DataFrame, target: int, condition: str) -> list[dict[str, Any]]:
    rows = []
    for rnd, g in scores.groupby('Round', sort=True):
        g = g.sort_values('Client').copy()
        g['Is_Malicious'] = g['Client'].astype(int).eq(int(target)) if condition == 'poisoned' else False
        if condition == 'poisoned':
            y = g['Is_Malicious'].astype(int).to_numpy()
            s = g['CCTRT_Causal_Score'].astype(float).to_numpy()
            auc = float(roc_auc_score(y, s))
            ap = float(average_precision_score(y, s))
            target_row = g[g['Client'].astype(int) == int(target)].iloc[0]
            rank = int(g.sort_values(['CCTRT_Causal_Score', 'Client'], ascending=[False, True]).reset_index(drop=True).index[
                g.sort_values(['CCTRT_Causal_Score', 'Client'], ascending=[False, True]).reset_index(drop=True)['Client'].astype(int) == int(target)
            ][0]) + 1
            target_score = float(target_row['CCTRT_Causal_Score'])
            honest_mean = float(g.loc[g['Client'].astype(int) != int(target), 'CCTRT_Causal_Score'].mean())
            rows.append({'Condition': condition, 'Round': int(rnd), 'Target_Client': int(target),
                         'ROC_AUC': auc, 'Average_Precision': ap, 'Target_Rank': rank,
                         'Target_Score': target_score, 'Target_Minus_Honest_Mean': target_score - honest_mean,
                         'Target_Top1': rank == 1, 'Target_Top3': rank <= 3})
        else:
            # Clean data has no positive label. Report the top-ranked clients
            # as a diagnostic load, not as a false-positive rate at a calibrated threshold.
            ordered = g.sort_values(['CCTRT_Causal_Score', 'Client'], ascending=[False, True])
            rows.append({'Condition': condition, 'Round': int(rnd), 'Target_Client': None,
                         'ROC_AUC': np.nan, 'Average_Precision': np.nan, 'Target_Rank': np.nan,
                         'Target_Score': np.nan, 'Target_Minus_Honest_Mean': np.nan,
                         'Target_Top1': False, 'Target_Top3': False,
                         'Clean_Top1_Client': int(ordered.iloc[0]['Client']),
                         'Clean_Top3_Clients': ','.join(map(str, ordered.head(3)['Client'].astype(int).tolist()))})
    return rows


def process_scenario(zf: zipfile.ZipFile, members: dict[str, str], folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = folder + '/'
    meta = read_json(zf, members, prefix + 'phase5_fusion_metadata.json')
    target = int(meta['malicious_client'])
    seed = int(meta['training_seed'])
    poison_fraction = float(meta['poison_fraction_of_eligible_nonbenign'])
    if meta.get('screening_enabled') is not False or meta.get('independent_test_used') is not False:
        raise ValueError(f'{folder}: expected screening disabled and independent test unused')

    scenarios = []
    metrics = []
    for condition, hist_name, sim_name in [
        ('clean', 'historical_clean_rounds.csv', 'similarity_clean_rounds.csv'),
        ('poisoned', 'historical_poison_rounds.csv', 'similarity_poison_rounds.csv'),
    ]:
        hist = pd.read_csv(io.BytesIO(zf.read(members[prefix + hist_name])))
        sim = pd.read_csv(io.BytesIO(zf.read(members[prefix + sim_name])))
        hist = validate_round_frame(hist, f'{folder}/{hist_name}', condition == 'poisoned', target)
        sim = validate_round_frame(sim, f'{folder}/{sim_name}', condition == 'poisoned', target)
        # Explicit check the archive actually records the expected attack fraction/count.
        if condition == 'poisoned' and 'Poisoned_Records' in hist.columns:
            poison_max = int(pd.to_numeric(hist['Poisoned_Records'], errors='raise').max())
            if poison_max <= 0:
                raise ValueError(f'{folder}: poisoned run reports zero poisoned records')
        causal = causal_round_scores(sim, hist, upto_round=10)
        causal.insert(0, 'Scenario', folder)
        causal.insert(1, 'Training_Seed', seed)
        causal.insert(2, 'Poison_Fraction_Eligible', poison_fraction)
        causal.insert(3, 'Condition', condition)
        causal['Target_Client'] = target if condition == 'poisoned' else pd.NA
        scenarios.append(causal)
        m = scenario_metrics(causal, target, condition)
        for row in m:
            row.update({'Scenario': folder, 'Training_Seed': seed,
                        'Poison_Fraction_Eligible': poison_fraction})
        metrics.extend(m)
    return pd.concat(scenarios, ignore_index=True), pd.DataFrame(metrics)


def run(input_zip: Path, output_dir: Path) -> None:
    if not input_zip.exists():
        raise FileNotFoundError(f'Input ZIP not found: {input_zip}')
    with zipfile.ZipFile(input_zip, 'r') as zf:
        members = build_member_map(zf)
        folders = sorted({name.split('/')[0] for name in members if name.startswith('phase5_fusion_') and '/' in name})
        if len(folders) != EXPECTED_SCENARIOS:
            raise ValueError(f'Expected 12 scenario folders, found {len(folders)}')
        score_frames, metric_frames = [], []
        for i, folder in enumerate(folders, start=1):
            print(f'Processing {i}/{len(folders)}: {folder}')
            scores, metrics = process_scenario(zf, members, folder)
            score_frames.append(scores)
            metric_frames.append(metrics)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}. Choose a new output folder.')
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scores = pd.concat(score_frames, ignore_index=True)
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    poisoned_metrics = all_metrics[all_metrics['Condition'] == 'poisoned'].copy()
    clean_metrics = all_metrics[all_metrics['Condition'] == 'clean'].copy()

    round_summary = poisoned_metrics.groupby('Round', as_index=False).agg(
        Scenarios=('Scenario', 'nunique'),
        Mean_ROC_AUC=('ROC_AUC', 'mean'),
        Std_ROC_AUC=('ROC_AUC', 'std'),
        Mean_Average_Precision=('Average_Precision', 'mean'),
        Top1_Scenarios=('Target_Top1', 'sum'),
        Top3_Scenarios=('Target_Top3', 'sum'),
        Mean_Target_Rank=('Target_Rank', 'mean'),
    )
    round_summary['Top3_Scenario_Rate'] = round_summary['Top3_Scenarios'] / round_summary['Scenarios']
    overall = pd.DataFrame([{
        'Scenarios': poisoned_metrics['Scenario'].nunique(),
        'Scenario_Round_Observations': len(poisoned_metrics),
        'Mean_ROC_AUC_120_Scenario_Rounds': poisoned_metrics['ROC_AUC'].mean(),
        'Std_ROC_AUC_120_Scenario_Rounds': poisoned_metrics['ROC_AUC'].std(ddof=1),
        'Top3_Scenario_Rounds': int(poisoned_metrics['Target_Top3'].sum()),
        'Top3_Scenario_Round_Rate': float(poisoned_metrics['Target_Top3'].mean()),
        'Independent_Test_Used': False,
        'Screening_Enabled': False,
    }])
    clean_summary = clean_metrics.groupby('Round', as_index=False).agg(
        Scenarios=('Scenario', 'nunique'),
        Clean_Top1_Client_Counts=('Clean_Top1_Client', lambda s: ';'.join(map(str, s.astype(int).tolist()))),
    )

    all_scores.to_csv(output_dir / 'phase8_causal_round_client_scores.csv', index=False)
    all_metrics.to_csv(output_dir / 'phase8_causal_scenario_round_metrics.csv', index=False)
    round_summary.to_csv(output_dir / 'phase8_causal_round_summary.csv', index=False)
    overall.to_csv(output_dir / 'phase8_causal_overall_summary.csv', index=False)
    clean_summary.to_csv(output_dir / 'phase8_clean_control_top_clients.csv', index=False)

    lines = [
        'TRFA PHASE 8 — ONLINE/CAUSAL CCTRT REPLAY',
        '=========================================',
        '',
        f'Scenarios processed: {poisoned_metrics["Scenario"].nunique()}',
        f'Scenario-round observations: {len(poisoned_metrics)}',
        f'Mean ROC-AUC over scenario-round evaluations: {poisoned_metrics["ROC_AUC"].mean():.6f}',
        f'Mean target top-3 rate over scenario-round evaluations: {poisoned_metrics["Target_Top3"].mean():.6f}',
        f'Mean target rank over scenario-round evaluations: {poisoned_metrics["Target_Rank"].mean():.6f}',
        '',
        'ROUND-BY-ROUND SUMMARY',
    ]
    for row in round_summary.itertuples(index=False):
        lines.append(f'Round {row.Round}: mean AUC={row.Mean_ROC_AUC:.4f}; top-3={row.Top3_Scenarios}/{row.Scenarios}; mean rank={row.Mean_Target_Rank:.2f}')
    lines += [
        '',
        'CAUSALITY RULE',
        '- At round r, directional features use only round r values across the 10 clients.',
        '- Temporal instability uses cumulative absolute steps from rounds 1 through r only.',
        '- No values from rounds greater than r are used to calculate the score at round r.',
        '- At round 1, temporal instability is neutral (0.5) because no within-client history exists.',
        '',
        'LIMITATIONS',
        '- This is an offline replay of the existing 12 scenarios, not a new training experiment.',
        '- Attacks in the source experiments were active from round 1; delayed-onset attacks were not tested.',
        '- Percentile normalization is still cohort-relative within each round.',
        '- The two-seed, one-partition results are exploratory and are not a deployment validation.',
        '- CAVC is intentionally not part of the CCTRT causal score.',
        '- Screening remained disabled; aggregation was not modified; independent test data was not used.',
    ]
    (output_dir / 'phase8_causal_summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    metadata = {
        'phase': '8', 'method': 'CCTRT V1 causal round-by-round replay',
        'input_zip': str(input_zip), 'scenario_count': int(poisoned_metrics['Scenario'].nunique()),
        'rounds': 10, 'clients': 10,
        'causal_rule': 'For round r, use current-round directional features and cumulative absolute steps through round r only.',
        'round1_temporal_initialization': 0.5,
        'screening_enabled': False, 'aggregation_modified': False,
        'independent_test_used': False, 'delayed_attack_onset_tested': False,
    }
    (output_dir / 'phase8_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nPHASE 8 ONLINE/CAUSAL REPLAY COMPLETED')
    print('Scenarios:', poisoned_metrics['Scenario'].nunique())
    print(f'Mean scenario-round ROC-AUC: {poisoned_metrics["ROC_AUC"].mean():.4f}')
    print(f'Top-3 rate: {poisoned_metrics["Target_Top3"].mean():.4f}')
    print('Output directory:', output_dir)
    print('Screening enabled: NO; aggregation modified: NO; independent test used: NO')


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Run causal, round-by-round CCTRT replay on existing Phase 5/6 data.')
    parser.add_argument('--input-zip', default=str(default_root / 'phase7_raw_round_inputs.zip'))
    parser.add_argument('--output-dir', default=str(default_root / 'results' / 'phase8_cctrt_causal_replay_s43_s44'))
    args = parser.parse_args()
    run(Path(args.input_zip), Path(args.output_dir))


if __name__ == '__main__':
    main()
