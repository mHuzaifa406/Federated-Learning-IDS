# Calibrated CCTRT / TRFA — two-day pilot runbook

## What this experiment changes

This branch adds a clean-reference calibration layer around the existing causal CCTRT signal
`Cumulative_Mean_Abs_Step_Combined_Trust`. A clean profile is stored for each client and
communication round. The observed value is standardized against that profile. A client is
excluded only after the calibrated upper-tail alert persists for the configured number of
consecutive rounds. Accepted clients are aggregated using the existing sample-size-weighted
FedAvg implementation.

The detector is not a probability estimator. The default 97.5th percentile and two-round
persistence are prespecified pilot settings, not parameters optimized on the attack data.
The two-seed option is preliminary; use at least three distinct clean calibration seeds if
compute time permits.

## 1. Get the experiment branch

From PowerShell:

```powershell
cd E:\IoMT_FL_IDS_TRFA
git status --short
git fetch origin
git switch --track origin/experiment/calibrated-cctrt
```

If the branch already exists locally, use `git switch experiment/calibrated-cctrt`.
Do not discard any uncommitted local changes to switch branches.

## 2. Run code checks

```powershell
python -m compileall -q trfa scripts
python -m unittest scripts.test_cctrt_calibration -v
```

Both commands should complete without errors before starting the long FL runs.

## 3. Create two independent clean calibration runs

The existing repository includes a Phase 9 all-honest client-05 control with training seed 44:
`results\phase9_clean_control_client05_p0.2000_s44_ps2028_c05_p20_clean_s44`.

Create two more clean controls with distinct seeds. These runs do not poison any labels;
`--poison-fraction` records the paired attack setting in their metadata only.

```powershell
python -m scripts.run_phase9_clean_control --malicious-client 5 --poison-fraction 0.20 --training-seed 41 --poison-seed 2028 --output-tag cctrt_cal_seed41
python -m scripts.run_phase9_clean_control --malicious-client 5 --poison-fraction 0.20 --training-seed 42 --poison-seed 2028 --output-tag cctrt_cal_seed42
```

The expected output directories are:

- `results\phase9_clean_control_client05_p0.2000_s41_ps2028_cctrt_cal_seed41`
- `results\phase9_clean_control_client05_p0.2000_s42_ps2028_cctrt_cal_seed42`

Before proceeding, confirm the existing seed-44 clean-control folder exists. If it does not,
do not silently substitute another experiment.

## 4. Build the calibration file

```powershell
python -m scripts.build_cctrt_calibration `
  --clean-dir "results\phase9_clean_control_client05_p0.2000_s41_ps2028_cctrt_cal_seed41" `
  --clean-dir "results\phase9_clean_control_client05_p0.2000_s42_ps2028_cctrt_cal_seed42" `
  --output "results\cctrt_clean_calibration_v1.json"
```

The calibration builder checks that every run is all-honest, uses the same partition and
training configuration, has a distinct training seed, and contains all 10 clients across all
10 rounds. It uses leave-one-run-out clean residuals. The resulting JSON records its seeds,
threshold, scale and limitations.

## 5. Evaluate the held-out clean control first

```powershell
python -m scripts.evaluate_cctrt_calibration `
  --calibration "results\cctrt_clean_calibration_v1.json" `
  --score-csv "results\phase9_clean_control_client05_p0.2000_s44_ps2028_c05_p20_clean_s44\phase9_client_round_scores.csv" `
  --output-dir "results\cctrt_eval_clean_seed44"
```

Read `results\cctrt_eval_clean_seed44\calibrated_detection_summary.json`.
Do not proceed to claim a useful defense if the clean false-exclusion rate is high.

## 6. Run the screened attack with the same seed as the existing unscreened Phase 9 attack

```powershell
python -m scripts.run_phase9_delayed_onset `
  --malicious-client 5 `
  --poison-fraction 0.20 `
  --attack-start-round 6 `
  --training-seed 44 `
  --poison-seed 2028 `
  --output-tag calibrated_persistence2 `
  --calibration "results\cctrt_clean_calibration_v1.json" `
  --persistence 2
```

The expected output directory is:
`results\phase9_delayed_client05_p0.2000_start6_s44_ps2028_calibrated_persistence2`.

This run uses the calibrated screening decision during aggregation. If fewer than two clients
remain, the run stops rather than silently reverting to unfiltered FedAvg.

## 7. Evaluate the screened attack and compare

```powershell
python -m scripts.evaluate_cctrt_calibration `
  --calibration "results\cctrt_clean_calibration_v1.json" `
  --score-csv "results\phase9_delayed_client05_p0.2000_start6_s44_ps2028_calibrated_persistence2\phase9_client_round_scores.csv" `
  --output-dir "results\cctrt_eval_attack_seed44"
```

Compare these outputs with the existing unscreened run:
`results\phase9_delayed_client05_p0.2000_start6_s44_ps2028_c05_p20_start06_s44`.

Report at least:

- clean false-exclusion rate from the held-out clean control;
- target detection rate after round 6 and first detection delay;
- honest-client false-exclusion rate during the attack run;
- per-round ROC-AUC and average precision;
- validation macro-F1, balanced accuracy and accuracy by round;
- accepted/rejected clients and aggregation weights by round.

Keep the seed-44 comparison paired. Do not claim that a higher ROC-AUC alone proves the global IDS is better.

## Scope and limitations

This is a pilot on one Dirichlet partition (alpha 0.5, partition seed 43), one attack target (client 5),
one poisoning fraction (20% of eligible non-Benign labels), and one onset schedule (round 6).
The clean calibration runs use different training seeds from the held-out seed-44 evaluation.

The independent test set remains untouched by this detector calibration and these runner metrics.
Do not tune the threshold or persistence after inspecting the seed-44 attack results and then report
those same results as independent evaluation. The classifier's macro-F1 weakness is a separate
class-imbalance issue; this branch does not change the model architecture or loss function.
