# Independent test-set evaluation

The Phase 9 training runner intentionally uses the server-validation split only. This
standalone evaluator loads a saved round checkpoint and evaluates it on the held-out,
preprocessed CIC-IDS-2017 test split. It does not train or tune any model or threshold.

## Local checks

Run from the repository root after activating the project's virtual environment:

```cmd
python -m compileall -q scripts
```

## Evaluate the seed-45 clean checkpoint

```cmd
python -m scripts.evaluate_independent_test --checkpoint results\phase9_clean_control_client05_p0.2000_s45_ps2028_q90_holdout_clean_s45\phase9_round_10.pt --output-dir results\independent_test_seed45_clean
```

## Evaluate the seed-45 unfiltered attack checkpoint

```cmd
python -m scripts.evaluate_independent_test --checkpoint results\phase9_delayed_client05_p0.2000_start6_s45_ps2028_baseline_s45\phase9_round_10.pt --output-dir results\independent_test_seed45_attack_baseline
```

## Evaluate the seed-45 calibrated-defense checkpoint

```cmd
python -m scripts.evaluate_independent_test --checkpoint results\phase9_delayed_client05_p0.2000_start6_s45_ps2028_calibrated_q90_s45\phase9_round_10.pt --output-dir results\independent_test_seed45_attack_calibrated
```

Compare macro-F1, balanced accuracy, accuracy, per-class recall, benign false-alarm rate,
attack-to-Benign rate and the confusion matrices. The 565,573-record test split is evaluated
once per checkpoint. Do not choose a new threshold or modify the method based on these
test metrics and then report the same test set as independent confirmation.
