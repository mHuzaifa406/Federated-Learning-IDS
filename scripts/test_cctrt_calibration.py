"""Unit tests for clean-reference CCTRT calibration."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trfa.cctrt_calibration import (
    FEATURE,
    build_calibration,
    score_calibrated_round,
    apply_calibration_to_frame,
)


class CCTRTCalibrationTests(unittest.TestCase):
    def _make_clean_run(self, root: Path, seed: int) -> Path:
        run_dir = root / f"clean_seed_{seed}"
        run_dir.mkdir(parents=True)
        metadata = {
            "phase": "Phase 9 matched all-honest control",
            "dataset": "CIC-IDS-2017",
            "classification_task": "binary_benign_vs_attack",
            "model_config": {"input_features": 78, "num_classes": 2},
            "local_epochs": 2,
            "learning_rate": 0.015,
            "batch_size": 512,
            "training_seed": seed,
            "partition_seed": 43,
            "dirichlet_alpha": 0.5,
            "rounds": 10,
            "clients": 10,
            "malicious_client": None,
            "attack_active": False,
            "screening_enabled": False,
            "all_clients_aggregated": True,
        }
        (run_dir / "phase9_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        rows = []
        seed_offset = float(seed - 40)
        for rnd in range(1, 11):
            for client in range(1, 11):
                value = (
                    0.01 * rnd
                    + 0.001 * client
                    + seed_offset * (0.0001 + 0.000003 * rnd + 0.000002 * client)
                )
                rows.append({
                    "Round": rnd,
                    "Client": client,
                    "Attack_Active": False,
                    FEATURE: value,
                })
        pd.DataFrame(rows).to_csv(run_dir / "phase9_client_round_scores.csv", index=False)
        return run_dir

    def test_build_score_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dirs = [self._make_clean_run(root, seed) for seed in (41, 42)]
            calibration = build_calibration(run_dirs)
            self.assertEqual(calibration["training_seeds"], [41, 42])
            self.assertEqual(len(calibration["profile"]), 100)
            self.assertTrue(calibration["threshold_z"] > 0.0)

            # An unmodified clean-reference point should not alert.
            clean_features = {
                cid: {FEATURE: calibration["profile"][f"1:{cid}"]["center"]}
                for cid in range(1, 11)
            }
            scored = score_calibrated_round(clean_features, 1, calibration)
            self.assertFalse(any(row["Calibrated_Alert"] for row in scored.values()))

            # A very large positive deviation across two consecutive rounds
            # should only become a rejection after the persistence condition.
            rows = []
            for rnd in range(1, 11):
                for cid in range(1, 11):
                    center = calibration["profile"][f"{rnd}:{cid}"]["center"]
                    if cid == 5 and rnd in (1, 2):
                        center += 100.0 * calibration["pooled_clean_residual_scale"]
                    rows.append({"Round": rnd, "Client": cid, FEATURE: center})
            result = apply_calibration_to_frame(
                pd.DataFrame(rows), calibration, persistence=2
            )
            target = result[result["Client"] == 5].set_index("Round")
            self.assertFalse(bool(target.loc[1, "Rejected_By_Calibrated_TRFA"]))
            self.assertTrue(bool(target.loc[2, "Rejected_By_Calibrated_TRFA"]))

    def test_rejects_duplicate_training_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._make_clean_run(root, 41)
            b = self._make_clean_run(root, 42)
            metadata_path = b / "phase9_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["training_seed"] = 41
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_calibration([a, b])

    def test_rejects_mismatched_training_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._make_clean_run(root, 41)
            b = self._make_clean_run(root, 42)
            metadata_path = b / "phase9_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["learning_rate"] = 0.01
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_calibration([a, b])

    def test_rejects_attack_run_as_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._make_clean_run(root, 41)
            b = self._make_clean_run(root, 42)
            metadata_path = b / "phase9_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["malicious_client"] = 5
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_calibration([a, b])


if __name__ == "__main__":
    unittest.main()
