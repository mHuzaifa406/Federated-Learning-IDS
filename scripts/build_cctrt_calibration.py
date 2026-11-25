"""Build a CCTRT clean-reference calibration JSON from all-honest Phase 9 runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trfa.cctrt_calibration import build_calibration, save_calibration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-dir", action="append", required=True,
        help="All-honest Phase 9 output directory; repeat once per distinct training seed.",
    )
    parser.add_argument("--quantile", type=float, default=0.975)
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "results" / "cctrt_clean_calibration_v1.json")
    )
    args = parser.parse_args()

    calibration = build_calibration(args.clean_dir, quantile=args.quantile)
    output = save_calibration(calibration, args.output)
    print("CCTRT clean-reference calibration created.")
    print("Independent all-honest runs:", calibration["calibration_run_count"])
    print("Training seeds:", calibration["training_seeds"])
    print("Feature:", calibration["feature"])
    print("Clean residual scale:", f'{calibration["pooled_clean_residual_scale"]:.8f}')
    print("Empirical clean threshold z:", f'{calibration["threshold_z"]:.6f}')
    print("Output:", output)
    print("This is an anomaly threshold, not a probability of maliciousness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
