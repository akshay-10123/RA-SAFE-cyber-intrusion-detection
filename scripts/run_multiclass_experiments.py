"""Run multi-class 1D-CNN for drone1 cyber / physical / combined fusions."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CNN_SCRIPT = PROJECT_ROOT / "1D-CNN -based IDS" / "multiclass_cnn.py"

# Central fusion includes all 5 attack types; drone1/drone2 are half-splits.
DEFAULT_FUSIONS = ["central_cyber", "central_physical", "central_combined"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = use full fusion split")
    parser.add_argument("--quick", action="store_true", help="Fewer epochs and subsample for smoke test")
    parser.add_argument("--fusion", action="append", default=None)
    args = parser.parse_args()

    epochs = 5 if args.quick else args.epochs
    max_samples = 8000 if args.quick else args.max_samples
    fusions = args.fusion or DEFAULT_FUSIONS

    py = sys.executable
    for fusion in fusions:
        cmd = [
            py,
            str(CNN_SCRIPT),
            "--fusion",
            fusion,
            "--epochs",
            str(epochs),
            "--metrics_title",
            fusion,
        ]
        if max_samples:
            cmd.extend(["--max_samples", str(max_samples)])
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
