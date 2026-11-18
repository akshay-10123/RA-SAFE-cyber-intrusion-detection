"""
Run all IDS models for drone1 cyber / physical / combined fusions.

Usage:
  python scripts/run_all_experiments.py
  python scripts/run_all_experiments.py --quick   # faster (fewer epochs / samples)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

FUSIONS = ["drone1_cyber", "drone1_physical", "drone1_combined"]


def run_step(name: str, cwd: Path, cmd: list[str]) -> bool:
    print("\n" + "=" * 60)
    print(name)
    print(" ".join(cmd))
    print("=" * 60)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"FAILED: {name} (exit {result.returncode})")
        return False
    print(f"OK: {name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs/samples")
    args = parser.parse_args()

    cnn_epochs = "5" if args.quick else "50"
    fnn_epochs = "5" if args.quick else "30"
    lstm_epochs = "3" if args.quick else "10"
    lstm_samples = "2000" if args.quick else "5000"
    svm_samples = ["--max_samples=3000"] if args.quick else ["--max_samples=8000"]

    ok = True
    for fusion in FUSIONS:
        ok &= run_step(
            f"1D-CNN · {fusion}",
            PROJECT_ROOT / "1D-CNN -based IDS",
            [PYTHON, "1D_CNN.py", f"--fusion={fusion}", f"--metrics_title={fusion}", f"--epochs={cnn_epochs}"],
        )

    for fusion in FUSIONS:
        ok &= run_step(
            f"SVM · {fusion}",
            PROJECT_ROOT / "SVM-based IDS",
            [PYTHON, "svm.py", f"--fusion={fusion}", f"--png={fusion}.png", f"--metrics_title={fusion}", *svm_samples],
        )

    for fusion in FUSIONS:
        ok &= run_step(
            f"FNN · {fusion}",
            PROJECT_ROOT / "FNN-based IDS",
            [PYTHON, "FNN.py", f"--fusion={fusion}", f"--title_name={fusion}", f"--epochs={fnn_epochs}"],
        )

    for fusion in FUSIONS:
        ok &= run_step(
            f"LSTM · {fusion}",
            PROJECT_ROOT / "LSTM-based IDS",
            [
                PYTHON, "LSTM.py",
                f"--fusion={fusion}",
                f"--metrics_title={fusion}",
                f"--epochs={lstm_epochs}",
                f"--max_samples={lstm_samples}",
                "--cv_folds=2",
            ],
        )

    print("\n" + ("All experiments finished." if ok else "Some experiments failed."))
    print(f"Metrics: {PROJECT_ROOT / 'results' / 'metrics.json'}")
    print("Dashboard: python -m streamlit run dashboard/app.py")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
