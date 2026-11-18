"""Import existing 1D-CNN results.txt into results/metrics.json."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.results_loader import parse_cnn_results, MODELS  # noqa: E402
from scripts.save_metrics import save_metrics  # noqa: E402

if __name__ == "__main__":
    path = MODELS["1D-CNN"]["dir"] / MODELS["1D-CNN"]["metrics_file"]
    for row in parse_cnn_results(path):
        save_metrics(row.model, row.fusion, row.accuracy, row.precision, row.recall, row.f1, row.auc)
        print(f"Imported {row.model} · {row.fusion}")
