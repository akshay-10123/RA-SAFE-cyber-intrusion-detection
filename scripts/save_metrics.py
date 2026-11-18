"""Save training metrics to a unified JSON file for the dashboard."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"


def save_metrics(
    model: str,
    fusion: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    auc: float,
) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {"runs": []}
    if METRICS_PATH.exists():
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    entry = {
        "model": model,
        "fusion": fusion,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    runs = payload.get("runs", [])
    runs = [r for r in runs if not (r.get("model") == model and r.get("fusion") == fusion)]
    runs.append(entry)
    payload["runs"] = runs
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
