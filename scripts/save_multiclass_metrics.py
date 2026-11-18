"""Save multi-class IDS metrics (separate from binary results/metrics.json)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics_multiclass.json"


def save_multiclass_metrics(
    model: str,
    fusion: str,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    f1_weighted: float,
    auc_macro: float,
    per_class: dict[str, dict[str, float]] | None = None,
    confusion_matrix: list[list[int]] | None = None,
    class_names: list[str] | None = None,
    extras: dict | None = None,
) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {"runs": []}
    if METRICS_PATH.exists():
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    entry = {
        "model": model,
        "fusion": fusion,
        "task": "multiclass",
        "num_classes": 5,
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "auc_macro": float(auc_macro),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if per_class is not None:
        entry["per_class"] = per_class
    if confusion_matrix is not None:
        entry["confusion_matrix"] = confusion_matrix
    if class_names is not None:
        entry["class_names"] = class_names
    if extras:
        entry.update(extras)

    runs = payload.get("runs", [])
    runs = [r for r in runs if not (r.get("model") == model and r.get("fusion") == fusion)]
    runs.append(entry)
    payload["runs"] = runs
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
