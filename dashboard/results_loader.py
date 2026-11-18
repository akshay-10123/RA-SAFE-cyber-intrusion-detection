"""Load metrics and plot paths from IDS training outputs."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "1D-CNN": {
        "dir": PROJECT_ROOT / "1D-CNN -based IDS",
        "results_dir": "cnn_results",
        "metrics_file": "results.txt",
        "plots": [
            ("F1 Score", "F1_{fusion}.png"),
            ("Loss", "loss_{fusion}.png"),
        ],
    },
    "SVM": {
        "dir": PROJECT_ROOT / "SVM-based IDS",
        "results_dir": "svm_results",
        "metrics_file": None,
        "plots": [
            ("Learning curve", "{fusion}.png"),
        ],
    },
    "FNN": {
        "dir": PROJECT_ROOT / "FNN-based IDS",
        "results_dir": "fnn_results",
        "metrics_file": None,
        "plots": [
            ("F1 Score", "F1_{fusion}.png"),
            ("Loss", "loss_{fusion}.png"),
        ],
    },
    "LSTM": {
        "dir": PROJECT_ROOT / "LSTM-based IDS",
        "results_dir": "lstm_results",
        "metrics_file": None,
        "plots": [
            ("F1 Score", "F1_{fusion}.png"),
            ("Loss", "loss_{fusion}.png"),
        ],
    },
}

FUSION_OPTIONS = [
    "drone1_cyber",
    "drone2_cyber",
    "central_cyber",
    "drone1_physical",
    "drone2_physical",
    "central_physical",
    "drone1_combined",
    "drone2_combined",
    "central_combined",
]

METRIC_LABELS = ["accuracy", "precision", "recall", "f1", "auc"]


@dataclass
class MetricRow:
    model: str
    fusion: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


def parse_cnn_results(path: Path) -> list[MetricRow]:
    if not path.exists():
        return []

    rows: list[MetricRow] = []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    i = 0
    while i < len(lines):
        fusion = lines[i].strip()
        if i + 1 < len(lines) and lines[i + 1].startswith("accuracy"):
            match = re.search(r"\[(.+)\]", lines[i + 1])
            if match:
                values = [float(x) for x in ast.literal_eval(f"[{match.group(1)}]")]
                if len(values) == 5:
                    rows.append(
                        MetricRow(
                            model="1D-CNN",
                            fusion=fusion,
                            accuracy=values[0],
                            precision=values[1],
                            recall=values[2],
                            f1=values[3],
                            auc=values[4],
                        )
                    )
            i += 2
        else:
            i += 1
    return rows


def latest_metrics_by_fusion(rows: list[MetricRow]) -> list[MetricRow]:
    """Keep the last run per (model, fusion) pair."""
    seen: dict[tuple[str, str], MetricRow] = {}
    for row in rows:
        seen[(row.model, row.fusion)] = row
    return list(seen.values())


def plot_path(model: str, fusion: str, plot_template: str) -> Path | None:
    info = MODELS[model]
    rel = plot_template.format(fusion=fusion)
    path = info["dir"] / info["results_dir"] / rel
    return path if path.exists() else None


def available_fusions(model: str) -> list[str]:
    info = MODELS[model]
    results_root = info["dir"] / info["results_dir"]
    if not results_root.exists():
        return []

    found: set[str] = set()
    for path in results_root.glob("*.png"):
        name = path.stem
        for fusion in FUSION_OPTIONS:
            if fusion in name or name == fusion:
                found.add(fusion)
    return sorted(found, key=FUSION_OPTIONS.index)


def all_discovered_plots() -> list[dict]:
    items = []
    for model, info in MODELS.items():
        results_root = info["dir"] / info["results_dir"]
        if not results_root.exists():
            continue
        for png in sorted(results_root.glob("*.png")):
            fusion = "unknown"
            for opt in FUSION_OPTIONS:
                if opt in png.stem or png.stem == opt:
                    fusion = opt
                    break
            items.append(
                {
                    "model": model,
                    "fusion": fusion,
                    "title": png.stem,
                    "path": png,
                }
            )
    return items


def load_json_metrics() -> list[MetricRow]:
    path = PROJECT_ROOT / "results" / "metrics.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in data.get("runs", []):
        rows.append(
            MetricRow(
                model=r["model"],
                fusion=r["fusion"],
                accuracy=r["accuracy"],
                precision=r["precision"],
                recall=r["recall"],
                f1=r["f1"],
                auc=r["auc"],
            )
        )
    return rows


def load_all_metrics() -> list[MetricRow]:
    rows = load_json_metrics()
    if not rows:
        cnn_path = MODELS["1D-CNN"]["dir"] / MODELS["1D-CNN"]["metrics_file"]
        rows = parse_cnn_results(cnn_path)
    return latest_metrics_by_fusion(rows)


@dataclass
class MulticlassMetricRow:
    model: str
    fusion: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    auc_macro: float
    per_class: dict | None
    confusion_matrix: list[list[int]] | None
    class_names: list[str] | None
    n_params: int | None = None
    inference_ms_per_sample: float | None = None
    fusion_by_class: dict | None = None


def load_multiclass_metrics() -> list[MulticlassMetricRow]:
    path = PROJECT_ROOT / "results" / "metrics_multiclass.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[MulticlassMetricRow] = []
    for r in data.get("runs", []):
        rows.append(
            MulticlassMetricRow(
                model=r.get("model", "1D-CNN (multi-class)"),
                fusion=r["fusion"],
                accuracy=r["accuracy"],
                precision_macro=r["precision_macro"],
                recall_macro=r["recall_macro"],
                f1_macro=r["f1_macro"],
                f1_weighted=r.get("f1_weighted", r["f1_macro"]),
                auc_macro=r.get("auc_macro", 0.0),
                per_class=r.get("per_class"),
                confusion_matrix=r.get("confusion_matrix"),
                class_names=r.get("class_names"),
                n_params=r.get("n_params"),
                inference_ms_per_sample=r.get("inference_ms_per_sample"),
                fusion_by_class=r.get("fusion_by_class"),
            )
        )
    seen: dict[tuple[str, str], MulticlassMetricRow] = {}
    for row in rows:
        seen[(row.model, row.fusion)] = row
    return list(seen.values())


def cp_attn_confusion_plot_path(fusion: str) -> Path | None:
    path = PROJECT_ROOT / "1D-CNN -based IDS" / "cp_attn_results" / f"confusion_{fusion}.png"
    return path if path.exists() else None


def multiclass_confusion_plot_path(fusion: str) -> Path | None:
    path = PROJECT_ROOT / "1D-CNN -based IDS" / "multiclass_results" / f"confusion_{fusion}.png"
    return path if path.exists() else None


def ra_safe_confusion_plot_path() -> Path | None:
    path = PROJECT_ROOT / "results" / "confusion_ra_safe.png"
    return path if path.exists() else None


def load_ra_safe_metrics() -> dict | None:
    path = PROJECT_ROOT / "results" / "ra_safe_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

