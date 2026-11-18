"""Helpers for Keras history metric keys (Keras 2 vs 3 naming)."""
from __future__ import annotations

import numpy as np


def history_metric(history: dict, name: str) -> np.ndarray:
    if name.startswith("val_"):
        base = name[4:]
        candidates = [
            name,
            f"val_{base.lower()}",
            f"val_{base.capitalize()}",
        ]
    else:
        candidates = [name, name.lower(), name.capitalize()]
    for key in candidates:
        if key in history:
            return np.array(history[key])
    raise KeyError(f"Metric '{name}' not found. Keys: {list(history.keys())}")
