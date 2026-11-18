"""Load multi-class fusion datasets and saved 1D-CNN models for streaming inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CNN_DIR = PROJECT_ROOT / "1D-CNN -based IDS"
MODEL_DIR = CNN_DIR / "multiclass_models"

KEEP_COMBINED = [
    "frame.len", "wlan.ta", "wlan.sa", "wlan.ra", "wlan.da", "wlan.bssid", "wlan.duration", "wlan.seq",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.flags", "wlan.fcs", "wlan.fcs.status", "wlan.ccmp.extiv",
    "data.len", "radiotap.hdr_length", "radiotap.signal_quality", "wlan_radio.datarate",
    "wlan_radio.channel", "wlan_radio.SNR (db)", "wlan_radio.preamble",
    "pitch", "roll", "yaw", "vgx", "vgy", "vgz", "flight_time", "agx", "agy", "agz", "tof",
    "attack_class",
]
KEEP_CYBER = [
    "frame.len", "wlan.ta", "wlan.sa", "wlan.ra", "wlan.da", "wlan.bssid", "wlan.duration", "wlan.seq",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.flags", "wlan.fcs", "wlan.fcs.status", "wlan.ccmp.extiv",
    "data.len", "radiotap.hdr_length", "radiotap.signal_quality", "wlan_radio.datarate",
    "wlan_radio.channel", "wlan_radio.SNR (db)", "wlan_radio.preamble", "attack_class",
]
KEEP_PHYSICAL = [
    "pitch", "roll", "yaw", "vgx", "vgy", "vgz", "flight_time", "agx", "agy", "agz", "tof", "attack_class",
]

STREAM_FUSIONS = ["central_cyber", "central_physical", "central_combined"]


def model_path(fusion: str) -> Path:
    return MODEL_DIR / f"multiclass_{fusion}.keras"


def load_fusion_frame(fusion: str) -> pd.DataFrame:
    from scripts.paths import standardized_csv

    path = standardized_csv(fusion)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run scripts/prepare_dataset.py")

    df = pd.read_csv(path, low_memory=False)
    if "attack_class" not in df.columns:
        raise ValueError("attack_class missing — re-run scripts/prepare_dataset.py")

    df = df.drop(columns=[c for c in ("target", "attack_label") if c in df.columns])

    if fusion in ["drone1_cyber", "drone2_cyber", "central_cyber"]:
        cols = KEEP_CYBER
    elif fusion in ["drone1_physical", "drone2_physical", "central_physical"]:
        cols = KEEP_PHYSICAL
    elif fusion in ["drone1_combined", "drone2_combined", "central_combined"]:
        cols = KEEP_COMBINED
    else:
        raise ValueError(f"Unknown fusion: {fusion}")

    return df[cols].reset_index(drop=True)


def frame_to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(int)
    return x, y


def reshape_for_cnn(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(1, x.shape[0], 1)
    return x.reshape(x.shape[0], x.shape[1], 1)


def available_stream_models() -> list[str]:
    if not MODEL_DIR.exists():
        return []
    found = []
    for p in MODEL_DIR.glob("multiclass_*.keras"):
        fusion = p.stem.replace("multiclass_", "", 1)
        found.append(fusion)
    return sorted(found, key=lambda f: STREAM_FUSIONS.index(f) if f in STREAM_FUSIONS else 999)
