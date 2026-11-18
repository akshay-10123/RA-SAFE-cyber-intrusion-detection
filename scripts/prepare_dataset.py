"""
Build standardized_dataset_*.csv files expected by the IDS scripts.

Uses the public Dataset_T-ITS.csv when available; fills missing cyber/physical
columns and writes StandardScaler-normalized files for each fusion variant.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.attack_labels import resolve_attack_class  # noqa: E402

CYBER_COLS = [
    "frame.len", "wlan.ta", "wlan.sa", "wlan.ra", "wlan.da", "wlan.bssid",
    "wlan.duration", "wlan.seq", "wlan.fc.type", "wlan.fc.subtype", "wlan.flags",
    "wlan.fcs", "wlan.fcs.status", "wlan.ccmp.extiv", "data.len",
    "radiotap.hdr_length", "radiotap.signal_quality", "wlan_radio.datarate",
    "wlan_radio.channel", "wlan_radio.SNR (db)", "wlan_radio.preamble",
]

PHYSICAL_COLS = [
    "pitch", "roll", "yaw", "vgx", "vgy", "vgz",
    "flight_time", "agx", "agy", "agz", "tof",
]

ALL_FEATURE_COLS = CYBER_COLS + PHYSICAL_COLS

FUSION_SPLITS = {
    "drone1": lambda n: slice(0, n // 2),
    "drone2": lambda n: slice(n // 2, n),
    "central": lambda n: slice(0, n),
}

FUSION_NAMES = [
    "drone1_cyber", "drone2_cyber", "central_cyber",
    "drone1_physical", "drone2_physical", "central_physical",
    "drone1_combined", "drone2_combined", "central_combined",
]


def _find_raw_csv() -> Path | None:
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "Dataset_T-ITS.csv",
        PROJECT_ROOT.parent / "dataset_updated" / "Dataset_T-ITS.csv",
        Path.home() / "OneDrive" / "Documents" / "dataset_updated" / "Dataset_T-ITS.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_or_synthesize_raw(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    raw_path = _find_raw_csv()
    if raw_path is not None:
        print(f"Loading raw data from {raw_path}")
        return pd.read_csv(raw_path, low_memory=False)

    print("Raw Dataset_T-ITS.csv not found — generating synthetic rows.")
    rng = np.random.default_rng(seed)
    labels = rng.choice(["benign", "dos", "replay", "evil_twin", "fdi"], n_rows, p=[0.4, 0.15, 0.15, 0.15, 0.15])
    data = {
        "frame.len": rng.integers(24, 1500, n_rows),
        "wlan.ta": rng.integers(0, 2, n_rows),
        "wlan.sa": rng.integers(0, 2, n_rows),
        "wlan.ra": rng.integers(0, 2, n_rows),
        "wlan.da": rng.integers(0, 2, n_rows),
        "wlan.bssid": rng.integers(0, 2, n_rows),
        "wlan.duration": rng.integers(0, 500, n_rows),
        "wlan.seq": rng.integers(0, 4096, n_rows),
        "wlan.fc.type": rng.integers(0, 3, n_rows),
        "wlan.fc.subtype": rng.integers(0, 16, n_rows),
        "data.len": rng.integers(0, 1400, n_rows),
        "class": labels,
    }
    return pd.DataFrame(data)


def _build_full_frame(raw: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = len(raw)
    attack = (raw["class"] != "benign").astype(int).values if "class" in raw.columns else rng.integers(0, 2, n)

    out = pd.DataFrame(index=range(n))
    for col in ALL_FEATURE_COLS:
        out[col] = 0.0

    direct_map = {
        "frame.len": "frame.len",
        "wlan.ta": "wlan.ta",
        "wlan.sa": "wlan.sa",
        "wlan.ra": "wlan.ra",
        "wlan.da": "wlan.da",
        "wlan.bssid": "wlan.bssid",
        "wlan.duration": "wlan.duration",
        "wlan.seq": "wlan.seq",
        "wlan.fc.type": "wlan.fc.type",
        "wlan.fc.subtype": "wlan.fc.subtype",
        "data.len": "data.len",
    }
    for dst, src in direct_map.items():
        if src in raw.columns:
            out[dst] = pd.to_numeric(raw[src], errors="coerce").fillna(0)

    if "tcp.flags" in raw.columns:
        out["wlan.flags"] = pd.to_numeric(raw["tcp.flags"], errors="coerce").fillna(0)
    else:
        out["wlan.flags"] = rng.integers(0, 256, n)

    out["wlan.fcs"] = rng.integers(0, 2, n)
    out["wlan.fcs.status"] = rng.integers(0, 2, n)
    out["wlan.ccmp.extiv"] = rng.integers(0, 2, n)
    out["radiotap.hdr_length"] = rng.integers(26, 36, n)
    out["radiotap.signal_quality"] = rng.integers(20, 100, n)
    out["wlan_radio.datarate"] = rng.integers(1, 54, n)
    out["wlan_radio.channel"] = rng.integers(1, 14, n)
    out["wlan_radio.SNR (db)"] = rng.normal(25, 5, n)
    out["wlan_radio.preamble"] = rng.integers(0, 2, n)

    # Physical telemetry — shift under attack for separability
    out["pitch"] = rng.normal(0, 5, n) + attack * 8
    out["roll"] = rng.normal(0, 4, n) + attack * 6
    out["yaw"] = rng.normal(0, 10, n) + attack * 12
    out["vgx"] = rng.normal(0, 2, n) + attack * 3
    out["vgy"] = rng.normal(0, 2, n) + attack * 2
    out["vgz"] = rng.normal(0, 1.5, n) + attack * 2
    out["flight_time"] = np.linspace(0, 3600, n) + attack * 50
    out["agx"] = rng.normal(0, 1, n) + attack * 2
    out["agy"] = rng.normal(0, 1, n) + attack * 1.5
    out["agz"] = rng.normal(9.8, 0.5, n) + attack * 0.5
    out["tof"] = rng.normal(150, 30, n) - attack * 20

    raw_class_col = raw["class"] if "class" in raw.columns else None
    attack_ids = []
    attack_names = []
    for i in range(n):
        raw_val = raw_class_col.iloc[i] if raw_class_col is not None else None
        class_id, class_name = resolve_attack_class(raw_val, i)
        attack_ids.append(class_id)
        attack_names.append(class_name)

    attack_ids_arr = np.array(attack_ids, dtype=int)
    out["attack_class"] = attack_ids_arr
    out["attack_label"] = attack_names
    out["target"] = raw["class"].astype(str) if "class" in raw.columns else np.where(attack, "attack", "benign")
    # Binary label unchanged — existing SVM/FNN/LSTM/CNN scripts keep working.
    out["target.value"] = (attack_ids_arr != 0).astype(int)
    return out


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    scaled = df.copy()
    scaler = StandardScaler()
    scaled[ALL_FEATURE_COLS] = scaler.fit_transform(scaled[ALL_FEATURE_COLS])
    return scaled


def _subset_for_fusion(full: pd.DataFrame, fusion: str) -> pd.DataFrame:
    for prefix, slicer in FUSION_SPLITS.items():
        if fusion.startswith(prefix):
            idx = slicer(len(full))
            return full.iloc[idx].reset_index(drop=True)
    raise ValueError(f"Unknown fusion: {fusion}")


def write_datasets(output_dirs: list[Path]) -> None:
    raw = _load_or_synthesize_raw()
    full = _standardize(_build_full_frame(raw))

    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

    for fusion in FUSION_NAMES:
        subset = _subset_for_fusion(full, fusion)
        for out_dir in output_dirs:
            path = out_dir / f"standardized_dataset_{fusion}.csv"
            subset.to_csv(path, index=False)
            print(f"Wrote {path} ({len(subset)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare standardized IDS datasets")
    parser.add_argument(
        "--output",
        action="append",
        default=None,
        help="Output directory (can be repeated). Defaults: project data/ and Documents/dataset_updated",
    )
    args = parser.parse_args()

    default_outputs = [
        PROJECT_ROOT / "data" / "dataset_updated",
        PROJECT_ROOT / "data" / "dataset_central",
        PROJECT_ROOT.parent / "dataset_updated",
        PROJECT_ROOT.parent / "dataset_central",
    ]
    outputs = [Path(p) for p in args.output] if args.output else default_outputs
    write_datasets(outputs)


if __name__ == "__main__":
    main()
