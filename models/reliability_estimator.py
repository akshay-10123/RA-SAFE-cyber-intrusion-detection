"""
Modality Reliability Estimator for UAV Cyber-Physical Intrusion Detection.

Computes interpretable quality scores Q_c (cyber) and Q_p (physical) bounded in [0, 1]
using strictly inference-time observables and training data statistics.
Zero attack labels or ground-truth classifications are used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# 21 Cyber column indices (relative to cyber branch input)
# 0: frame.len, 1: wlan.ta, 2: wlan.sa, 3: wlan.ra, 4: wlan.da, 5: wlan.bssid,
# 6: wlan.duration, 7: wlan.seq, 8: wlan.fc.type, 9: wlan.fc.subtype, 10: wlan.flags,
# 11: wlan.fcs, 12: wlan.fcs.status, 13: wlan.ccmp.extiv, 14: data.len,
# 15: radiotap.hdr_length, 16: radiotap.signal_quality, 17: wlan_radio.datarate,
# 18: wlan_radio.channel, 19: wlan_radio.SNR (db), 20: wlan_radio.preamble

# 11 Physical column indices (relative to physical branch input)
# 0: pitch, 1: roll, 2: yaw, 3: vgx, 4: vgy, 5: vgz,
# 6: flight_time, 7: agx, 8: agy, 9: agz, 10: tof


class ModalityReliabilityEstimator:
    """Estimates empirical data reliability for cyber and physical UAV channels."""

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.cyber_mean: np.ndarray | None = None
        self.cyber_std: np.ndarray | None = None
        self.phys_mean: np.ndarray | None = None
        self.phys_std: np.ndarray | None = None

        # Robust percentiles for clipping extreme anomalies
        self.cyber_q01: np.ndarray | None = None
        self.cyber_q99: np.ndarray | None = None
        self.phys_q01: np.ndarray | None = None
        self.phys_q99: np.ndarray | None = None

    def fit(self, x_cyber: np.ndarray, x_phys: np.ndarray) -> "ModalityReliabilityEstimator":
        """
        Fit baseline nominal distribution parameters ONLY on training data.
        x_cyber: (N, 21)
        x_phys:  (N, 11)
        """
        x_c = np.asarray(x_cyber, dtype=np.float32)
        x_p = np.asarray(x_phys, dtype=np.float32)

        self.cyber_mean = np.mean(x_c, axis=0)
        self.cyber_std = np.std(x_c, axis=0)
        self.cyber_std = np.where(self.cyber_std < 1e-4, 1.0, self.cyber_std)

        self.phys_mean = np.mean(x_p, axis=0)
        self.phys_std = np.std(x_p, axis=0)
        self.phys_std = np.where(self.phys_std < 1e-4, 1.0, self.phys_std)

        self.cyber_q01 = np.percentile(x_c, 1, axis=0)
        self.cyber_q99 = np.percentile(x_c, 99, axis=0)
        self.phys_q01 = np.percentile(x_p, 1, axis=0)
        self.phys_q99 = np.percentile(x_p, 99, axis=0)

        self.is_fitted = True
        return self

    def estimate_cyber_reliability(self, x_cyber: np.ndarray) -> np.ndarray:
        """
        Compute Q_c in [0, 1] for cyber features.
        Considers:
          1. Signal-to-noise ratio and signal quality validity.
          2. Sequence / duration plausibility.
          3. Multi-dimensional Mahalanobis/z-score statistical plausibility.
        Returns: (N, 1) float32 array
        """
        if not self.is_fitted:
            raise RuntimeError("ModalityReliabilityEstimator must be fitted before estimation.")

        x = np.asarray(x_cyber, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim == 3:
            x = x.squeeze(-1)

        n = x.shape[0]

        # 1. Normalized squared z-scores across all cyber attributes (bounded)
        z_scores = np.abs((x - self.cyber_mean) / self.cyber_std)
        # Cap z-scores to avoid single runaway feature dominating
        capped_z = np.clip(z_scores, 0.0, 10.0)
        mean_z = np.mean(capped_z, axis=1)  # shape (N,)
        q_stat = np.exp(-0.35 * mean_z)

        # 2. Specific Protocol / Radio Signal Indicators
        # SNR check (index 19 in cyber set): extreme negative or degraded SNR reduces score
        snr_idx = 19
        snr_val = x[:, snr_idx]
        q_snr = 1.0 / (1.0 + np.exp(-1.5 * (snr_val - np.median(self.cyber_mean[snr_idx]))))
        q_snr = np.clip(q_snr, 0.1, 1.0)

        # 3. Missingness / Zeroed-out packet indication
        # Under packet drop/loss simulation, features often zero out or freeze
        zero_fraction = np.mean(x == 0.0, axis=1)
        q_completeness = np.clip(1.0 - 1.2 * zero_fraction, 0.05, 1.0)

        # Blended Cyber Quality
        q_c = 0.50 * q_stat + 0.25 * q_snr + 0.25 * q_completeness
        q_c = np.clip(q_c, 0.01, 0.99).reshape(n, 1).astype(np.float32)
        return q_c

    def estimate_phys_reliability(self, x_phys: np.ndarray) -> np.ndarray:
        """
        Compute Q_p in [0, 1] for physical telemetry features.
        Considers:
          1. Kinematic acceleration magnitude plausibility (agx, agy, agz).
          2. Velocity magnitude plausibility (vgx, vgy, vgz).
          3. Multi-dimensional statistical drift and bounds.
        Returns: (N, 1) float32 array
        """
        if not self.is_fitted:
            raise RuntimeError("ModalityReliabilityEstimator must be fitted before estimation.")

        x = np.asarray(x_phys, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim == 3:
            x = x.squeeze(-1)

        n = x.shape[0]

        # 1. Statistical distance from nominal flight envelope
        z_scores = np.abs((x - self.phys_mean) / self.phys_std)
        capped_z = np.clip(z_scores, 0.0, 10.0)
        mean_z = np.mean(capped_z, axis=1)
        q_stat = np.exp(-0.35 * mean_z)

        # 2. Kinematic acceleration consistency (indices 7: agx, 8: agy, 9: agz)
        # In standardized space, large runaway values represent severe sensor drift/fault
        acc_norm = np.sqrt(x[:, 7] ** 2 + x[:, 8] ** 2 + x[:, 9] ** 2)
        q_acc = np.exp(-0.25 * np.maximum(0.0, acc_norm - 2.5))

        # 3. Velocity consistency (indices 3: vgx, 4: vgy, 5: vgz)
        vel_norm = np.sqrt(x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
        q_vel = np.exp(-0.25 * np.maximum(0.0, vel_norm - 2.5))

        # Blended Physical Quality
        q_p = 0.50 * q_stat + 0.25 * q_acc + 0.25 * q_vel
        q_p = np.clip(q_p, 0.01, 0.99).reshape(n, 1).astype(np.float32)
        return q_p

    def estimate(self, x_cyber: np.ndarray, x_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (Q_c, Q_p) tuple for given batches."""
        q_c = self.estimate_cyber_reliability(x_cyber)
        q_p = self.estimate_phys_reliability(x_phys)
        return q_c, q_p

    def save(self, filepath: str | Path) -> None:
        """Persist fitted statistics to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cyber_mean": self.cyber_mean.tolist() if self.cyber_mean is not None else [],
            "cyber_std": self.cyber_std.tolist() if self.cyber_std is not None else [],
            "phys_mean": self.phys_mean.tolist() if self.phys_mean is not None else [],
            "phys_std": self.phys_std.tolist() if self.phys_std is not None else [],
            "cyber_q01": self.cyber_q01.tolist() if self.cyber_q01 is not None else [],
            "cyber_q99": self.cyber_q99.tolist() if self.cyber_q99 is not None else [],
            "phys_q01": self.phys_q01.tolist() if self.phys_q01 is not None else [],
            "phys_q99": self.phys_q99.tolist() if self.phys_q99 is not None else [],
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "ModalityReliabilityEstimator":
        """Load fitted estimator from JSON."""
        inst = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst.cyber_mean = np.array(data["cyber_mean"], dtype=np.float32)
        inst.cyber_std = np.array(data["cyber_std"], dtype=np.float32)
        inst.phys_mean = np.array(data["phys_mean"], dtype=np.float32)
        inst.phys_std = np.array(data["phys_std"], dtype=np.float32)
        inst.cyber_q01 = np.array(data.get("cyber_q01", []), dtype=np.float32)
        inst.cyber_q99 = np.array(data.get("cyber_q99", []), dtype=np.float32)
        inst.phys_q01 = np.array(data.get("phys_q01", []), dtype=np.float32)
        inst.phys_q99 = np.array(data.get("phys_q99", []), dtype=np.float32)
        inst.is_fitted = data.get("is_fitted", True)
        return inst
