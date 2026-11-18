"""
Corruption and Degradation Simulator for UAV Cyber-Physical Robustness Stress Testing.

Simulates:
  1. Network Degradation: Packet loss / dropouts, sequence jitter, RF jamming / SNR collapse.
  2. Physical Sensor Degradation: Gaussian IMU noise, calibration bias, sensor drift, stuck-at freeze.
  3. Simultaneous Cyber-Physical Degradation.

Strictly used ONLY at test/inference time to assess robustness and abstention behavior.
NEVER included in training data to prevent methodological leakage.
"""
from __future__ import annotations

import numpy as np


def apply_packet_loss(x_cyber: np.ndarray, drop_rate: float = 0.10, seed: int = 42) -> np.ndarray:
    """Simulates Wi-Fi packet drops by randomly zeroing out features with drop_rate."""
    rng = np.random.default_rng(seed)
    corrupted = np.copy(x_cyber)
    mask = rng.random(size=corrupted.shape) < drop_rate
    corrupted[mask] = 0.0
    return corrupted


def apply_rf_jamming(x_cyber: np.ndarray, snr_idx: int = 19, sig_idx: int = 16) -> np.ndarray:
    """Simulates hostile RF jamming by crashing SNR and signal quality indicators."""
    corrupted = np.copy(x_cyber)
    # Crash SNR to severe negative z-score
    if corrupted.ndim == 2:
        corrupted[:, snr_idx] = -3.5
        corrupted[:, sig_idx] = -3.0
    elif corrupted.ndim == 3:
        corrupted[:, snr_idx, 0] = -3.5
        corrupted[:, sig_idx, 0] = -3.0
    return corrupted


def apply_sensor_noise(x_phys: np.ndarray, noise_std: float = 1.0, seed: int = 42) -> np.ndarray:
    """Simulates high-vibration / noisy IMU sensors with zero-mean additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=x_phys.shape).astype(np.float32)
    return x_phys + noise


def apply_sensor_bias(x_phys: np.ndarray, bias_magnitude: float = 2.0) -> np.ndarray:
    """Simulates constant calibration drift / uncalibrated accelerometer bias."""
    corrupted = np.copy(x_phys)
    corrupted += bias_magnitude
    return corrupted


def apply_sensor_stuck(x_phys: np.ndarray, freeze_val: float = 0.0) -> np.ndarray:
    """Simulates complete sensor freeze / hardware lockup (stuck-at fault)."""
    corrupted = np.full_like(x_phys, freeze_val)
    return corrupted


def simulate_degradation(
    x_cyber: np.ndarray,
    x_phys: np.ndarray,
    mode: str = "nominal",
    intensity: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unified degradation dispatcher:
      - 'nominal': untouched inputs
      - 'cyber_packet_loss': random packet zeroing (intensity = drop rate, e.g. 0.15)
      - 'cyber_jamming': severe SNR / signal collapse
      - 'phys_noise': additive Gaussian IMU noise (intensity = std, e.g. 1.5)
      - 'phys_stuck': sensor freeze
      - 'phys_bias': sensor offset
      - 'simultaneous': both packet loss and IMU noise
    """
    xc = np.copy(x_cyber)
    xp = np.copy(x_phys)

    if mode == "nominal":
        return xc, xp
    elif mode == "cyber_packet_loss":
        drop_rate = min(max(intensity, 0.05), 0.50)
        xc = apply_packet_loss(xc, drop_rate=drop_rate, seed=seed)
    elif mode == "cyber_jamming":
        xc = apply_rf_jamming(xc)
    elif mode == "phys_noise":
        xp = apply_sensor_noise(xp, noise_std=intensity, seed=seed)
    elif mode == "phys_stuck":
        xp = apply_sensor_stuck(xp)
    elif mode == "phys_bias":
        xp = apply_sensor_bias(xp, bias_magnitude=intensity * 2.0)
    elif mode == "simultaneous":
        xc = apply_packet_loss(xc, drop_rate=0.20, seed=seed)
        xc = apply_rf_jamming(xc)
        xp = apply_sensor_noise(xp, noise_std=1.5, seed=seed)
    else:
        raise ValueError(f"Unknown degradation mode: {mode}")

    return xc, xp
