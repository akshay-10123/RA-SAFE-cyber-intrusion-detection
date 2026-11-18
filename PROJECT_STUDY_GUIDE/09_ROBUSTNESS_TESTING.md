# 09. Robustness Testing & Degradation Stress Benchmarks

This document details the stress-testing methodology, corruption simulation models, and empirical benchmark results comparing RA-SAFE against baseline CP-Attn-Lite under adverse conditions.

---

## 1. Motivation: Why Robustness Testing is Critical

In laboratory benchmarks, deep learning models are evaluated on clean, identically distributed test sets ($i.i.d.$ assumption). In real-world UAV operations, however, operating environments are hostile and dynamic:
- Wi-Fi links experience packet loss, multipath fading, and intentional RF jamming.
- MEMS IMU sensors suffer from motor vibration noise, calibration drift, and physical stuck-at faults.
- Cyber and physical impairments frequently strike concurrently.

A trustworthy UAV IDS must not merely perform well in ideal conditions; it must fail gracefully, adapt its fusion weights, and abstain when data integrity is compromised.

---

## 2. Implemented Degradation Scenarios (`utils/corruption_simulator.py`)

All corruptions are injected **exclusively at test time**. Zero corrupted samples are included in training.

```
                                  Test Sample (Clean)
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
   Cyber Degradations                                                Physical Degradations
   - Packet Loss (10%, 20%)                                          - Gaussian Noise (σ=1.0, 2.0)
   - RF Jamming (SNR collapse)                                       - Calibration Bias
                                                                     - Sensor Freeze (Stuck-at)
         │                                                                   │
         └─────────────────────────────────┬─────────────────────────────────┘
                                           ▼
                             Simultaneous Degradation
                     (20% Packet Loss + Jamming + IMU Noise)
```

---

## 3. Scenario-by-Scenario Technical Breakdown

### Scenario 1: Nominal Conditions (Clean Test Set)
- **Physical Meaning**: Standard uncompromised flight.
- **Implementation**: Baseline test split passed directly without perturbation.
- **CP-Attn-Lite Performance**: Accuracy = **78.6%**, F1 = **80.5%**, $w_c = 0.66, w_p = 0.34$.
- **RA-SAFE Performance**: Raw Accuracy = **78.4%**, Coverage = **43.9%**, **Accepted Accuracy = 94.3%**, $w_c = 0.66, w_p = 0.34$.

---

### Scenario 2: Network Packet Loss (10% and 20%)
- **Physical Meaning**: Wi-Fi frames dropped due to distance, building occlusion, or light buffer congestion.
- **Code Implementation**:
  ```python
  # From utils/corruption_simulator.py:
  mask = rng.random(size=corrupted.shape) < drop_rate
  corrupted[mask] = 0.0
  ```
- **Features Affected**: Randomly zeroes out cyber features across the 21 columns.
- **CP-Attn-Lite Response**: Accuracy degrades from 78.6% down to **74.6%** (10% drop) and **68.6%** (20% drop).
- **RA-SAFE Response**:
  - The Modality Reliability Estimator detects missing features via $q_{\text{comp}}$, lowering $R_c$.
  - Under 10% packet drop: Coverage = **39.2%**, Accepted Accuracy = **94.7%**.
  - Under 20% packet drop: Coverage = **35.3%**, Accepted Accuracy = **94.2%**.
  - **Takeaway**: RA-SAFE prunes out corrupted packets and maintains over 94% accuracy on the accepted set.

---

### Scenario 3: Hostile RF Jamming / SNR Collapse
- **Physical Meaning**: An adversary activates a broadband wireless noise jammer, overwhelming the 2.4 GHz channel.
- **Code Implementation**:
  ```python
  # Crashes SNR (feature 20) and signal quality (feature 17) to severe negative z-scores
  corrupted[:, 19] = -3.5  # wlan_radio.SNR (db)
  corrupted[:, 16] = -3.0  # radiotap.signal_quality
  ```
- **Features Affected**: RadioTap signal quality and physical layer SNR.
- **CP-Attn-Lite Response**: Accuracy drops to **71.8%**. Attention weights remain static ($w_c = 0.65, w_p = 0.35$), failing to recognize that the cyber channel is jammed.
- **RA-SAFE Response**:
  - $q_{\text{snr}}$ collapses, driving $R_c$ down from 0.26 to 0.12.
  - The adaptive gate responds by **shifting fusion weights**: $w_c$ suppresses from 0.66 down to **0.61**, while physical weight $w_p$ increases from 0.34 up to **0.39**.
  - Accepted Set Accuracy remains **89.6%**.

---

### Scenario 4: Physical Sensor Noise ($\sigma = 1.0$ and $\sigma = 2.0$)
- **Physical Meaning**: Severe airframe vibrations, loose motor mounts, or aerodynamic turbulence corrupting IMU readings.
- **Code Implementation**:
  ```python
  noise = rng.normal(0.0, noise_std, size=x_phys.shape).astype(np.float32)
  return x_phys + noise
  ```
- **Features Affected**: Accelerations (`agx`, `agy`, `agz`), velocities (`vgx`, `vgy`, `vgz`), Euler angles (`pitch`, `roll`, `yaw`).
- **CP-Attn-Lite Response**: Accuracy drops sharply: **71.6%** ($\sigma=1.0$) and **55.1%** ($\sigma=2.0$).
- **RA-SAFE Response**:
  - Acceleration norm violations drop $Q_p$ and $R_p$.
  - At $\sigma=1.0$: Accepted Accuracy = **87.9%**.
  - At $\sigma=2.0$: Accepted Accuracy = **72.2%** (vs CP-Attn-Lite's 55.1%).

---

### Scenario 5: Physical Sensor Freeze (Stuck-at Fault)
- **Physical Meaning**: Hardware failure, I2C bus lockup, or power fault causing sensor registers to freeze at constant zero values.
- **Code Implementation**:
  ```python
  corrupted = np.full_like(x_phys, freeze_val)  # Freeze all physical values at 0.0
  ```
- **Features Affected**: All 11 physical telemetry features.
- **CP-Attn-Lite Response**: Accuracy drops to **53.9%**.
- **RA-SAFE Response**:
  - Raw accuracy = 52.5%, but Accepted Accuracy = **62.6%**, with abstentions flagging hardware failure.

---

### Scenario 6: Simultaneous Cyber-Physical Degradation (Dual-Domain Failure)
- **Physical Meaning**: A coordinated attack where an adversary simultaneously jams the Wi-Fi C2 link and induces severe physical sensor disturbances.
- **Code Implementation**: Combines 20% packet drop, RF jamming, and $\sigma=1.5$ Gaussian IMU noise.
- **CP-Attn-Lite Response**: **Catastrophic collapse to 38.3% accuracy** (worse than random guessing among remaining classes). Because CP-Attn-Lite cannot abstain, it issues severe false classifications on 61.7% of all samples.
- **RA-SAFE Response**:
  - $R_c$ and $R_p$ both collapse below threshold $\tau_{\text{rel}} = 0.25$.
  - Rather than outputting blind guesses, the system safely triggers **ABSTENTION** on 54.4% of samples.
  - On the remaining accepted samples, accuracy is **66.6%** (a 28.3 percentage-point improvement over CP-Attn-Lite).
  - Triggers the **Critical Operator Action Playbook**: *"CRITICAL: Dual-domain failure. Cyber link and physical sensor health both severely compromised. Immediately engage hardware Return-to-Home failsafe or safe parachute descent."*

---

## 4. Comprehensive Benchmark Comparison Table

Quantitative results logged from held-out test evaluation (`results/robustness_metrics.json`):

| Test Scenario | Baseline CP-Attn Acc | RA-SAFE Raw Acc | RA-SAFE Coverage | RA-SAFE Accepted Acc | Cyber Weight ($w_c$) | Phys Weight ($w_p$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal (Clean)** | 78.6% | 78.4% | 43.9% | **94.3%** | 0.66 | 0.34 |
| **Packet Loss 10%** | 74.6% | 74.8% | 39.2% | **94.7%** | 0.65 | 0.35 |
| **Packet Loss 20%** | 68.6% | 69.5% | 35.3% | **94.2%** | 0.64 | 0.36 |
| **RF Jamming** | 71.8% | 62.6% | 43.0% | **89.6%** | **0.61** $\downarrow$ | **0.39** $\uparrow$ |
| **Sensor Noise ($\sigma=1.0$)** | 71.6% | 72.1% | 45.9% | **87.9%** | 0.68 | 0.32 |
| **Sensor Noise ($\sigma=2.0$)** | 55.1% | 62.2% | 55.2% | **72.2%** | 0.69 | 0.31 |
| **Sensor Freeze (Stuck)** | 53.9% | 52.5% | 47.7% | **62.6%** | 0.68 | 0.32 |
| **Simultaneous Degr.** | **38.3%** | 47.4% | 45.6% | **66.6%** | 0.65 | 0.35 |

---

## 5. Scientific Summary of Findings

1. **Passive Attention is Fragile**: Attention weights in CP-Attn-Lite remain virtually static under RF jamming ($w_c \approx 0.65$) because the model cannot measure channel noise.
2. **RA-SAFE Adapts Dynamically**: Explicit reliability estimation allows RA-SAFE to penalize degraded modalities in real time, shifting gate weights toward the trustworthy channel.
3. **Abstention Prevents Flight Disasters**: Under simultaneous multi-domain attacks, forced classification is dangerous. Prudent abstention allows RA-SAFE to maintain operational safety and alert human supervisors before catastrophic failure occurs.
