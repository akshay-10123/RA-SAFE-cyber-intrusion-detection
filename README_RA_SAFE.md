# RA-SAFE: Reliability-Aware Selective Fusion for UAV Cyber-Physical Intrusion Detection

**A Research-Grade Intrusion Detection Architecture with Input Quality Estimation, Predictive Uncertainty Quantification, Adaptive Softmax Gating, and Calibrated Selective Abstention.**

---

## 1. Problem Formulation & Motivation

Modern Unmanned Aerial Vehicles (UAVs) rely on a coupled cyber-physical architecture:
1. **Cyber Subsystem**: Wireless telemetry, 802.11 Wi-Fi Command and Control (C2) framing, sequence counters, and signal-to-noise metrics.
2. **Physical Subsystem**: Onboard flight kinematics, attitude telemetry (pitch, roll, yaw), accelerometers, 3-axis velocities, and Time-of-Flight (ToF) ground distance.

Intrusions can target either or both domains:
- **De-authentication DoS** attacks flood the wireless spectrum, severing the C2 link.
- **Replay attacks** inject stale valid packets, causing control latency.
- **Evil Twin access points** impersonate ground stations.
- **False Data Injection (FDI)** tampers with flight telemetry and sensor feedback.

### The Limitation of Prior Work (CP-Attn-Lite)
Existing dual-branch architectures such as **CP-Attn-Lite** utilize a standard softmax attention gate:
$$[w_c, w_p] = \text{Softmax}\left(\mathbf{W}_g [z_c, z_p] + b_g\right)$$
While CP-Attn-Lite provides lightweight attention weights, **attention is not an indicator of data reliability**:
1. When wireless channels experience heavy packet loss, jitter, or hostile RF jamming, standard attention may still assign high weights to corrupted cyber inputs.
2. When physical sensors experience bias, linear drift, or stuck-at freeze faults, CP-Attn-Lite has no objective measurement of sensor validity.
3. Softmax predictions are notoriously overconfident and cannot abstain. In safety-critical aviation, forcing hard predictions under severe signal corruption leads to catastrophic false alarms or undetected takeovers.

---

## 2. RA-SAFE Architecture Overview

RA-SAFE extends CP-Attn-Lite by introducing three synergistic research modules:

```
                           Raw Input (32 Cyber-Physical Features)
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
     Cyber Branch (21 Features)                               Physical Branch (11 Features)
               │                                                           │
        Conv1D(32, k=3)                                             Conv1D(32, k=3)
        BatchNorm + GAP                                             BatchNorm + GAP
               │                                                           │
          z_c ∈ R^32                                                  z_p ∈ R^32
               ├──► Cyber Aux Classifier                                   ├──► Phys Aux Classifier
               │    Entropy U_c ∈ [0, 1]                                   │    Entropy U_p ∈ [0, 1]
               │                                                           │
     [Cyber Quality Estimator]                                   [Phys Quality Estimator]
        Quality Q_c ∈ [0, 1]                                        Quality Q_p ∈ [0, 1]
               │                                                           │
     Branch Reliability R_c                                      Branch Reliability R_p
     R_c = Q_c · (1 - U_c)                                       R_p = Q_p · (1 - U_p)
               │                                                           │
               └─────────────────────────────┬─────────────────────────────┘
                                             ▼
                             [Reliability Adaptive Gate]
                     u_gate = [z_c, z_p, Q_c, Q_p, U_c, U_p, R_c, R_p] ∈ R^70
                     Dense(16, ReLU) -> Dense(2) -> Softmax
                                      [w_c, w_p]
                                             │
                                             ▼
                       Weighted Fusion: z_fused = [w_c·z_c || w_p·z_p] ∈ R^64
                                             │
                                  [Squeeze-and-Excitation]
                                             │
                                  [Classification Head]
                                       Logits v ∈ R^5
                                             │
                                [Temperature Scaling: T]
                                  Calibrated Probs p ∈ R^5
                                             │
                           [Selective Prediction Engine]
                 Decision: ACCEPT (Predicted Attack) or ABSTAIN
```

---

## 3. Mathematical Specifications

### Module A: Modality Reliability Estimator
Computes non-parametric empirical quality scores $Q_c, Q_p \in [0, 1]$ using strictly inference-time properties (zero attack labels, no data leakage):

- **Cyber Quality ($Q_c$)**:
  Evaluates normalized sequence counter continuity, signal-to-noise ratio plausibility, and Mahalanobis statistical distance $d_c$ from clean training baselines:
  $$d_c(\mathbf{x}_c) = \frac{1}{21}\sum_{i=1}^{21} \min\left(\left(\frac{x_{c,i} - \mu_{c,i}}{\sigma_{c,i}}\right)^2, 25.0\right)$$
  $$Q_c = \text{clip}\left(0.50 e^{-0.35 d_c} + 0.25 q_{\text{snr}} + 0.25 q_{\text{comp}}, 0.01, 0.99\right)$$

- **Physical Quality ($Q_p$)**:
  Evaluates kinematic acceleration magnitude bounds ($\|\mathbf{a}\| = \sqrt{a_{gx}^2 + a_{gy}^2 + a_{gz}^2}$), velocity plausibility, and sensor distribution consistency:
  $$d_p(\mathbf{x}_p) = \frac{1}{11}\sum_{j=1}^{11} \min\left(\left(\frac{x_{p,j} - \mu_{p,j}}{\sigma_{p,j}}\right)^2, 25.0\right)$$
  $$Q_p = \text{clip}\left(0.50 e^{-0.35 d_p} + 0.25 q_{\text{acc}} + 0.25 q_{\text{vel}}, 0.01, 0.99\right)$$

### Module B: Uncertainty-Aware Adaptive Fusion
Each branch features an auxiliary classification head that outputs modality-specific predictions $\hat{\mathbf{y}}_c$ and $\hat{\mathbf{y}}_p$.
1. **Branch Predictive Uncertainty**:
   $$U_c = -\frac{1}{\ln 5} \sum_{k=1}^5 \hat{y}_{c,k} \ln(\hat{y}_{c,k} + \epsilon), \quad U_p = -\frac{1}{\ln 5} \sum_{k=1}^5 \hat{y}_{p,k} \ln(\hat{y}_{p,k} + \epsilon)$$
2. **Unified Branch Reliability**:
   $$R_c = Q_c \cdot (1 - U_c) \in [0, 1], \quad R_p = Q_p \cdot (1 - U_p) \in [0, 1]$$
3. **Adaptive Fusion Gate ($g$)**:
   $$\mathbf{u}_{\text{gate}} = [\mathbf{z}_c \,\|\, \mathbf{z}_p \,\|\, Q_c \,\|\, Q_p \,\|\, U_c \,\|\, U_p \,\|\, R_c \,\|\, R_p] \in \mathbb{R}^{70}$$
   $$[w_c, w_p] = \text{Softmax}\left(\mathbf{W}_{g2} \cdot \text{ReLU}(\mathbf{W}_{g1} \mathbf{u}_{\text{gate}} + \mathbf{b}_{g1}) + \mathbf{b}_{g2}\right)$$
   $$\mathbf{z}_{\text{fused}} = [w_c \mathbf{z}_c \,\|\, w_p \mathbf{z}_p] \in \mathbb{R}^{64}$$

### Module C: Selective Prediction & Calibration
1. **Temperature Scaling**: Logits $\mathbf{v}$ are scaled by validation-optimized temperature $T=0.8553$:
   $$p_k = \frac{\exp(v_k / T)}{\sum_{j=1}^5 \exp(v_j / T)}, \quad \kappa = \max_k p_k$$
2. **Decision Rule**:
   $$\text{Decision} = \begin{cases} 
   \text{ACCEPTED} & \text{if } \kappa \ge \tau_{\text{conf}} \text{ and } \max(R_c, R_p) \ge \tau_{\text{rel}} \\
   \text{ABSTAINED} & \text{otherwise}
   \end{cases}$$
   Optimal thresholds determined on validation data: $\tau_{\text{conf}} = 0.65, \tau_{\text{rel}} = 0.25$.

---

## 4. Empirical Evaluation Results

Evaluated on the held-out stratified test set (8,218 samples):

| Metric | Baseline CP-Attn-Lite | RA-SAFE (Proposed) |
| :--- | :---: | :---: |
| **Model Parameters** | 3,095 | **4,465** |
| **Inference Latency** | 0.084 ms | **0.063 ms** |
| **Test Accuracy (Unfiltered)** | 79.36% | **78.38%** |
| **Macro F1 Score** | 80.53% | **79.28%** |
| **Weighted F1 Score** | 78.86% | **77.68%** |
| **Expected Calibration Error (ECE)** | 0.0480 | **0.0202** (-58% error) |
| **Brier Score** | 0.3120 | **0.2841** |
| **Selective Coverage** | 100.0% (Forced) | **43.89%** |
| **Accepted Set Accuracy** | 79.36% | **94.29%** (+14.9%) |
| **Selective Risk (Error on Accepted)** | 20.64% | **5.71%** (-72% risk) |

### Robustness Stress Benchmarks Across Degradations

| Scenario | CP-Attn-Lite Acc | RA-SAFE Raw Acc | RA-SAFE Coverage | RA-SAFE Accepted Acc | $w_c$ (Cyber) | $w_p$ (Phys) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal (Clean)** | 78.6% | 78.4% | 43.9% | **94.3%** | 0.66 | 0.34 |
| **Packet Loss 10%** | 74.6% | 74.8% | 39.2% | **94.7%** | 0.65 | 0.35 |
| **Packet Loss 20%** | 68.6% | 69.5% | 35.3% | **94.2%** | 0.64 | 0.36 |
| **RF Jamming** | 71.8% | 62.6% | 43.0% | **89.6%** | **0.61** $\downarrow$ | **0.39** $\uparrow$ |
| **Sensor Noise ($\sigma=1.0$)** | 71.6% | 72.1% | 45.9% | **87.9%** | 0.68 | 0.32 |
| **Sensor Noise ($\sigma=2.0$)** | 55.1% | 62.2% | 55.2% | **72.2%** | 0.69 | 0.31 |
| **Sensor Freeze (Stuck)** | 53.9% | 52.5% | 47.7% | **62.6%** | 0.68 | 0.32 |
| **Simultaneous Degradation** | 38.3% | 47.4% | 45.6% | **66.6%** | 0.65 | 0.35 |

---

## 5. Critical Dataset Provenance Disclosure

> [!WARNING]
> In the public IEEE T-ITS UAV dataset repository, the 11 physical telemetry features (`pitch`, `roll`, `yaw`, `vgx`, `vgy`, `vgz`, `flight_time`, `agx`, `agy`, `agz`, `tof`) and certain radiotap indicators were synthetically generated in `scripts/prepare_dataset.py` using Gaussian distributions conditioned on the attack indicator ($attack \times \delta$).
> 
> **Scientific Integrity Requirement:**
> - These results demonstrate the mathematical and algorithmic validity of reliability-aware selective fusion on the standardized benchmark.
> - They **must not** be claimed as validated against genuine flight telemetry or deployed physical hardware until verified on an uncorrupted flight telemetry recorder.

---

## 6. How to Run

### Environment Setup
```powershell
$py = "C:\uav-venv\Scripts\python.exe"
cd "C:\Users\HP\OneDrive\Documents\Cyber-Physical-Intrusion-Detection-System-for-Unmanned-Aerial-Vehicles"
```

### 1. Train RA-SAFE Architecture
```powershell
& $py training/train_ra_safe.py --epochs 12
```

### 2. Run Robustness & Degradation Benchmark
```powershell
& $py evaluation/robustness_tests.py
```

### 3. Run Unit & Integration Test Suite
```powershell
& $py evaluation/test_ra_safe_unit.py
```

### 4. Launch Streamlit Interactive Console
```powershell
& $py -m streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`:
- Navigate to **"RA-SAFE Detection"** for interactive single-row inference, real-time corruption injection, and operator playbook diagnostics.
- Navigate to **"RA-SAFE Robustness"** for full degradation comparison tables and stress-test charts.
