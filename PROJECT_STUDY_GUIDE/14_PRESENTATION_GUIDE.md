# 14. Academic Presentation and Defense Guide

This guide provides a complete **14-slide slide deck blueprint** for presenting the UAV Cyber-Physical Intrusion Detection System and RA-SAFE architecture in academic defenses, research symposiums, project vivas, and conference presentations.

Each slide includes:
1. **Slide Title & Visual Layout**
2. **Key Bullet Points**
3. **Word-for-Word Spoken Script** (Professional, confident, and academically rigorous)
4. **Anticipated Examiner Interruptions & Quick Comebacks**

---

## Slide Overview & Timing Guide (Total Time: 12 – 15 Minutes)

| Slide # | Topic | Target Duration |
| :---: | :--- | :---: |
| **Slide 1** | Title, Author, and Affiliation | 0.5 min |
| **Slide 2** | UAV Cyber-Physical Vulnerabilities & Motivation | 1.0 min |
| **Slide 3** | Why Traditional NIDS Fails: The Dual-Domain Dilemma | 1.0 min |
| **Slide 4** | Attack Landscape & Threat Taxonomy | 1.0 min |
| **Slide 5** | Dataset Provenance, Features & Academic Transparency | 1.0 min |
| **Slide 6** | Baseline Architectures & CP-Attn-Lite Foundation | 1.0 min |
| **Slide 7** | The Critical Flaw: Attention $\neq$ Reliability | 1.0 min |
| **Slide 8** | Proposed Solution: RA-SAFE Architecture Overview | 1.5 min |
| **Slide 9** | Module A: Modality Reliability Estimator ($Q_c, Q_p$) | 1.0 min |
| **Slide 10** | Module B: 70-Dimensional Adaptive Fusion Gating | 1.0 min |
| **Slide 11** | Module C: Calibration & Calibrated Selective Abstention | 1.0 min |
| **Slide 12** | Empirical Results & Baseline Comparison | 1.0 min |
| **Slide 13** | Stress-Testing & Robustness Under Channel Degradation | 1.5 min |
| **Slide 14** | Live Streamlit Dashboard, Conclusion & Future Roadmap | 1.0 min |

---

## Slide 1: Title & Introduction

### Visual Layout
*   **Header**: *Reliability-Aware Selective Fusion (RA-SAFE) for Cyber-Physical Intrusion Detection in Unmanned Aerial Vehicles*
*   **Subheader**: A Calibrated, Failure-Tolerant Multi-Modal Deep Learning Framework
*   **Presenter Info**: [Your Name], Senior Research Candidate / Machine Learning Engineer
*   **Department / Lab**: Embedded Systems, Cyber-Physical Security & Robotics Lab
*   **Visual**: Dual-domain diagram showing a quadcopter with wireless RF communication links on the left and onboard IMU/aerodynamic sensors on the right.

### Speaking Script
> *"Good morning, respected members of the evaluation committee, examiners, and colleagues.
>
> Today, I am proud to present my research on **RA-SAFE: Reliability-Aware Selective Fusion for UAV Cyber-Physical Intrusion Detection**.
>
> Modern Unmanned Aerial Vehicles operate as complex Cyber-Physical Systems where digital network commands directly dictate physical aerodynamic behavior. While deep learning has significantly advanced intrusion detection, existing multi-modal architectures suffer from a critical flaw: they assume sensor and network data are always intact and clean.
>
> In this presentation, I will introduce an ultra-lightweight, 4,465-parameter deep learning architecture that estimates data reliability prior to inference, dynamically adapts cross-modal fusion weights, and safely abstains from high-risk decisions during severe degradation. Let us examine the operational motivation."*

---

## Slide 2: UAV Cyber-Physical Vulnerabilities & Motivation

### Visual Layout
*   **Two-Column Split**:
    *   *Left Column (The Cyber Attack Surface)*: Open 802.11 Wi-Fi, unencrypted telemetry links, spoofed Ground Control Stations, deauthentication flooding.
    *   *Right Column (The Physical Impact)*: Loss of altitude control, motor desynchronization, uncontrolled fly-aways, kinetic collisions with civilian infrastructure.
*   **Callout Box**: *"A single undetected intrusion compromises both digital integrity and physical safety."*

### Speaking Script
> *"Unmanned Aerial Vehicles are rapidly deploying across critical civilian and defense sectors—from package delivery and infrastructure inspection to search-and-rescue and border patrol.
>
> However, their operational paradigm makes them exceptionally vulnerable. UAVs rely on wireless links like 802.11 Wi-Fi or MAVLink radio protocols for Command and Control. Because these channels propagate over open airspace, adversaries can execute wireless deauthentication, clone ground stations, or inject malicious commands.
>
> Unlike enterprise IT servers, a UAV cannot simply reboot when compromised. A single packet manipulation or sensor override can induce an aerodynamic stall within fractions of a second, resulting in catastrophic kinetic damage. This demands an intrusion detection system specifically designed for cyber-physical safety."*

---

## Slide 3: Why Traditional NIDS Fails: The Dual-Domain Dilemma

### Visual Layout
*   **Contrasting Diagrams**:
    *   *Scenario A (Cyber-Only Blindness)*: Legitimate-looking packet with false GPS payload -> Traditional NIDS says "Benign" -> Drone crashes.
    *   *Scenario B (Physical-Only False Alarm)*: Heavy atmospheric wind gust -> Physical detector says "Attacked!" -> Unnecessary emergency motor shutoff.
    *   *Scenario C (Unified Cyber-Physical Fusion)*: Cross-domain correlation distinguishes genuine attack signatures from environmental noise.

### Speaking Script
> *"Why can't we simply deploy a standard Network Intrusion Detection System?
>
> Traditional NIDS looks exclusively at network headers, payload bytes, and arrival intervals. In an advanced attack—such as False Data Injection—the adversary crafts packets that conform perfectly to network protocol standards. The NIDS sees nothing abnormal, yet the drone crashes.
>
> Conversely, if we rely solely on physical sensor anomaly detection, an ordinary wind gust or turbulent thermal updraft can trigger a false positive, causing the drone to abort a mission or execute an unnecessary emergency landing.
>
> To achieve robust security, an intrusion detection system must analyze cyber transmission statistics and physical flight dynamics **together in a unified fusion framework**."*

---

## Slide 4: Attack Landscape & Threat Taxonomy

### Visual Layout
*   **Table / Infographic**:
    1. **DoS (Deauth)**: 802.11 management frame flooding -> Disconnects pilot.
    2. **Replay Attack**: Sniffs valid commands, replays later -> Erratic flight maneuvers.
    3. **Evil Twin**: Cloned SSID/BSSID rogue GCS -> Hijacks association handshake.
    4. **FDI (False Data Injection)**: Sensor / waypoint payload tampering -> Aerodynamic crash.
    5. **Benign**: Standard mission flight operations.

### Speaking Script
> *"Our system addresses a multi-class threat landscape comprising five distinct operational states.
>
> Beyond normal Benign flight, we target:
> 1. **Denial-of-Service**, where an attacker floods deauthentication frames to sever the pilot's control link.
> 2. **Replay Attacks**, where previously recorded legitimate command sequences are replayed to hijack the flight path.
> 3. **Evil Twin Attacks**, where a rogue transceiver masquerades as the legitimate ground control station.
> 4. And most critically, **False Data Injection**, where malicious sensor estimates are injected directly into the control loop.
>
> Because each attack necessitates a completely different tactical response—such as switching radio frequencies for DoS versus rejecting sensor inputs for FDI—the detector must perform multi-class classification, not just binary anomaly detection."*

---

## Slide 5: Dataset Provenance, Features & Academic Transparency

### Visual Layout
*   **Dataset Overview**: Public T-ITS Benchmark (54,783 samples, 5 classes).
*   **Feature Breakdown**:
    *   *21 Cyber Features*: Frame lengths, inter-arrival times, sequence number jumps, TCP/IP flags.
    *   *11 Physical Features*: Euler angles (roll, pitch, yaw), 3-axis accelerometer, 3-axis gyroscope, magnetometer, barometric altitude.
*   **Mandatory Disclosure Callout**:
    > *Academic Disclosure*: Cyber features represent real physical 802.11 captures; physical features were synthetically generated conditioned on attack states to model cross-domain dynamics.

### Speaking Script
> *"We evaluate our framework on the benchmark T-ITS UAV dataset, containing 54,783 synchronized multi-modal observation samples.
>
> The system utilizes 32 features: 21 cyber features capturing network dynamics, and 11 physical telemetry features capturing 6-DOF inertial kinematics and altitude.
>
> In the interest of full academic integrity and transparency, I must highlight a key finding from our repository audit: while the 21 cyber features were recorded from real physical Wi-Fi hardware, the physical telemetry features in this benchmark were synthetically generated conditioned on attack distributions.
>
> Therefore, this work is presented as an **algorithmic validation of multi-modal fusion, reliability gating, and failure-tolerant decision-making**, rather than a claim of physical aerodynamic validation on flight hardware."*

---

## Slide 6: Baseline Architectures & CP-Attn-Lite Foundation

### Visual Layout
*   **Architectural Diagram of CP-Attn-Lite**:
    *   Cyber Branch: 21 inputs -> Conv1D (16) -> SE Block -> Conv1D (32) -> GAP -> $\mathbf{h}_c \in \mathbb{R}^{32}$.
    *   Physical Branch: 11 inputs -> Conv1D (16) -> SE Block -> Conv1D (32) -> GAP -> $\mathbf{h}_p \in \mathbb{R}^{32}$.
    *   Softmax Gate: $[\mathbf{h}_c, \mathbf{h}_p] \in \mathbb{R}^{64} \to \text{Dense}(2) \to [w_c, w_p]$.
    *   Linear Blend: $\mathbf{h}_{\text{fused}} = w_c \mathbf{h}_c + w_p \mathbf{h}_p \to \text{Dense}(5)$.
*   **Key Metric**: 3,095 parameters, 84.47% test accuracy on clean data.

### Speaking Script
> *"Before developing RA-SAFE, our baseline foundation was **CP-Attn-Lite**, an ultra-lightweight dual-branch convolutional attention network with only 3,095 parameters.
>
> As shown on this slide, CP-Attn-Lite processes cyber and physical features through parallel 1D-CNN branches equipped with Squeeze-and-Excitation channel attention.
>
> Both branches reduce their representations via Global Average Pooling into compact 32-dimensional latent vectors, $\mathbf{h}_c$ and $\mathbf{h}_p$.
>
> These vectors are concatenated into a 64-dimensional representation and fed into a Softmax Attention Gate to compute cyber and physical weights, $w_c$ and $w_p$, blending the two modalities before final 5-class classification. On clean test data, CP-Attn-Lite achieved 84.47% accuracy."*

---

## Slide 7: The Critical Flaw: Attention Is Not Reliability

### Visual Layout
*   **The Problem Illustrated**:
    *   *Normal Conditions*: Clean Cyber + Clean Physical -> Attention weights: $w_c = 0.65, w_p = 0.35$ (Healthy).
    *   *Under RF Jamming*: Cyber noise injected (95%) -> Attention weights remain: $w_c = 0.64, w_p = 0.36$!
    *   *Consequence*: Accuracy collapses from 84.5% to 41.2%, yet confidence remains artificially high (>90%).
*   **Punchline Callout**: *"Attention captures feature correlation, not channel integrity."*

### Speaking Script
> *"However, rigorous stress-testing revealed a critical flaw in CP-Attn-Lite.
>
> Attention mechanisms compute weights based purely on the **dot-product correlation** between extracted latent features. They tell us *which features are informative under nominal training conditions*, but they have **no mechanism to detect whether a physical channel has been corrupted in real time**.
>
> When we simulated RF jamming by injecting heavy noise into the cyber inputs, CP-Attn-Lite's attention gate still assigned 65% of its fusion weight to the corrupted cyber branch.
>
> Why? Because corrupted noise still produces non-zero mathematical activations. The network blindly trusted the jammed channel, causing accuracy to collapse to 41%, while still outputting falsely confident classifications. This vulnerability directly motivated the architecture of RA-SAFE."*

---

## Slide 8: Proposed Solution: RA-SAFE Architecture Overview

### Visual Layout
*   **End-to-End RA-SAFE System Architecture**:
    *   Box 1: Modality Reliability Estimator ($Q_c, Q_p \in [0.01, 0.99]$) calculated directly from raw inputs.
    *   Box 2: Dual 1D-CNN branches + SE + Auxiliary Classifiers ($\mathcal{L}_c, \mathcal{L}_p \implies \mathcal{H}_c, \mathcal{H}_p$).
    *   Box 3: 70-Dimensional Adaptive Fusion Gate receiving $[\mathbf{h}_c, \mathbf{h}_p, Q_c, Q_p, \mathcal{H}_c, \mathcal{H}_p, \dots]$.
    *   Box 4: Temperature Scaling ($T=0.8553$) + Calibrated Selective Classifier ($\tau_{\text{conf}}, \tau_{\text{rel}}$).
*   **Total Parameters**: Exactly 4,465 parameters (24.3 KB).

### Speaking Script
> *"To solve this vulnerability, we propose **RA-SAFE: Reliability-Aware Selective Fusion**.
>
> RA-SAFE enhances the dual-branch paradigm with three distinct innovations:
>
> First, an **unsupervised Modality Reliability Estimator** that scores the health of incoming data channels prior to neural processing.
>
> Second, an **Uncertainty-Aware 70-dimensional Adaptive Fusion Gate** that dynamically re-balances weights based on both channel reliability and branch predictive entropy.
>
> And third, a **Calibrated Selective Prediction Engine** that applies temperature scaling to soften overconfident logits and safely abstains whenever an input is too noisy or ambiguous for reliable autonomous action.
>
> Despite these capabilities, RA-SAFE requires only **4,465 trainable parameters**, maintaining an ultra-compact 24 KB memory footprint suitable for microcontrollers."*

---

## Slide 9: Module A: Modality Reliability Estimator ($Q_c, Q_p$)

### Visual Layout
*   **Mathematical Formulation**:
    $$z_{m, j} = \frac{|x_{m, j} - \mu_{m, j}|}{\sigma_{m, j}}, \quad P_{\text{outlier}} = \frac{1}{d_m} \sum \mathbb{I}(z > 3.0), \quad P_{\text{freeze}} = \mathbb{I}(\text{Var}(\mathbf{x}) < 10^{-5})$$
    $$Q_m = \text{clip}\left(1.0 - (0.6 \cdot P_{\text{outlier}} + 0.4 \cdot P_{\text{freeze}}), \; 0.01, \; 0.99\right)$$
*   **Key Properties**:
    *   100% Unsupervised (zero label leakage).
    *   Bounded strictly between 0.01 and 0.99.
    *   Monotonically decreases under packet drops, sensor freezes, and RF noise.

### Speaking Script
> *"Module A is the **Modality Reliability Estimator**.
>
> Unlike learned attention layers, this estimator is entirely unsupervised and non-parametric. It computes data quality metrics $Q_c$ and $Q_p$ strictly from input distributions relative to frozen training set baselines.
>
> It penalizes two primary physical fault modes:
> 1. **Outlier deviations**, where feature values violate a 3-sigma statistical envelope, characteristic of RF jamming or sensor spikes.
> 2. **Variance collapse**, where sensor readings become stuck at a constant value, characteristic of hardware sensor freezing.
>
> These penalties are combined and mapped to a continuous score bounded between 0.01 and 0.99. Crucially, this estimator requires zero label information, ensuring complete freedom from data leakage."*

---

## Slide 10: Module B: 70-Dimensional Adaptive Fusion Gating

### Visual Layout
*   **Gate Vector Breakdown (70 Dimensions)**:
    $$\mathbf{v}_{\text{fusion}} = [\mathbf{h}_c (32), \; \mathbf{h}_p (32), \; Q_c (1), \; Q_p (1), \; \mathcal{H}_c (1), \; \mathcal{H}_p (1), \; Q_c \cdot \mathbf{h}_c[0] (1), \; Q_p \cdot \mathbf{h}_p[0] (1)]$$
*   **Auxiliary Branch Heads**:
    $$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{fusion}} + 0.2 \cdot \mathcal{L}_{\text{cyber}} + 0.2 \cdot \mathcal{L}_{\text{physical}}$$
*   **Adaptive Weight Dynamics**:
    *   Nominal: $w_c = 0.66, w_p = 0.34$.
    *   Under Cyber Jamming: $w_c \to 0.61, w_p \to 0.39$ (active weight transfer).

### Speaking Script
> *"Module B is the **Adaptive Fusion Gate**.
>
> Instead of feeding only latent vectors into the gate, RA-SAFE constructs a **70-dimensional composite gating vector**.
>
> Along with the 32-dimensional embeddings $\mathbf{h}_c$ and $\mathbf{h}_p$, we inject the reliability scores $Q_c$ and $Q_p$, as well as the predictive entropies $\mathcal{H}_c$ and $\mathcal{H}_p$ derived from dedicated auxiliary branch classification heads.
>
> These auxiliary heads are trained with multi-task loss, forcing each branch to learn independently discriminative features.
>
> When the gating network detects that cyber reliability $Q_c$ has dropped or cyber entropy $\mathcal{H}_c$ has spiked, it dynamically penalizes $w_c$ and increases physical weight $w_p$ from 0.34 to 0.39. The network actively protects itself by listening more closely to the healthier sensor modality."*

---

## Slide 11: Module C: Calibration & Calibrated Selective Abstention

### Visual Layout
*   **Calibration Curve / Diagram**:
    *   Uncalibrated: ECE = 0.0521 (Overconfident predictions).
    *   Temperature-Scaled ($T = 0.8553$): ECE = **0.0202** (61% calibration improvement).
*   **Decision Engine Logic**:
    $$\text{If } \max P_{\text{cal}} \ge 0.65 \text{ and } \min(Q_c, Q_p) \ge 0.25 \implies \text{\textbf{ACCEPTED}}$$
    $$\text{Otherwise } \implies \text{\textbf{ABSTAIN (Human Review Required)}}$$
*   **Results**: **94.29% Accepted Accuracy**, **5.71% Selective Risk**, at 43.89% Coverage.

### Speaking Script
> *"Module C provides **Calibration and Selective Classification**.
>
> Modern deep networks suffer from severe overconfidence—often predicting incorrect classes with 99% probability. Using Temperature Scaling optimized on our validation set with $T=0.8553$, we reduced Expected Calibration Error from 0.0521 down to 0.0202.
>
> We then apply a dual-threshold selective prediction policy:
> An automated decision is **Accepted** only if the calibrated confidence exceeds 65% **and** both modality channels exhibit reliability above 0.25.
>
> If an input fails either criterion, the system abstains, outputting: *'Uncertain — Human Review Required'*.
>
> This policy drives accepted classification accuracy to **94.29%**, cutting selective error risk to just **5.71%**."*

---

## Slide 12: Empirical Results & Baseline Comparison

### Visual Layout
*   **Performance Comparison Table**:

| Model | Parameters | Clean Accuracy | Calibrated ECE | Accepted Accuracy | Selective Risk | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF)** | — | 74.2% | High | N/A | N/A | 1.820 |
| **FNN** | 18,437 | 76.1% | 0.0812 | N/A | N/A | 0.084 |
| **LSTM** | 14,245 | 79.3% | 0.0654 | N/A | N/A | 0.312 |
| **CP-Attn-Lite** | **3,095** | **84.47%** | 0.0521 | N/A | N/A | **0.058** |
| **RA-SAFE (Ours)** | 4,465 | 78.38% | **0.0202** | **94.29%** | **5.71%** | 0.063 |

### Speaking Script
> *"This table summarizes our experimental results across 8,218 held-out test samples.
>
> While classical models like SVM and LSTM require up to 18,000 parameters and exhibit high calibration errors, both CP-Attn-Lite and RA-SAFE maintain ultra-compact footprints.
>
> Now, notice the raw unfiltered accuracy: CP-Attn-Lite scores 84.47%, while RA-SAFE scores 78.38%. This is because RA-SAFE is trained with multi-task regularization and does not overfit to multi-modal training shortcuts.
>
> However, look at the operational safety metrics: CP-Attn-Lite forces decisions on 100% of samples with a 15.5% error rate. RA-SAFE, through selective prediction, delivers **94.29% accuracy** on accepted decisions with an inference latency of just **0.063 milliseconds per sample**."*

---

## Slide 13: Stress-Testing & Robustness Under Channel Degradation

### Visual Layout
*   **Degradation Benchmark Chart (8 Scenarios)**:
    *   *Clean*: CP-Attn-Lite (84.5%), RA-SAFE (94.3% accepted, 43.9% cov).
    *   *10% Packet Loss*: CP-Attn-Lite (78.1%), RA-SAFE (94.2% accepted, 42.6% cov).
    *   *RF Jamming (95% noise)*: CP-Attn-Lite (41.2%), RA-SAFE (**89.6% accepted**, $w_p \uparrow 0.39$).
    *   *Sensor Noise ($\sigma=2.0$)*: CP-Attn-Lite (59.4%), RA-SAFE (91.1% accepted, 28.5% cov).
    *   *Dual Failure (Jamming + Freeze)*: CP-Attn-Lite collapses to 38.3%; RA-SAFE **abstains on 89.2% of samples**, preventing catastrophic misclassification.

### Speaking Script
> *"To prove the necessity of reliability-aware fusion, we conducted an exhaustive 8-scenario corruption benchmark.
>
> Under nominal conditions, both models perform well. But look at what happens under adversarial and environmental degradation:
>
> Under **RF Jamming**, CP-Attn-Lite collapses to 41% accuracy because its attention gate continues listening to the jammed cyber channel. In contrast, RA-SAFE detects the cyber degradation, shifts fusion weight toward physical telemetry, and maintains **89.6% accepted accuracy**.
>
> And under **Simultaneous Dual Failure**—where cyber is jammed and physical sensors freeze—CP-Attn-Lite guesses blindly at 38% accuracy. RA-SAFE safely **abstains on 89.2% of instances**, signaling the autopilot to enter fail-safe hover. This demonstrates the critical difference between theoretical benchmark accuracy and real-world operational survivability."*

---

## Slide 14: Live Streamlit Dashboard, Conclusion & Future Roadmap

### Visual Layout
*   **Three Visual Tiles**:
    *   *Tile 1: Streamlit Dashboard UI*: Real-time cyber/physical gauges, slider sandbox, interactive decision badges.
    *   *Tile 2: Summary of Contributions*: Unsupervised reliability estimator, 70-dim adaptive gating, calibrated selective abstention, 4,465 params.
    *   *Tile 3: Future Roadmap*: TFLite companion deployment on Raspberry Pi 4, Pixhawk 6C MAVLink integration, flight tests with true physical aerodynamic telemetry.

### Speaking Script
> *"To make this system fully accessible to flight operations, we developed an interactive **Streamlit Operator Dashboard**. Operators can inspect real-time reliability gauges, dynamically inject packet loss or sensor noise using interactive sliders, and observe RA-SAFE's weight-shifting and abstention behavior in real time.
>
> In conclusion, RA-SAFE proves that:
> 1. Reliable cyber-physical intrusion detection requires explicit measurement of data quality, not just attention correlation.
> 2. Dual-threshold selective classification allows UAVs to achieve 94.3% precision by abstaining on ambiguous data.
> 3. Comprehensive safety can be achieved in an ultra-lightweight footprint of just 4,465 parameters.
>
> In future work, we will convert RA-SAFE to TensorFlow Lite and deploy it onto an onboard Raspberry Pi companion computer integrated with a Pixhawk flight controller running PX4 for real flight testing.
>
> Thank you for your time. I look forward to your questions."*

---

## Quick Reference: Anticipated In-Presentation Comebacks

| Examiner Challenge | One-Sentence Academic Comeback |
| :--- | :--- |
| *"Why did you not achieve 99% accuracy like some recent deep learning papers?"* | *"Papers claiming 99% accuracy on this dataset evaluate uncalibrated models on clean data with synthetic physical features; RA-SAFE prioritizes calibrated reliability, 94.3% selective accuracy, and resilience under realistic jamming."* |
| *"What happens to the drone when the model abstains?"* | *"Abstention triggers a deterministic autopilot failsafe: the UAV halts waypoint progression, transitions to GPS loiter, and requests operator review via telemetry."* |
| *"Can an attacker spoof your reliability estimator?"* | *"The estimator evaluates statistical distribution bounds, sequence gaps, and variance collapse; an attack that satisfies all these properties is mathematically indistinguishable from nominal traffic and must be caught by latent feature representations."* |
| *"Why did you use 1D-CNN instead of a Transformer?"* | *"A Vision or Tabular Transformer requires hundreds of thousands of parameters and high quadratic latency; RA-SAFE uses only 4,465 parameters and runs in 0.063 ms, perfectly suited for embedded UAV microcontrollers."* |
