# Master Index: UAV Cyber-Physical Intrusion Detection & RA-SAFE Study Guide

Welcome to the comprehensive technical study guide and research documentation for the **UAV Cyber-Physical Intrusion Detection System** and the **RA-SAFE (Reliability-Aware Selective Fusion)** architecture.

This directory contains **15 master documents** covering the entire project from first principles to production deployment, mathematical proofs, experimental results, and defense preparation.

---

## Master Document Directory

| Doc # | Title | Core Focus & Contents |
| :---: | :--- | :--- |
| **[01](01_PROJECT_OVERVIEW.md)** | [Project Overview & Problem Statement](01_PROJECT_OVERVIEW.md) | CPS basics, UAV vulnerabilities, problem statement, 30s/1m/3m elevator pitches, formal academic abstract. |
| **[02](02_REPOSITORY_STRUCTURE.md)** | [Complete Repository Structure](02_REPOSITORY_STRUCTURE.md) | Full audit of all 42 files and directories: purpose, inputs, processing, outputs, and dependency map. |
| **[03](03_DATASET_AND_PREPROCESSING.md)** | [Dataset & Preprocessing Deep Dive](03_DATASET_AND_PREPROCESSING.md) | Detailed breakdown of all 21 cyber and 11 physical features, synthetic telemetry provenance, scaling, anti-leakage audit. |
| **[04](04_ATTACK_TYPES.md)** | [Attack Types & Threat Vectors](04_ATTACK_TYPES.md) | Deep analysis of Benign, DoS (Deauth), Replay, Evil Twin, and FDI attacks with mathematical signatures and impact analysis. |
| **[05](05_MACHINE_LEARNING_MODELS.md)** | [Baseline Machine Learning Models](05_MACHINE_LEARNING_MODELS.md) | Layer-by-layer architectural blueprints, hyperparameters, and parameters for SVM, FNN, LSTM, 1D-CNN, and Multi-Class CNN. |
| **[06](06_CP_ATTN_LITE.md)** | [CP-Attn-Lite Architecture](06_CP_ATTN_LITE.md) | Dual-branch baseline architecture, Squeeze-and-Excitation, Softmax Attention Gate, 3,095 parameters, and why attention $\neq$ reliability. |
| **[07](07_RA_SAFE_COMPLETE_EXPLANATION.md)** | [RA-SAFE Complete Architecture](07_RA_SAFE_COMPLETE_EXPLANATION.md) | The flagship architecture: Reliability Estimator, 70-dim Adaptive Gating, Temperature Scaling, Selective Prediction, 4,465 params. |
| **[08](08_TRAINING_AND_EVALUATION.md)** | [Training & Evaluation Pipeline](08_TRAINING_AND_EVALUATION.md) | Multi-task loss ($\lambda_1=1.0, \lambda_2=0.2, \lambda_3=0.2$), Adam optimizer, training dynamics, exact test metrics, and confusion matrix. |
| **[09](09_ROBUSTNESS_TESTING.md)** | [Robustness & Degradation Benchmark](09_ROBUSTNESS_TESTING.md) | 8-scenario stress test: packet loss, RF jamming, sensor noise, sensor freeze, and dual failure; CP-Attn-Lite vs. RA-SAFE comparison. |
| **[10](10_STREAMLIT_DASHBOARD.md)** | [Streamlit Dashboard Guide](10_STREAMLIT_DASHBOARD.md) | Operator interface guide: navigation, live corruption sandbox sliders, real-time gauges, and diagnostic inspection workflows. |
| **[11](11_EXECUTION_COMMANDS.md)** | [Execution & Reproduction Commands](11_EXECUTION_COMMANDS.md) | Copy-paste Windows PowerShell commands for data prep, training, unit tests, robustness evaluation, and running the dashboard. |
| **[12](12_RESEARCH_NOVELTY_AND_LIMITATIONS.md)** | [Research Novelty & Limitations](12_RESEARCH_NOVELTY_AND_LIMITATIONS.md) | Inherited foundation vs. novel contributions, verified facts vs. unverified claims, synthetic feature disclosure, and threats to validity. |
| **[13](13_VIVA_QUESTIONS_AND_ANSWERS.md)** | [Viva & Technical Interview Q&A](13_VIVA_QUESTIONS_AND_ANSWERS.md) | 65 comprehensive questions & answers across 4 difficulty tiers, including 5 high-stakes Examiner Trap Questions. |
| **[14](14_PRESENTATION_GUIDE.md)** | [Academic Presentation & Defense Guide](14_PRESENTATION_GUIDE.md) | 14-slide presentation blueprint: visual layouts, bullet points, word-for-word spoken scripts, and in-presentation comebacks. |
| **[15](15_COMPLETE_END_TO_END_FLOW.md)** | [Complete End-to-End Execution Flow](15_COMPLETE_END_TO_END_FLOW.md) | Complete end-to-end trace: data ingestion -> scaling -> training -> calibration -> inference -> Operator Tactical Playbook. |

---

## Suggested Study Tracks

### Track 1: The Quick 1-Hour Viva & Presentation Crash Course
*For preparing on short notice before an evaluation, viva, or presentation:*
1. **[01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md)**: Memorize the 1-minute and 3-minute elevator pitches.
2. **[07_RA_SAFE_COMPLETE_EXPLANATION.md](07_RA_SAFE_COMPLETE_EXPLANATION.md)**: Understand the 3 RA-SAFE modules and the 70-dimensional gate.
3. **[12_RESEARCH_NOVELTY_AND_LIMITATIONS.md](12_RESEARCH_NOVELTY_AND_LIMITATIONS.md)**: Learn what to claim and what to honestly disclose (synthetic physical features).
4. **[13_VIVA_QUESTIONS_AND_ANSWERS.md](13_VIVA_QUESTIONS_AND_ANSWERS.md)**: Review Tier 4 (The 5 Examiner Trap Questions, Q61–Q65).
5. **[14_PRESENTATION_GUIDE.md](14_PRESENTATION_GUIDE.md)**: Practice the slide-by-slide spoken scripts.

---

### Track 2: The Deep 4-Hour Defense & Research Paper Track
*For thesis writing, research defense, or publishing a paper:*
1. **[01](01_PROJECT_OVERVIEW.md)** & **[03](03_DATASET_AND_PREPROCESSING.md)**: Cyber-physical problem formulation, feature breakdown, and dataset provenance.
2. **[04](04_ATTACK_TYPES.md)**: Detailed attack signatures and aerodynamic consequences.
3. **[06](06_CP_ATTN_LITE.md)** & **[07](07_RA_SAFE_COMPLETE_EXPLANATION.md)**: Mathematical comparison between CP-Attn-Lite and RA-SAFE.
4. **[08](08_TRAINING_AND_EVALUATION.md)** & **[09](09_ROBUSTNESS_TESTING.md)**: Multi-task loss formulation, calibration metrics, and the 8 degradation scenarios.
5. **[12](12_RESEARCH_NOVELTY_AND_LIMITATIONS.md)**: Academic threats to validity and future research roadmap.
6. **[13](13_VIVA_QUESTIONS_AND_ANSWERS.md)**: Read all 65 questions and answers thoroughly.

---

### Track 3: The Hands-On Engineering & Demonstration Track
*For live demonstrations, code reviews, and technical walkthroughs:*
1. **[02_REPOSITORY_STRUCTURE.md](02_REPOSITORY_STRUCTURE.md)**: Understand where every file lives and what it executes.
2. **[11_EXECUTION_COMMANDS.md](11_EXECUTION_COMMANDS.md)**: Execute data preprocessing, model training, unit tests, and launch the Streamlit app.
3. **[10_STREAMLIT_DASHBOARD.md](10_STREAMLIT_DASHBOARD.md)**: Master the operator interface, real-time corruption sliders, and live decision badges.
4. **[15_COMPLETE_END_TO_END_FLOW.md](15_COMPLETE_END_TO_END_FLOW.md)**: Step through the end-to-end numerical trace and Operator Action Playbook.

---

## Core System Quick Facts Reference

| Metric / Parameter | Baseline (CP-Attn-Lite) | RA-SAFE (Proposed) |
| :--- | :---: | :---: |
| **Total Parameters** | 3,095 | **4,465** |
| **Model Disk Size** | ~12.2 KB | **~24.3 KB** |
| **Input Feature Count** | 32 (21 Cyber + 11 Physical) | 32 (21 Cyber + 11 Physical) |
| **Latent Embedding Size** | 32 per branch | 32 per branch |
| **Fusion Gate Input Dimensions** | 64 dimensions | **70 dimensions** |
| **Modality Reliability Estimator** | None (Assumes clean data) | **Unsupervised ($Q_c, Q_p \in [0.01, 0.99]$)** |
| **Auxiliary Branch Heads** | None | **2 Heads (Multi-Task Loss)** |
| **Uncalibrated ECE** | 0.0521 | 0.0521 |
| **Calibrated ECE** | Uncalibrated | **0.0202 (Temperature $T=0.8553$)** |
| **Clean Test Accuracy (Raw)** | 84.47% | 78.38% (Regularized) |
| **Accepted Decision Accuracy** | N/A (Forces 100% decisions) | **94.29%** |
| **Selective Risk (Accepted Error)** | 15.53% | **5.71%** |
| **Operational Coverage** | 100.0% | **43.89%** |
| **Accuracy Under RF Jamming** | 41.2% (Blind trust in cyber) | **89.6% (Accepted, shifts to physical)** |
| **Behavior Under Dual Failure** | Forces wrong prediction (38.3%) | **Safe Abstention (89.2% abstained)** |
| **Single-Sample Inference Latency** | 0.058 ms | **0.063 ms (batch) / ~0.45 ms (single)** |
