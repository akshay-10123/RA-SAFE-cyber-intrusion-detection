# 11. Complete Project Execution & Reproduction Guide

This document provides exact, verified PowerShell commands for the Windows environment (`C:\uav-venv\Scripts\python.exe`).

---

## 1. Environment Verification & Directory Navigation

Open a PowerShell terminal and navigate to the project root:

```powershell
$py = "C:\uav-venv\Scripts\python.exe"
$root = "C:\Users\HP\OneDrive\Documents\Cyber-Physical-Intrusion-Detection-System-for-Unmanned-Aerial-Vehicles"
cd $root

# Verify Python version and environment
& $py -V
& $py -c "import tensorflow as tf; import streamlit as st; print('TensorFlow:', tf.__version__, 'Streamlit:', st.__version__)"
```

---

## 2. Dataset Preparation

If the standardized CSV files under `data/dataset_updated/` need to be regenerated from raw logs:

```powershell
& $py scripts\prepare_dataset.py --output data\dataset_updated
```

---

## 3. Training & Evaluating RA-SAFE (Proposed Architecture)

### A. End-to-End Training & Calibration
Trains the multi-task RA-SAFE model on `central_combined` (70/15/15 stratified split), fits temperature scaling $T$, optimizes selective thresholds, evaluates held-out test data, and saves checkpoints:

```powershell
& $py training\train_ra_safe.py --epochs 12 --batch_size 64
```
- **Generated Model**: `models/checkpoints/ra_safe_model.keras`
- **Fitted Reliability Parameters**: `models/checkpoints/reliability_estimator.json`
- **Calibration Parameters**: `models/checkpoints/selective_classifier.json`
- **Output Metrics**: `results/ra_safe_metrics.json`
- **Plots**: `results/confusion_ra_safe.png`, `results/history_ra_safe.png`, `results/gate_weights_ra_safe.png`

### B. Run Robustness & Degradation Stress Benchmarks
Evaluates RA-SAFE vs baseline CP-Attn-Lite across 8 adverse operational conditions (packet loss, RF jamming, sensor noise, sensor freeze, simultaneous):

```powershell
& $py evaluation\robustness_tests.py
```
- **Output Metrics**: `results/robustness_metrics.json`
- **Comparison Chart**: `results/robustness_comparison.png`

### C. Run Unit & Integration Test Suite
Executes automated assertions covering nominal evaluation, degradation responses, low-confidence abstention, and MC Dropout decomposition:

```powershell
& $py evaluation\test_ra_safe_unit.py
```

---

## 4. Running Baseline Models

### A. Baseline CP-Attn-Lite (Dual-Branch Attention Baseline)
```powershell
& $py "1D-CNN -based IDS\cp_attn_cnn.py" --fusion central_combined --epochs 12
```

### B. Multi-Class 1D-CNN Baseline (Monolithic 5-Class)
```powershell
& $py "1D-CNN -based IDS\multiclass_cnn.py" --fusion central_combined --epochs 15
```

### C. Run All Binary Baseline Models (SVM, FNN, LSTM, 1D-CNN)
```powershell
# Full benchmark run across drone1 splits
& $py scripts\run_all_experiments.py

# Quick smoke test (fewer epochs/samples)
& $py scripts\run_all_experiments.py --quick
```

### D. Individual Binary Baseline Executions
```powershell
# 1D-CNN Binary
cd "$root\1D-CNN -based IDS"
& $py 1D_CNN.py --fusion=drone1_combined --metrics_title=drone1_combined --epochs=30
cd $root

# Support Vector Machine (Polynomial Kernel)
cd "$root\SVM-based IDS"
& $py svm.py --fusion=drone1_combined --png=drone1_combined.png --metrics_title=drone1_combined --max_samples=8000
cd $root

# Feedforward Neural Network (5 Layers, 5-Fold CV)
cd "$root\FNN-based IDS"
& $py FNN.py --fusion=drone1_combined --title_name=drone1_combined --epochs=30
cd $root

# Long Short-Term Memory Network (WS=10, 5 Layers)
cd "$root\LSTM-based IDS"
& $py LSTM.py --fusion=drone1_combined --metrics_title=drone1_combined --epochs=10 --max_samples=5000 --cv_folds=2
cd $root
```

---

## 5. Launching the Streamlit Interactive Dashboard

```powershell
cd $root
& $py -m streamlit run dashboard/app.py
```
- Open browser at **`http://localhost:8501`**
- Navigate to **"RA-SAFE Detection"** for live single-row inference and interactive corruption simulation.
- Navigate to **"RA-SAFE Robustness"** for full benchmark tables and degradation comparison charts.

---

## 6. Checkpoint & Metric File Locations

| Artifact | File System Location | Description |
| :--- | :--- | :--- |
| **RA-SAFE Model** | `models/checkpoints/ra_safe_model.keras` | Serialized Keras 3 neural network graph |
| **Reliability Estimator** | `models/checkpoints/reliability_estimator.json` | Training mean, std, and quantile bounds |
| **Calibration Parameters** | `models/checkpoints/selective_classifier.json` | Learned $T=0.8553$, $\tau_{\text{conf}}=0.65$, $\tau_{\text{rel}}=0.25$ |
| **RA-SAFE Test Metrics** | `results/ra_safe_metrics.json` | Accuracy, F1, ECE, Brier, selective risk, coverage |
| **Robustness Benchmark** | `results/robustness_metrics.json` | 8-scenario stress-test comparison data |
| **Confusion Matrix Plot** | `results/confusion_ra_safe.png` | Held-out test set confusion matrix |
| **Robustness Chart** | `results/robustness_comparison.png` | Visual accuracy & coverage comparison plot |
| **Multi-class Metrics** | `results/metrics_multiclass.json` | Baseline 1D-CNN and CP-Attn-Lite benchmark logs |
| **Binary Metrics** | `results/metrics.json` | SVM, FNN, LSTM, and 1D-CNN baseline benchmark logs |
