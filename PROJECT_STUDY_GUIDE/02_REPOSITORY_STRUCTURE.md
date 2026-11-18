# 02. Repository Structure & File-by-File Technical Audit

This document provides a comprehensive audit of every functional file and directory in the project repository.

```
Cyber-Physical-Intrusion-Detection-System-for-Unmanned-Aerial-Vehicles/
├── 1D-CNN -based IDS/          # Baseline 1D-CNN, Multi-Class CNN, and CP-Attn-Lite scripts
├── FNN-based IDS/             # Feedforward Neural Network implementation (5-fold CV)
├── LSTM-based IDS/            # Long Short-Term Memory recurrent IDS (WS=10)
├── SVM-based IDS/             # Support Vector Classifier implementation (Polynomial kernel)
├── dashboard/                 # Streamlit application entry point and modular UI views
├── data/                      # Raw dataset capture and standardized multi-modal CSV splits
├── evaluation/                # Robustness benchmark, degradation stress test, and unit tests
├── models/                    # RA-SAFE neural graph, reliability estimator, uncertainty module
├── results/                   # JSON metric logs, training plots, confusion matrices
├── scripts/                   # Shared preprocessing, label mapping, streaming utilities
├── training/                  # RA-SAFE end-to-end training and calibration scripts
├── utils/                     # Corruption simulator and calibration metrics (ECE, Brier, NLL)
├── PROJECT_STUDY_GUIDE/       # Complete master study guide documentation suite
├── commands                   # PowerShell automation cheat-sheet
├── README.md                  # High-level project summary
├── README_RA_SAFE.md          # Dedicated RA-SAFE technical architecture reference
└── requirements.txt           # Core Python dependencies
```

---

## 1. Top-Level Core Modules

### `README.md`
- **Purpose**: High-level repository orientation explaining the problem, four binary baseline models, multi-class extension, and dashboard execution.
- **Input**: Markdown documentation.
- **Processing**: Rendered by Git repositories and project viewers.
- **Output**: User onboarding text.
- **Used by**: New developers and examiners reviewing the repository.

### `README_RA_SAFE.md`
- **Purpose**: Dedicated research specification for the RA-SAFE architecture, mathematical formulations, empirical benchmark comparisons, and dataset provenance disclosures.
- **Input**: Empirical findings and architecture definitions.
- **Processing**: Detailed technical Markdown with alerts, equations, and tables.
- **Output**: Reference documentation for RA-SAFE.
- **Used by**: Researchers, examiners, and project reviewers.

### `requirements.txt`
- **Purpose**: Specifies top-level Python runtime dependencies.
- **Contents**: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tensorflow`, `streamlit`.
- **Used by**: Environment setup (`pip install -r requirements.txt`).

### `commands`
- **Purpose**: PowerShell batch execution reference providing copy-paste commands for data preparation, model training, evaluation, and dashboard launching.
- **Used by**: Developers running experiments on Windows PowerShell.

---

## 2. Directory: `data/`

### `data/raw/Dataset_T-ITS.csv`
- **Purpose**: Original captured dataset log from the IEEE T-ITS UAV benchmark study (2023).
- **Size & Shape**: 54,783 rows, 38 columns (6.15 MB).
- **Input**: Raw packet capture logs containing network attributes (`frame.len`, `wlan.seq`, `wlan.duration`, `ip.proto`, `tcp.flags`, `time_since_last_packet`, `class`).
- **Processing**: Ingested by `scripts/prepare_dataset.py`.
- **Output**: Feeds dataset standardization pipelines.
- **Used by**: `scripts/prepare_dataset.py`.

### `data/dataset_updated/`
- **Purpose**: Clean, standardized CSV splits for multi-class and binary IDS workflows.
- **Files**:
  - `standardized_dataset_central_combined.csv` (54,783 rows, 36 columns: 32 input features + `attack_class`, `attack_label`, `target`, `target.value`).
  - `standardized_dataset_central_cyber.csv` (54,783 rows).
  - `standardized_dataset_central_physical.csv` (54,783 rows).
  - `standardized_dataset_drone1_*.csv` (First half split: rows 0 to 27,391).
  - `standardized_dataset_drone2_*.csv` (Second half split: rows 27,391 to 54,783).
- **Processing**: Standardized to zero-mean, unit-variance.
- **Used by**: `scripts/paths.py`, `scripts/multiclass_data.py`, `training/train_ra_safe.py`, `evaluation/robustness_tests.py`, and Streamlit views.

### `data/dataset_central/`
- **Purpose**: Legacy 34-column CSV split files without `attack_class` column.
- **Used by**: Older binary scripts fallback when `dataset_updated` is missing.

---

## 3. Directory: `models/`

### `models/ra_safe_model.py`
- **Purpose**: Defines the end-to-end Keras/TensorFlow computational graph for the RA-SAFE architecture.
- **Classes / Functions**:
  - `ReliabilityAdaptiveFusionGate(layers.Layer)`: Implements the 70-dimensional gating network with LeakyReLU and Softmax output.
  - `SqueezeExcite(layers.Layer)`: Channel attention bottleneck on fused representations.
  - `_conv_branch(x, name)`: 1D convolutional feature encoder (`Conv1D(32, 3) -> BN -> GAP1D`).
  - `build_ra_safe_model() -> Model`: Returns the full multi-input multi-output Keras Model.
- **Input**: 4 tensors: `cyber` (21, 1), `physical` (11, 1), `q_cyber` (1,), `q_physical` (1,).
- **Output**: 7 outputs: `probs` (5,), `weights` (2,), `r_pair` (2,), `u_pair` (2,), `cyber_aux` (5,), `phys_aux` (5,), `logits` (5,).
- **Used by**: `training/train_ra_safe.py`, `evaluation/robustness_tests.py`, `dashboard/ra_safe_view.py`.

### `models/reliability_estimator.py`
- **Purpose**: Computes empirical data quality scores $Q_c, Q_p \in [0, 1]$ using unsupervised signal bounds and statistical distances without labels.
- **Classes / Functions**:
  - `ModalityReliabilityEstimator`: Implements `fit()`, `estimate_cyber_reliability()`, `estimate_phys_reliability()`, `save()`, `load()`.
- **Input**: Raw or standardized cyber (21) and physical (11) feature arrays.
- **Output**: Quality scores $Q_c \in [0, 1]$ and $Q_p \in [0, 1]$.
- **Used by**: `training/train_ra_safe.py`, `evaluation/robustness_tests.py`, `dashboard/ra_safe_view.py`.

### `models/uncertainty_estimator.py`
- **Purpose**: Implements normalized predictive entropy calculation and Monte Carlo Dropout for uncertainty quantification.
- **Functions**:
  - `compute_normalized_entropy(probs, num_classes=5)`: Returns Shannon entropy scaled to $[0, 1]$.
  - `compute_mc_dropout_uncertainty(model, inputs, n_passes=10)`: Performs stochastic forward passes to isolate epistemic model uncertainty from aleatoric data noise.
- **Input**: Probability vectors or model inputs.
- **Output**: Dict of total entropy, epistemic mutual information, aleatoric expected entropy, and variance.
- **Used by**: `evaluation/test_ra_safe_unit.py`, `dashboard/ra_safe_view.py`.

### `models/selective_classifier.py`
- **Purpose**: Manages post-hoc temperature scaling, validation-based threshold optimization, and selective prediction decision logic (Accept vs Abstain).
- **Classes / Functions**:
  - `SelectiveClassifier`: Implements `fit_temperature()`, `calibrate_probs()`, `tune_thresholds_on_validation()`, `predict_sample()`, `evaluate_selective_performance()`, `save()`, `load()`.
- **Input**: Raw classification logits, branch reliability scores, and true labels.
- **Output**: Calibrated probabilities, selective prediction status (`ACCEPTED` or `ABSTAINED`), and specific Operator Action Playbook recommendations.
- **Used by**: `training/train_ra_safe.py`, `evaluation/robustness_tests.py`, `dashboard/ra_safe_view.py`.

### `models/checkpoints/`
- **Purpose**: Holds persisted trained weights and configuration artifacts.
- **Files**:
  - `ra_safe_model.keras`: Trained TensorFlow/Keras neural network (4,465 parameters).
  - `reliability_estimator.json`: Fitted mean, standard deviation, and quantile parameters.
  - `selective_classifier.json`: Fitted calibration temperature ($T=0.8553$) and decision thresholds ($\tau_{\text{conf}}=0.65, \tau_{\text{rel}}=0.25$).

---

## 4. Directory: `training/`

### `training/train_ra_safe.py`
- **Purpose**: Primary end-to-end training and evaluation script for RA-SAFE.
- **Input**: Dataset key (default: `central_combined`), epochs, batch size.
- **Processing**:
  1. Stratified 70/15/15 train/val/test split.
  2. Fits `ModalityReliabilityEstimator` strictly on training split (no leakage).
  3. Builds multi-task RA-SAFE model with auxiliary branch losses.
  4. Trains with Adam optimizer and early stopping.
  5. Fits temperature scaling $T$ on validation logits.
  6. Tunes selective thresholds $\tau_{\text{conf}}, \tau_{\text{rel}}$ on validation data.
  7. Evaluates on held-out test data for classification, calibration (ECE, Brier, NLL), and selective metrics.
  8. Generates confusion matrix, training curves, and gate weight plots.
- **Output**: Checkpoints in `models/checkpoints/` and metrics in `results/ra_safe_metrics.json`.
- **Used by**: PowerShell CLI, automated experiment pipelines.

---

## 5. Directory: `evaluation/`

### `evaluation/robustness_tests.py`
- **Purpose**: Comprehensive comparative degradation benchmark testing RA-SAFE against baseline CP-Attn-Lite across 8 operational conditions (nominal, packet loss 10%, packet loss 20%, RF jamming, sensor noise 1.0, sensor noise 2.0, sensor freeze, simultaneous degradation).
- **Input**: Test split from `central_combined`.
- **Processing**: Corrupts test inputs with `utils/corruption_simulator.py`, runs inference through both models, logs accuracy, coverage, accepted set accuracy, and adaptive gate weights.
- **Output**: `results/robustness_metrics.json` and `results/robustness_comparison.png`.
- **Used by**: Researchers, benchmark pipelines, Streamlit robustness view.

### `evaluation/test_ra_safe_unit.py`
- **Purpose**: Automated regression and unit test suite verifying nominal evaluation, degradation responses, low-confidence abstention, MC Dropout decomposition, and operator playbook triggering.
- **Output**: Terminal test report (`[PASS]` indicators, 0 errors).
- **Used by**: Continuous verification and pre-deployment checks.

---

## 6. Directory: `utils/`

### `utils/corruption_simulator.py`
- **Purpose**: Simulates realistic test-time network degradation and physical sensor faults without modifying training distributions.
- **Functions**: `apply_packet_loss()`, `apply_rf_jamming()`, `apply_sensor_noise()`, `apply_sensor_bias()`, `apply_sensor_stuck()`, `simulate_degradation()`.
- **Used by**: `evaluation/robustness_tests.py`, `dashboard/ra_safe_view.py`.

### `utils/calibration_metrics.py`
- **Purpose**: Implements statistical metrics for evaluating probability calibration quality.
- **Functions**:
  - `expected_calibration_error(y_true, y_probs, n_bins=10)`: Computes ECE.
  - `brier_score(y_true, y_probs, num_classes=5)`: Mean squared probabilistic error.
  - `negative_log_likelihood(y_true, y_probs)`: Multi-class cross-entropy.
  - `compute_risk_coverage_curve(y_true, y_probs, reliabilities)`: Evaluates selective risk vs coverage tradeoff.
- **Used by**: `training/train_ra_safe.py`.

---

## 7. Directory: `dashboard/`

### `dashboard/app.py`
- **Purpose**: Main Streamlit application entry point configuring sidebar navigation and page routing.
- **Navigation Options**:
  - `All algorithms`
  - `Multi-class attack ID`
  - `CP-Attn-Lite`
  - `RA-SAFE Detection` (New)
  - `RA-SAFE Robustness` (New)
  - `Live streaming demo`
  - `Explainable alerts`
  - `Compare fusions`
  - `Model details`
  - `All plots`
  - `How to run`
- **Used by**: `streamlit run dashboard/app.py`.

### `dashboard/ra_safe_view.py`
- **Purpose**: Interactive diagnostic console for RA-SAFE.
- **Capabilities**:
  - Real-time row selector across 54,783 samples.
  - Live injection of packet loss, RF jamming, IMU noise, or freeze faults.
  - Displays predicted attack class, calibrated confidence, cyber reliability ($R_c$), physical reliability ($R_p$), adaptive weights ($w_c, w_p$), acceptance/abstention badge, and operator action playbook.
  - Displays confusion matrix and training curves.
- **Used by**: `dashboard/app.py`.

### `dashboard/robustness_view.py`
- **Purpose**: Displays the comparative robustness benchmark table and visual chart comparing RA-SAFE against baseline CP-Attn-Lite across all 8 degradation modes.
- **Used by**: `dashboard/app.py`.

### `dashboard/cp_attn_view.py`
- **Purpose**: Displays the CP-Attn-Lite dual-branch baseline metrics, confusion matrix, and average gate weights by true class.
- **Used by**: `dashboard/app.py`.

### `dashboard/live_streaming.py`
- **Purpose**: Replays CSV rows sequentially to simulate a real-time telemetry stream and measure per-sample inference latency using saved 1D-CNN models.
- **Used by**: `dashboard/app.py`.

### `dashboard/alert_panel.py` & `dashboard/explainable_alerts.py`
- **Purpose**: Renders operator alert cards mapping predicted attack IDs to plain-language severity, technical indicators, and recommended countermeasures.
- **Used by**: `dashboard/live_streaming.py`, `dashboard/app.py`.

### `dashboard/results_loader.py`
- **Purpose**: Data access layer loading JSON metrics and plot paths for all models.
- **Used by**: All dashboard views.

---

## 8. Directory: `scripts/`

### `scripts/prepare_dataset.py`
- **Purpose**: Processes raw `Dataset_T-ITS.csv` into standardized CSVs across cyber, physical, and combined splits.
- **Functions**: `_build_full_frame()`, `_standardize()`, `_subset_for_fusion()`, `write_datasets()`.
- **Note**: Synthesizes the 11 physical features and missing radio fields when not found in raw CSV.
- **Used by**: Dataset setup pipeline.

### `scripts/attack_labels.py`
- **Purpose**: Canonical definitions and mapping utilities for the 5 attack classes (`CLASS_BENIGN=0`, `CLASS_DOS=1`, `CLASS_REPLAY=2`, `CLASS_EVIL_TWIN=3`, `CLASS_FDI=4`).
- **Functions**: `normalize_class_string()`, `label_from_row_index()`, `resolve_attack_class()`.
- **Used by**: All training scripts, evaluation modules, and dashboard pages.

### `scripts/attack_info.py`
- **Purpose**: Stores operator-facing incident response metadata (layer, severity, description, indicators, recommended actions) for each attack class.
- **Used by**: `dashboard/alert_panel.py`, `dashboard/explainable_alerts.py`.

### `scripts/multiclass_data.py`
- **Purpose**: Helper functions to load multi-class CSV frames, slice column subsets (`KEEP_COMBINED`, `KEEP_CYBER`, `KEEP_PHYSICAL`), and reshape arrays for 1D-CNN inference.
- **Used by**: `1D-CNN -based IDS/multiclass_cnn.py`, `1D-CNN -based IDS/cp_attn_cnn.py`, `training/train_ra_safe.py`.

### `scripts/paths.py`
- **Purpose**: Centralized path resolver locating `dataset_updated` and standardized CSVs.
- **Used by**: All scripts requiring dataset file access.

### `scripts/cp_attn_model.py`
- **Purpose**: Defines baseline CP-Attn-Lite model architecture (`build_cp_attn_model()`, `FusionGate`, `SqueezeExcite`, `split_modalities()`).
- **Used by**: `1D-CNN -based IDS/cp_attn_cnn.py`, `evaluation/robustness_tests.py`.

### `scripts/save_metrics.py` & `scripts/save_multiclass_metrics.py`
- **Purpose**: Utilities to append/update run records in `results/metrics.json` and `results/metrics_multiclass.json`.
- **Used by**: Baseline training scripts.

### `scripts/run_all_experiments.py` & `scripts/run_multiclass_experiments.py`
- **Purpose**: Batch orchestrators running SVM, FNN, LSTM, and 1D-CNN training sequentially across splits.
- **Used by**: Automated benchmarking runs.

---

## 9. Baseline Folders

### `1D-CNN -based IDS/`
- `1D_CNN.py`: Trains binary 1D-CNN (Conv1D 64 -> Conv1D 128 -> Dropout 0.5 -> Dense 2).
- `multiclass_cnn.py`: Trains monolithic 5-class 1D-CNN on concatenated features.
- `cp_attn_cnn.py`: Trains baseline dual-branch CP-Attn-Lite model (3,095 parameters).
- `cp_attn_models/cp_attn_central_combined.keras`: Saved baseline model checkpoint.
- `cp_attn_results/confusion_central_combined.png`: Saved confusion matrix plot.

### `SVM-based IDS/`
- `svm.py`: Trains Support Vector Classifier (`SVC(kernel='poly', C=5, gamma=0.1)`) with 5-fold cross-validation.
- `SVM_GRID.py`: Hyperparameter grid search utility for C, gamma, and kernel selection.
- `SVM_SHAP.py`: SHAP kernel explainer attempt for feature attribution.

### `FNN-based IDS/`
- `FNN.py`: Trains a 5-layer Feedforward Neural Network (256 units per hidden layer, ReLU activation, Sigmoid output) using 5-fold cross-validation.

### `LSTM-based IDS/`
- `LSTM.py`: Windows input features into sequences of length $WS=10$ and trains a 5-layer recurrent LSTM network (512 units, He uniform initialization, L1/L2 regularization, Dropout 0.7) using cross-validation.

---

## 10. Directory: `results/`

- `ra_safe_metrics.json`: Full evaluation results for RA-SAFE on the held-out test split (accuracy, F1, calibration error, selective risk, coverage).
- `robustness_metrics.json`: Quantitative stress-test results across 8 degradation scenarios comparing CP-Attn-Lite and RA-SAFE.
- `confusion_ra_safe.png`: Held-out test set confusion matrix for RA-SAFE.
- `history_ra_safe.png`: Epoch convergence curves for RA-SAFE accuracy and loss.
- `gate_weights_ra_safe.png`: Average cyber vs physical gate weights grouped by true attack class.
- `robustness_comparison.png`: Visual bar/line chart of accuracy and coverage under severe degradations.
- `metrics_multiclass.json`: Multi-class baseline benchmark records.
- `metrics.json`: Binary baseline benchmark records.
