# 08. Training Pipeline, Metric Definitions & Empirical Evaluation

This document details the training configuration, loss formulations, metric equations, and verified test results for RA-SAFE.

---

## 1. Experimental Setup & Splitting Protocol

### Dataset Partitioning (70 / 15 / 15 Stratified Split)
To ensure reliable evaluation without data leakage, the complete `central_combined` dataset (54,783 rows) was partitioned as follows:

| Split | Sample Count | Percentage | Class Stratification | Purpose |
| :---: | :---: | :---: | :---: | :--- |
| **Train** | 38,348 | 70.0% | Preserved across all 5 classes | Model parameter optimization via gradient descent |
| **Validation** | 8,217 | 15.0% | Preserved across all 5 classes | Early stopping, temperature scaling $T$, threshold tuning |
| **Test** | 8,218 | 15.0% | Preserved across all 5 classes | Final unbiased held-out benchmark evaluation |

### Preprocessing & Leakage Safeguards
- The `ModalityReliabilityEstimator` was fitted **exclusively on the 38,348 training samples** (`models/reliability_estimator.py`).
- Feature means $\mu_{\text{train}}$ and variances $\sigma_{\text{train}}^2$ were stored in `models/checkpoints/reliability_estimator.json`.
- Validation and test sets were transformed strictly using training statistics.
- Attack labels were never accessed during reliability estimation.

---

## 2. Multi-Task Loss Formulation & Optimization

The RA-SAFE computational graph is optimized using a composite multi-task objective:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{probs}} + \lambda_2 \mathcal{L}_{\text{cyber\_aux}} + \lambda_3 \mathcal{L}_{\text{phys\_aux}} + \sum_{j \in \{\text{weights}, r, u, \text{logits}\}} 0.0 \cdot \mathcal{L}_j$$

Where:
- $\mathcal{L}_{\text{probs}} = -\sum_{k=1}^5 y_k \ln(p_k)$ (Categorical cross-entropy on fused classification head, weight $\lambda_1 = 1.0$).
- $\mathcal{L}_{\text{cyber\_aux}} = -\sum_{k=1}^5 y_k \ln(\hat{y}_{c, k})$ (Auxiliary loss on cyber branch, weight $\lambda_2 = 0.20$).
- $\mathcal{L}_{\text{phys\_aux}} = -\sum_{k=1}^5 y_k \ln(\hat{y}_{p, k})$ (Auxiliary loss on physical branch, weight $\lambda_3 = 0.20$).
- Optimizer: **Adam** with initial learning rate $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$.
- Batch size: **64 samples**.
- Epochs: **12 epochs** (Training completed in **80.4 seconds** on CPU).
- Callbacks: `EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)` and `ModelCheckpoint`.

### Why Are Auxiliary Losses Essential?
Without $\mathcal{L}_{\text{cyber\_aux}}$ and $\mathcal{L}_{\text{phys\_aux}}$, the auxiliary branch heads would remain random linear projections. By assigning an active loss weight ($\lambda_2 = \lambda_3 = 0.20$), backpropagation forces the individual branch representations $z_c$ and $z_p$ to predict attack classes independently. Consequently, their Shannon entropy $U_c$ and $U_p$ accurately reflects modality-specific ambiguity.

---

## 3. Mathematical Metric Definitions

### 1. Accuracy
Overall proportion of correctly classified samples:
$$\text{Accuracy} = \frac{\sum_{k=1}^5 \text{TP}_k}{N_{\text{total}}}$$

### 2. Macro Precision
Unweighted average of per-class precision (penalizes false alarms equally across rare and frequent classes):
$$\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}, \quad \text{Precision}_{\text{macro}} = \frac{1}{5}\sum_{k=1}^5 \text{Precision}_k$$

### 3. Macro Recall
Unweighted average of per-class recall (measures attack detection completeness):
$$\text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}, \quad \text{Recall}_{\text{macro}} = \frac{1}{5}\sum_{k=1}^5 \text{Recall}_k$$

### 4. Macro F1-Score
Harmonic mean of macro precision and macro recall:
$$\text{F1}_k = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}, \quad \text{F1}_{\text{macro}} = \frac{1}{5}\sum_{k=1}^5 \text{F1}_k$$

### 5. Expected Calibration Error (ECE)
Partitions predictions into $M=10$ equally spaced confidence bins $B_1, \dots, B_M$. Computes the weighted absolute difference between bin accuracy and bin confidence:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$
*(A perfectly calibrated model has $\text{ECE} = 0.0$)*.

### 6. Brier Score
Measures mean squared error of predicted probability vectors against one-hot true target vectors $\mathbf{y}$:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^5 \left(p_{i, k} - y_{i, k}\right)^2 \in [0, 2]$$
*(Lower is better; 0.0 indicates perfect probabilistic certainty)*.

### 7. Negative Log-Likelihood (NLL)
Multi-class cross-entropy loss evaluated across test samples:
$$\text{NLL} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^5 y_{i, k} \ln(p_{i, k})$$

### 8. Selective Coverage & Selective Risk
- **Coverage**: Proportion of test instances where the model's confidence and reliability satisfy the acceptance threshold:
  $$\text{Coverage} = \frac{N_{\text{accepted}}}{N_{\text{total}}}$$
- **Selective Risk**: Error rate evaluated **strictly over accepted instances**:
  $$\text{Selective Risk} = \frac{\sum_{i \in \text{Accepted}} \mathbb{I}(\hat{y}_i \neq y_i)}{N_{\text{accepted}}}$$
- **Accepted Set Accuracy**: $1.0 - \text{Selective Risk}$.

---

## 4. Empirical Test Results: Verified Benchmark Data

From held-out test evaluation (`results/ra_safe_metrics.json`):

### Overall Classification & Calibration Metrics
- **Total Test Samples**: 8,218
- **Unfiltered Test Accuracy**: **78.38%**
- **Macro Precision**: **80.52%**
- **Macro Recall**: **79.94%**
- **Macro F1**: **79.28%**
- **Weighted F1**: **77.68%**
- **ECE Before Calibration**: **0.0298**
- **ECE After Temperature Scaling**: **0.0202** (32.2% calibration improvement)
- **Brier Score**: **0.2841**
- **Negative Log-Likelihood**: **0.4509**
- **Inference Latency**: **0.063 ms / sample** (over 15,800 predictions/sec on CPU)
- **Model Parameters**: **4,465 parameters**

### Selective Prediction Performance
- **Selective Coverage**: **43.89%** (3,607 samples accepted)
- **Accepted Set Accuracy**: **94.29%**
- **Selective Risk**: **5.71%** (Only 206 classification errors in accepted set)
- **Abstention Rate**: **56.11%** (4,611 samples safely routed to human review)

### Per-Class Test Breakdown (Unfiltered)

| Class ID | Class Label | Precision | Recall | F1-Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | `benign` | 86.8% | 98.2% | **92.2%** | 2,058 |
| **1** | `dos` | 61.2% | 63.8% | **62.5%** | 1,897 |
| **2** | `replay` | 65.4% | 53.9% | **59.1%** | 1,947 |
| **3** | `evil_twin` | 98.4% | 97.2% | **97.8%** | 1,674 |
| **4** | `fdi` | 88.2% | 91.5% | **89.8%** | 642 |

### Held-Out Test Confusion Matrix

```
                      PREDICTED CLASS
             Benign    DoS   Replay  Evil Twin   FDI
True Benign  [ 2021     18      3        5       11 ]
True DoS     [  148   1210    531        1        7 ]
True Replay  [  135    748   1049        1       14 ]
True EvilTwin[    0      0      1     1627       46 ]
True FDI     [   24      2     19       10      587 ]
```

---

## 5. Scientific Explanation: Raw Accuracy vs. Accepted Accuracy

### The Apparent Paradox
An examiner might ask:
> *"Why is RA-SAFE's unfiltered test accuracy (78.38%) slightly lower than CP-Attn-Lite (79.36%), but you claim RA-SAFE is superior?"*

### The Scientific Defense:
1. **Unfiltered Accuracy Ignores Risk**: Standard models achieve 79.36% accuracy only by forcing a guess on every sample. In 20.64% of cases, the model is wrong, potentially issuing false alarms or missing genuine attacks.
2. **Selective Risk Reduction**: In aviation and industrial defense, **wrong actions carry severe consequences**. RA-SAFE identifies samples where feature evidence is ambiguous (e.g., DoS vs Replay boundary where both show Wi-Fi disruption).
3. **Accepted Accuracy Reaches 94.29%**: By abstaining from ambiguous samples, the predictions that RA-SAFE *does* accept have a **94.29% accuracy** and only **5.71% error rate**.
4. **Safety via Abstention**: The 56.11% of samples that are abstained are not silently ignored; they trigger the **Operator Action Playbook**, advising the human supervisor to inspect telemetry or switch to autonomous Return-to-Home.
