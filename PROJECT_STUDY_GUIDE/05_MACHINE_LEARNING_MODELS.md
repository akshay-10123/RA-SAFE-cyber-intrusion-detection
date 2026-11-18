# 05. Machine Learning & Deep Learning Model Architectures

This document inspects and details every baseline and advanced model architecture implemented in the project.

---

## Model Inventory & High-Level Comparison

| Model | Script File | Task | Input Shape | Param Count | Optimizer / Kernel | Loss Function |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM** | `SVM-based IDS/svm.py` | Binary | $(N, D)$ | Non-Parametric | Poly Kernel ($C=5, \gamma=0.1$) | Hinge / Margin Loss |
| **FNN** | `FNN-based IDS/FNN.py` | Binary | $(N, D)$ | $\approx 200\text{k}$ | Adam ($\text{lr}=10^{-3}$) | Binary Cross-Entropy |
| **LSTM** | `LSTM-based IDS/LSTM.py` | Binary | $(N, 10, D)$ | $\approx 3.2\text{M}$ | Adam ($\text{lr}=10^{-4}$) | Binary Cross-Entropy |
| **1D-CNN** | `1D-CNN -based IDS/1D_CNN.py` | Binary | $(N, D, 1)$ | $\approx 150\text{k}$ | Adam ($\text{lr}=10^{-3}$) | Categorical Cross-Entropy |
| **Multi-class 1D-CNN** | `1D-CNN -based IDS/multiclass_cnn.py` | 5-Class | $(N, D, 1)$ | $\approx 151\text{k}$ | Adam ($\text{lr}=10^{-3}$) | Categorical Cross-Entropy |
| **CP-Attn-Lite** | `scripts/cp_attn_model.py` | 5-Class | $(N, 21, 1) + (N, 11, 1)$ | **3,095** | Adam ($\text{lr}=10^{-3}$) | Categorical Cross-Entropy |
| **RA-SAFE** | `models/ra_safe_model.py` | 5-Class | $(N, 21, 1) + (N, 11, 1) + 2 \times (N, 1)$ | **4,465** | Adam ($\text{lr}=10^{-3}$) | Multi-Task Categorical CE + MSE |

*Where $D \in \{11, 21, 32\}$ depending on the fusion split (`cyber`, `physical`, or `combined`).*

---

## 1. Support Vector Machine (SVM)

### Overview & Motivation
SVM was chosen as a strong classical machine learning baseline. It seeks the optimal separating hyperplane that maximizes the geometric margin between normal traffic and attacks.

### Architecture & Hyperparameters
- **Implementation**: Scikit-Learn `SVC(kernel='poly', C=5, gamma=0.1)`.
- **Inputs**: Flat vector of standardized features ($D=11, 21, 32$).
- **Kernel Function**: Polynomial kernel:
  $$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$$
- **Training Process**: 5-fold cross-validation evaluated across training sizes using `learning_curve()`.
- **Strengths**: High generalization capability in high-dimensional spaces; effective margin boundaries.
- **Weaknesses**: Quadratic to cubic scaling with sample count $\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$; slow inference for real-time edge flight; produces uncalibrated decision values rather than probabilistic uncertainty.

### Saved Results (`results/metrics.json`)
- `drone1_cyber`: Accuracy = **88.09%**, Precision = **92.20%**, Recall = **89.39%**, F1 = **90.75%**, AUC = **0.929**
- `drone1_physical`: Accuracy = **98.67%**, Precision = **99.19%**, Recall = **98.79%**, F1 = **98.99%**, AUC = **0.998**
- `drone1_combined`: Accuracy = **98.93%**, Precision = **98.61%**, Recall = **99.80%**, F1 = **99.20%**, AUC = **0.999**

---

## 2. Feedforward Neural Network (FNN)

### Architecture Layer by Layer
```python
# From FNN-based IDS/FNN.py (create_model with hiddenlayers=5):
model = Sequential()
model.add(Dense(256, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```
1. **Input Layer**: Flat vector of shape $(B, D)$ where $D \in \{11, 21, 32\}$.
2. **Dense Layers 1–5**: 256 hidden units each, equipped with Rectified Linear Unit (ReLU) activations.
3. **Output Layer**: 1 unit with Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$.
4. **Loss & Optimizer**: `binary_crossentropy`, Adam optimizer ($\text{lr}=10^{-3}$), EarlyStopping patience=5.
5. **Batch Size & Epochs**: Batch size = 256, Max epochs = 30 to 100.
6. **Trainable Parameters**: $\approx 200,000$ parameters.

### Saved Results (`results/metrics.json`)
- `drone1_cyber`: Accuracy = **92.36%**, Precision = **98.19%**, Recall = **90.01%**, F1 = **93.92%**, AUC = **0.961**
- `drone1_physical`: Accuracy = **99.62%**, Precision = **99.77%**, Recall = **99.65%**, F1 = **99.71%**, AUC = **0.999**
- `drone1_combined`: Accuracy = **99.90%**, Precision = **99.92%**, Recall = **99.92%**, F1 = **99.92%**, AUC = **0.999**

---

## 3. Long Short-Term Memory (LSTM)

### Overview & Temporal Windowing
The LSTM model captures temporal dependencies across sequential telemetry. Instead of classifying a single instantaneous row, it slides a temporal window across $WS=10$ successive packets.

```python
# Temporal Window Construction in LSTM-based IDS/LSTM.py:
values = df.values.astype(np.float32)
n = len(values) - WS
xs = np.zeros((n, WS, feats), dtype=np.float32)
for i in range(n):
    xs[i] = values[i : i + WS]
ys = labels[WS:]
```

### Architecture Layer by Layer
```python
# From LSTM-based IDS/LSTM.py:
model = Sequential()
model.add(LSTM(512, input_shape=(10, feats), return_sequences=True, kernel_regularizer=L2(0.001)))
model.add(LSTM(512, kernel_initializer='he_uniform', activation='relu', return_sequences=True))
model.add(LSTM(512, kernel_initializer='he_uniform', activation='relu', return_sequences=True))
model.add(LSTM(512, kernel_initializer='he_uniform', activation='relu', return_sequences=True))
model.add(LSTM(512, kernel_initializer='he_uniform', activation='relu', return_sequences=False))
model.add(Dropout(0.7))
model.add(Dense(1, activation='sigmoid'))
```
- **Input Shape**: $(B, 10, D)$ where $WS=10$.
- **Recurrent Depth**: 5 stacked LSTM layers with 512 hidden memory cells per layer.
- **Regularization**: $L_2$ kernel regularization ($0.001$) and high Dropout ($0.7$).
- **Parameters**: $\approx 3,200,000$ parameters (heavyweight).
- **Strengths**: Models temporal progression, packet burst trends, and inertial trajectory drift.
- **Weaknesses**: Extremely compute-intensive for embedded flight microcontrollers; high latency; prone to memory bottlenecks.

---

## 4. Binary 1D Convolutional Neural Network (1D-CNN)

### Architecture Layer by Layer
```python
# From 1D-CNN -based IDS/1D_CNN.py:
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(D, 1)))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
```
1. **Input**: Reshaped as a 1D spatial sequence: $(B, D, 1)$.
2. **Conv1D Layer 1**: 64 filters, kernel size $k=3$, ReLU. Captures local correlations among adjacent features.
3. **Conv1D Layer 2**: 128 filters, kernel size $k=3$, ReLU.
4. **Dropout (0.5)**: Prevents co-adaptation of feature detectors.
5. **Flatten & Dense (2)**: 2-class softmax output ($0=\text{benign}, 1=\text{attack}$).
6. **Parameters**: $\approx 150,000$.

### Saved Results (`results/metrics.json`)
- `drone1_cyber`: Accuracy = **92.85%**, Precision = **97.10%**, Recall = **91.79%**, F1 = **94.37%**, AUC = **0.933**
- `drone1_physical`: Accuracy = **99.78%**, Precision = **99.86%**, Recall = **99.80%**, F1 = **99.83%**, AUC = **0.998**
- `drone1_combined`: Accuracy = **99.98%**, Precision = **100.0%**, Recall = **99.97%**, F1 = **99.99%**, AUC = **0.999**

---

## 5. Multi-Class 1D-CNN

### Overview
Extends the binary 1D-CNN to 5 discrete classes (`benign`, `dos`, `replay`, `evil_twin`, `fdi`).

```python
# From 1D-CNN -based IDS/multiclass_cnn.py:
model = Sequential([
    Conv1D(64, 3, activation="relu", input_shape=(D, 1)),
    Conv1D(128, 3, activation="relu"),
    Dropout(0.5),
    Flatten(),
    Dense(5, activation="softmax")
])
```
- **Input**: Concatenated 32 features shaped as $(B, 32, 1)$.
- **Output**: 5-class probability vector.
- **Saved Results (`results/metrics_multiclass.json`) on `central_combined`**:
  - Accuracy: **95.13%**
  - Macro F1: **95.40%**
  - Weighted F1: **95.06%**
  - Per-class F1: Benign = 93.4%, DoS = 89.9%, Replay = 97.3%, Evil Twin = 99.8%, FDI = 96.6%
- **Limitation**: Monolithic architecture. Convolves across cyber and physical boundaries indiscriminately; provides no modality attribution or reliability awareness.

---

## 6. End-to-End Deep Learning Lifecycle

```mermaid
flowchart LR
    A["Raw Data\n(Dataset_T-ITS.csv)"] --> B["Preprocessing\n(prepare_dataset.py)"]
    B --> C["Stratified Split\n(Train / Val / Test)"]
    C --> D["StandardScaler\n(Fit on Train)"]
    D --> E["Model Architecture\n(RA-SAFE Graph)"]
    E --> F["Backpropagation\n(Adam, Multi-task Loss)"]
    F --> G["Validation Calibration\n(Temperature Scaling T)"]
    G --> H["Selective Thresholding\n(tau_conf, tau_rel)"]
    H --> I["Saved Checkpoint\n(ra_safe_model.keras)"]
    I --> J["Inference Engine\n(Accept vs Abstain)"]
    J --> K["Operator Action\nPlaybook Alert"]
```
