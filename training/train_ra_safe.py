"""
End-to-End Training & Calibration Pipeline for RA-SAFE.

Features:
  - Clean 70/15/15 stratified train/val/test split.
  - Modality reliability statistics fitted exclusively on training data (zero leakage).
  - Multi-task training with auxiliary branch heads for uncertainty quantification.
  - Validation-set post-hoc temperature scaling calibration.
  - Validation-set selective prediction threshold optimization.
  - Rigorous evaluation on held-out test data reporting classification, calibration,
    and selective prediction metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.ra_safe_model import build_ra_safe_model
from models.reliability_estimator import ModalityReliabilityEstimator
from models.selective_classifier import SelectiveClassifier
from scripts.attack_labels import ATTACK_CLASS_NAMES, ATTACK_DISPLAY_NAMES, NUM_ATTACK_CLASSES
from scripts.multiclass_data import frame_to_xy, load_fusion_frame
from utils.calibration_metrics import brier_score, expected_calibration_error, negative_log_likelihood

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
CLASS_ORDER = [ATTACK_CLASS_NAMES[i] for i in range(NUM_ATTACK_CLASSES)]
DISPLAY_ORDER = [ATTACK_DISPLAY_NAMES[i] for i in range(NUM_ATTACK_CLASSES)]


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ticks = np.arange(len(DISPLAY_ORDER))
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=DISPLAY_ORDER,
        yticklabels=DISPLAY_ORDER,
        ylabel="True class",
        xlabel="Predicted class",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_training_history(history, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    acc_key = next((k for k in history.history if "acc" in k and not k.startswith("val_")), None)
    val_acc_key = next((k for k in history.history if "acc" in k and k.startswith("val_")), None)
    if acc_key and val_acc_key:
        ax1.plot(history.history[acc_key], label="Train Accuracy", color="#1f77b4", lw=2)
        ax1.plot(history.history[val_acc_key], label="Val Accuracy", color="#ff7f0e", lw=2)
    ax1.set_title("RA-SAFE Classification Accuracy", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    train_loss = history.history.get("probs_loss", history.history.get("loss"))
    val_loss = history.history.get("val_probs_loss", history.history.get("val_loss"))
    if train_loss is not None:
        ax2.plot(train_loss, label="Train Loss", color="#1f77b4", lw=2)
    if val_loss is not None:
        ax2.plot(val_loss, label="Val Loss", color="#ff7f0e", lw=2)
    ax2.set_title("RA-SAFE Loss Convergence", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_gate_weights(weights: np.ndarray, y_true: np.ndarray, path: Path) -> None:
    means_c = []
    means_p = []
    names = []
    for cid, name in enumerate(CLASS_ORDER):
        mask = y_true == cid
        if np.any(mask):
            means_c.append(float(np.mean(weights[mask, 0])))
            means_p.append(float(np.mean(weights[mask, 1])))
            names.append(ATTACK_DISPLAY_NAMES[cid])

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, means_c, width, label="Cyber Weight (w_c)", color="#2ca02c")
    ax.bar(x + width / 2, means_p, width, label="Physical Weight (w_p)", color="#d62728")
    ax.set_ylabel("Adaptive Fusion Weight")
    ax.set_title("RA-SAFE Adaptive Fusion Weights Across Attack Types")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate RA-SAFE")
    parser.add_argument("--fusion", default="central_combined")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RA-SAFE: RELIABILITY-AWARE SELECTIVE FUSION FOR UAV IDS")
    print("=" * 70)
    print("CRITICAL DATASET NOTICE:")
    print("  In the public T-ITS dataset layout, the 11 physical telemetry features")
    print("  were synthesized via Gaussian offsets conditioned on the attack indicator.")
    print("  RA-SAFE uses these features strictly as a multi-modal testbed for the")
    print("  reliability-aware adaptive fusion architecture.")
    print("=" * 70)

    # 1. Load Data
    print(f"Loading dataset: {args.fusion}")
    df = load_fusion_frame(args.fusion)
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows.")

    x_all, y_all = frame_to_xy(df)
    n_samples = len(x_all)
    print(f"Total samples: {n_samples:,} across {NUM_ATTACK_CLASSES} attack classes.")

    # 2. Strict 70/15/15 Stratified Split
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_all, y_all, test_size=0.30, random_state=42, stratify=y_all
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Data Splits: Train={len(x_train):,}, Val={len(x_val):,}, Test={len(x_test):,}")

    # 3. Fit Modality Reliability Estimator ONLY on Training Split
    print("\nFitting Modality Reliability Estimator on clean training set...")
    rel_estimator = ModalityReliabilityEstimator()
    rel_estimator.fit(x_train[:, :21], x_train[:, 21:])
    rel_path = CHECKPOINT_DIR / "reliability_estimator.json"
    rel_estimator.save(rel_path)
    print(f"Saved reliability estimator parameters: {rel_path}")

    # Compute Q_c and Q_p
    qc_train, qp_train = rel_estimator.estimate(x_train[:, :21], x_train[:, 21:])
    qc_val, qp_val = rel_estimator.estimate(x_val[:, :21], x_val[:, 21:])
    qc_test, qp_test = rel_estimator.estimate(x_test[:, :21], x_test[:, 21:])

    # Reshape features for Conv1D: (N, L, 1)
    xc_train = x_train[:, :21].reshape(-1, 21, 1)
    xp_train = x_train[:, 21:].reshape(-1, 11, 1)
    xc_val = x_val[:, :21].reshape(-1, 21, 1)
    xp_val = x_val[:, 21:].reshape(-1, 11, 1)
    xc_test = x_test[:, :21].reshape(-1, 21, 1)
    xp_test = x_test[:, 21:].reshape(-1, 11, 1)

    y_train_oh = to_categorical(y_train, num_classes=NUM_ATTACK_CLASSES)
    y_val_oh = to_categorical(y_val, num_classes=NUM_ATTACK_CLASSES)

    dummy_weights_train = np.zeros((len(y_train), 2), dtype=np.float32)
    dummy_r_train = np.zeros((len(y_train), 2), dtype=np.float32)
    dummy_u_train = np.zeros((len(y_train), 2), dtype=np.float32)
    dummy_logits_train = np.zeros((len(y_train), NUM_ATTACK_CLASSES), dtype=np.float32)

    dummy_weights_val = np.zeros((len(y_val), 2), dtype=np.float32)
    dummy_r_val = np.zeros((len(y_val), 2), dtype=np.float32)
    dummy_u_val = np.zeros((len(y_val), 2), dtype=np.float32)
    dummy_logits_val = np.zeros((len(y_val), NUM_ATTACK_CLASSES), dtype=np.float32)

    # 4. Build and Compile Model
    print("\nBuilding RA-SAFE Architecture...")
    model = build_ra_safe_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=[
            "categorical_crossentropy",  # probs
            "mse",                       # weights
            "mse",                       # r_pair
            "mse",                       # u_pair
            "categorical_crossentropy",  # cyber_aux
            "categorical_crossentropy",  # phys_aux
            "mse",                       # logits
        ],
        loss_weights=[1.0, 0.0, 0.0, 0.0, 0.20, 0.20, 0.0],
        metrics={"probs": ["accuracy"]},
    )
    n_params = model.count_params()
    print(f"RA-SAFE Total Parameters: {n_params:,}")

    # 5. Model Training
    model_save_path = CHECKPOINT_DIR / "ra_safe_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(str(model_save_path), monitor="val_loss", save_best_only=True),
    ]

    print("\nStarting Training...")
    t_train_start = time.perf_counter()
    history = model.fit(
        [xc_train, xp_train, qc_train, qp_train],
        [y_train_oh, dummy_weights_train, dummy_r_train, dummy_u_train, y_train_oh, y_train_oh, dummy_logits_train],
        validation_data=(
            [xc_val, xp_val, qc_val, qp_val],
            [y_val_oh, dummy_weights_val, dummy_r_val, dummy_u_val, y_val_oh, y_val_oh, dummy_logits_val],
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    train_duration = time.perf_counter() - t_train_start
    print(f"Training completed in {train_duration:.1f}s.")

    # 6. Validation Calibration (Temperature Scaling & Threshold Tuning)
    print("\nCalibrating Model on Validation Set...")
    val_out = model.predict([xc_val, xp_val, qc_val, qp_val], verbose=0, batch_size=256)
    val_probs = val_out[0]
    val_r_pair = val_out[2]
    val_logits = val_out[6]

    classifier = SelectiveClassifier()
    learned_t = classifier.fit_temperature(val_logits, y_val)
    print(f"Optimal Calibration Temperature T: {learned_t:.4f}")

    cal_val_probs = classifier.calibrate_probs(val_logits)
    conf_th, rel_th = classifier.tune_thresholds_on_validation(cal_val_probs, val_r_pair, y_val)
    print(f"Optimal Selective Thresholds: tau_conf={conf_th:.2f}, tau_rel={rel_th:.2f}")

    classifier_path = CHECKPOINT_DIR / "selective_classifier.json"
    classifier.save(classifier_path)
    print(f"Saved selective classifier configuration: {classifier_path}")

    # 7. Comprehensive Evaluation on Held-Out Test Set
    print("\nEvaluating RA-SAFE on Held-Out Test Set...")
    t0 = time.perf_counter()
    test_out = model.predict([xc_test, xp_test, qc_test, qp_test], verbose=0, batch_size=256)
    test_time = (time.perf_counter() - t0) * 1000
    ms_per_sample = test_time / max(len(y_test), 1)

    raw_test_probs = test_out[0]
    test_weights = test_out[1]
    test_r_pair = test_out[2]
    test_u_pair = test_out[3]
    test_logits = test_out[6]

    # Apply calibration
    cal_test_probs = classifier.calibrate_probs(test_logits)
    test_preds = np.argmax(cal_test_probs, axis=-1)

    # Standard Classification Metrics
    accuracy = accuracy_score(y_test, test_preds)
    precision_macro = precision_score(y_test, test_preds, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, test_preds, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, test_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, test_preds, labels=list(range(NUM_ATTACK_CLASSES)))

    # Uncertainty & Calibration Metrics
    ece_raw = expected_calibration_error(y_test, raw_test_probs)
    ece_cal = expected_calibration_error(y_test, cal_test_probs)
    bs_val = brier_score(y_test, cal_test_probs)
    nll_val = negative_log_likelihood(y_test, cal_test_probs)

    # Selective Prediction Metrics
    selective_metrics = classifier.evaluate_selective_performance(cal_test_probs, test_r_pair, y_test)

    # Class report
    clf_rep = classification_report(
        y_test, test_preds, labels=list(range(NUM_ATTACK_CLASSES)), target_names=CLASS_ORDER, output_dict=True, zero_division=0
    )
    per_class = {
        name: {
            "precision": clf_rep[name]["precision"],
            "recall": clf_rep[name]["recall"],
            "f1": clf_rep[name]["f1-score"],
            "support": clf_rep[name]["support"],
        }
        for name in CLASS_ORDER
        if name in clf_rep
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"Accuracy:           {accuracy:.2%}")
    print(f"Macro F1:           {f1_macro:.2%}")
    print(f"Weighted F1:        {f1_weighted:.2%}")
    print(f"Inference Latency:  {ms_per_sample:.3f} ms / sample")
    print(f"Parameter Count:    {n_params:,}")
    print("-" * 60)
    print(f"Expected Calibration Error (Before Calibration): {ece_raw:.4f}")
    print(f"Expected Calibration Error (After Calibration):  {ece_cal:.4f}")
    print(f"Brier Score:                                    {bs_val:.4f}")
    print(f"Negative Log-Likelihood (NLL):                   {nll_val:.4f}")
    print("-" * 60)
    print(f"Selective Coverage:                             {selective_metrics['coverage']:.2%}")
    print(f"Accepted Set Accuracy:                          {selective_metrics['accepted_accuracy']:.2%}")
    print(f"Selective Risk:                                 {selective_metrics['selective_risk']:.2%}")
    print(f"Abstention Rate:                                {selective_metrics['abstention_rate']:.2%}")
    print("=" * 60)

    # 8. Save Metrics & Visualizations
    results_payload = {
        "model": "RA-SAFE",
        "fusion": args.fusion,
        "n_params": n_params,
        "inference_ms_per_sample": float(ms_per_sample),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "ece_uncalibrated": float(ece_raw),
        "ece_calibrated": float(ece_cal),
        "brier_score": float(bs_val),
        "negative_log_likelihood": float(nll_val),
        "temperature": float(learned_t),
        "selective_coverage": float(selective_metrics["coverage"]),
        "selective_risk": float(selective_metrics["selective_risk"]),
        "accepted_accuracy": float(selective_metrics["accepted_accuracy"]),
        "abstention_rate": float(selective_metrics["abstention_rate"]),
        "conf_threshold": float(conf_th),
        "rel_threshold": float(rel_th),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_ORDER,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    metrics_file = RESULTS_DIR / "ra_safe_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved test metrics: {metrics_file}")

    # Generate Plots
    cm_path = RESULTS_DIR / "confusion_ra_safe.png"
    plot_confusion(cm, cm_path, "RA-SAFE Multi-Class Confusion Matrix")
    print(f"Saved confusion plot: {cm_path}")

    hist_path = RESULTS_DIR / "history_ra_safe.png"
    plot_training_history(history, hist_path)
    print(f"Saved training curves: {hist_path}")

    gate_path = RESULTS_DIR / "gate_weights_ra_safe.png"
    plot_gate_weights(test_weights, y_test, gate_path)
    print(f"Saved gate weights plot: {gate_path}")


if __name__ == "__main__":
    main()
