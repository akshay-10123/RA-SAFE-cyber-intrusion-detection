"""
Robustness & Degradation Stress Testing: RA-SAFE vs Baseline CP-Attn-Lite.

Evaluates:
  Experiment A: Nominal Conditions (Clean test set)
  Experiment B: Network Degradation (Packet loss 10%, 20%, RF Jamming)
  Experiment C: Physical Sensor Degradation (IMU Noise sigma=1.0, 2.0, Stuck-at freeze)
  Experiment D: Simultaneous Cyber-Physical Degradation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.ra_safe_model import ReliabilityAdaptiveFusionGate, SqueezeExcite, build_ra_safe_model
from models.reliability_estimator import ModalityReliabilityEstimator
from models.selective_classifier import SelectiveClassifier
from scripts.attack_labels import NUM_ATTACK_CLASSES
from scripts.cp_attn_model import FusionGate as CP_FusionGate
from scripts.cp_attn_model import SqueezeExcite as CP_SqueezeExcite
from scripts.multiclass_data import frame_to_xy, load_fusion_frame
from utils.corruption_simulator import simulate_degradation

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CP_MODEL_PATH = PROJECT_ROOT / "1D-CNN -based IDS" / "cp_attn_models" / "cp_attn_central_combined.keras"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_ra_safe_stack():
    """Load trained RA-SAFE model, reliability estimator, and selective classifier."""
    model_path = CHECKPOINT_DIR / "ra_safe_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"RA-SAFE model not found at {model_path}. Run training first.")

    # Load with custom layer mappings
    custom_objs = {
        "ReliabilityAdaptiveFusionGate": ReliabilityAdaptiveFusionGate,
        "SqueezeExcite": SqueezeExcite,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)

    rel_path = CHECKPOINT_DIR / "reliability_estimator.json"
    estimator = ModalityReliabilityEstimator.load(rel_path)

    cls_path = CHECKPOINT_DIR / "selective_classifier.json"
    classifier = SelectiveClassifier.load(cls_path)

    return model, estimator, classifier


def load_cp_attn_baseline():
    """Load baseline CP-Attn-Lite model if available."""
    if not CP_MODEL_PATH.exists():
        return None
    custom_objs = {
        "FusionGate": CP_FusionGate,
        "SqueezeExcite": CP_SqueezeExcite,
    }
    return tf.keras.models.load_model(CP_MODEL_PATH, custom_objects=custom_objs)


def run_stress_benchmarks(fusion: str = "central_combined", max_samples: int = 0):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test dataset...")
    df = load_fusion_frame(fusion)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    x_all, y_all = frame_to_xy(df)

    # Use exact same test split as training
    _, x_temp, _, y_temp = train_test_split(x_all, y_all, test_size=0.30, random_state=42, stratify=y_all)
    _, x_test, _, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    xc_test_clean = x_test[:, :21]
    xp_test_clean = x_test[:, 21:]

    ra_model, estimator, classifier = load_ra_safe_stack()
    cp_model = load_cp_attn_baseline()

    scenarios = [
        {"name": "Nominal (Clean)", "mode": "nominal", "intensity": 0.0},
        {"name": "Packet Loss 10%", "mode": "cyber_packet_loss", "intensity": 0.10},
        {"name": "Packet Loss 20%", "mode": "cyber_packet_loss", "intensity": 0.20},
        {"name": "RF Jamming", "mode": "cyber_jamming", "intensity": 1.0},
        {"name": "Sensor Noise (std=1.0)", "mode": "phys_noise", "intensity": 1.0},
        {"name": "Sensor Noise (std=2.0)", "mode": "phys_noise", "intensity": 2.0},
        {"name": "Sensor Freeze (Stuck)", "mode": "phys_stuck", "intensity": 1.0},
        {"name": "Simultaneous Degr.", "mode": "simultaneous", "intensity": 1.0},
    ]

    results = []

    print("\n" + "=" * 90)
    print(f"{'Condition':<22} | {'CP-Attn Acc':<11} | {'RA-SAFE Acc':<11} | {'Coverage':<9} | {'Accepted Acc':<12} | {'w_cyber':<8} | {'w_phys':<8}")
    print("=" * 90)

    for sc in scenarios:
        # Corrupt test inputs
        xc_curr, xp_curr = simulate_degradation(
            xc_test_clean, xp_test_clean, mode=sc["mode"], intensity=sc["intensity"], seed=42
        )

        # Baseline CP-Attn-Lite
        cp_acc = 0.0
        cp_f1 = 0.0
        if cp_model is not None:
            xc_in = xc_curr.reshape(-1, 21, 1)
            xp_in = xp_curr.reshape(-1, 11, 1)
            cp_out = cp_model.predict([xc_in, xp_in], verbose=0, batch_size=256)
            cp_probs = cp_out[0]
            cp_preds = np.argmax(cp_probs, axis=-1)
            cp_acc = float(accuracy_score(y_test, cp_preds))
            cp_f1 = float(f1_score(y_test, cp_preds, average="macro", zero_division=0))

        # RA-SAFE Evaluation
        qc_curr, qp_curr = estimator.estimate(xc_curr, xp_curr)
        xc_in = xc_curr.reshape(-1, 21, 1)
        xp_in = xp_curr.reshape(-1, 11, 1)

        ra_out = ra_model.predict([xc_in, xp_in, qc_curr, qp_curr], verbose=0, batch_size=256)
        test_probs = ra_out[0]
        test_weights = ra_out[1]
        test_r_pair = ra_out[2]
        test_logits = ra_out[6]

        # Calibrated selective prediction
        cal_probs = classifier.calibrate_probs(test_logits)
        raw_preds = np.argmax(cal_probs, axis=-1)
        raw_acc = float(accuracy_score(y_test, raw_preds))
        raw_f1 = float(f1_score(y_test, raw_preds, average="macro", zero_division=0))

        selective = classifier.evaluate_selective_performance(cal_probs, test_r_pair, y_test)

        mean_wc = float(np.mean(test_weights[:, 0]))
        mean_wp = float(np.mean(test_weights[:, 1]))
        mean_rc = float(np.mean(test_r_pair[:, 0]))
        mean_rp = float(np.mean(test_r_pair[:, 1]))

        print(
            f"{sc['name']:<22} | {cp_acc:>10.1%} | {raw_acc:>10.1%} | {selective['coverage']:>8.1%} | "
            f"{selective['accepted_accuracy']:>11.1%} | {mean_wc:>7.2f} | {mean_wp:>7.2f}"
        )

        results.append(
            {
                "condition": sc["name"],
                "mode": sc["mode"],
                "intensity": sc["intensity"],
                "cp_attn_accuracy": cp_acc,
                "cp_attn_f1": cp_f1,
                "ra_safe_raw_accuracy": raw_acc,
                "ra_safe_raw_f1": raw_f1,
                "coverage": selective["coverage"],
                "accepted_accuracy": selective["accepted_accuracy"],
                "abstention_rate": selective["abstention_rate"],
                "selective_risk": selective["selective_risk"],
                "mean_cyber_weight": mean_wc,
                "mean_phys_weight": mean_wp,
                "mean_cyber_reliability": mean_rc,
                "mean_phys_reliability": mean_rp,
            }
        )

    print("=" * 90)

    # Save to JSON
    out_file = RESULTS_DIR / "robustness_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"benchmark_results": results}, f, indent=2)
    print(f"\nSaved robustness metrics: {out_file}")

    # Plot Comparison
    names = [r["condition"] for r in results]
    cp_accs = [r["cp_attn_accuracy"] for r in results]
    ra_accs = [r["ra_safe_raw_accuracy"] for r in results]
    cov_accs = [r["accepted_accuracy"] for r in results]
    coverages = [r["coverage"] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(13, 6))
    if cp_model is not None:
        ax1.bar(x - width, cp_accs, width, label="Baseline CP-Attn-Lite (No Abstention)", color="#7f7f7f")
    ax1.bar(x, ra_accs, width, label="RA-SAFE Raw Accuracy", color="#1f77b4")
    ax1.bar(x + width, cov_accs, width, label="RA-SAFE Accepted Accuracy (Selective)", color="#2ca02c")

    ax1.set_ylabel("Accuracy")
    ax1.set_title("Robustness Stress Benchmark: Accuracy Under Severe Network & Sensor Degradations")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right")
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Line for coverage on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, coverages, color="#d62728", marker="o", lw=2.5, label="RA-SAFE Coverage Rate")
    ax2.set_ylabel("Coverage Rate (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0.0, 1.1)

    fig.tight_layout()
    plot_path = RESULTS_DIR / "robustness_comparison.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved robustness comparison plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", default="central_combined")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    run_stress_benchmarks(args.fusion, args.max_samples)
