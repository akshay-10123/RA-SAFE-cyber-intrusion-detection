"""
Streamlit Diagnostic View for RA-SAFE (Reliability-Aware Selective Fusion).

Displays:
  1. Live inference diagnostics: predicted class, calibrated confidence,
     cyber & physical reliability, adaptive fusion weights, uncertainty,
     acceptance/abstention status, and operator recommendations.
  2. Interactive Degraded-Input Simulation: test how packet loss, RF jamming,
     sensor noise, and sensor freeze dynamically modulate reliability and gate weights.
  3. Model performance overview and confusion matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

from models.ra_safe_model import ReliabilityAdaptiveFusionGate, SqueezeExcite
from models.reliability_estimator import ModalityReliabilityEstimator
from models.selective_classifier import SelectiveClassifier
from scripts.attack_labels import ATTACK_CLASS_NAMES, ATTACK_DISPLAY_NAMES, NUM_ATTACK_CLASSES
from scripts.multiclass_data import frame_to_xy, load_fusion_frame
from utils.corruption_simulator import simulate_degradation


@st.cache_resource(show_spinner="Loading RA-SAFE neural network...")
def _get_ra_safe_model():
    model_path = CHECKPOINT_DIR / "ra_safe_model.keras"
    if not model_path.exists():
        return None
    custom_objs = {
        "ReliabilityAdaptiveFusionGate": ReliabilityAdaptiveFusionGate,
        "SqueezeExcite": SqueezeExcite,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objs)


@st.cache_resource(show_spinner="Loading reliability & calibration modules...")
def _get_estimators():
    rel_path = CHECKPOINT_DIR / "reliability_estimator.json"
    cls_path = CHECKPOINT_DIR / "selective_classifier.json"
    if not rel_path.exists() or not cls_path.exists():
        return None, None
    estimator = ModalityReliabilityEstimator.load(rel_path)
    classifier = SelectiveClassifier.load(cls_path)
    return estimator, classifier


@st.cache_data(show_spinner="Loading dataset rows...")
def _get_sample_data():
    df = load_fusion_frame("central_combined")
    x, y = frame_to_xy(df)
    return x, y


def show_ra_safe_detection() -> None:
    st.subheader("🛡️ RA-SAFE: Reliability-Aware Selective Fusion")
    st.caption(
        "Extends CP-Attn-Lite with Modality Reliability Estimators (Q), Predictive Uncertainty (U), "
        "Learnable Adaptive Gating (w), and Calibrated Selective Prediction (Accept vs Abstain)."
    )

    model = _get_ra_safe_model()
    estimator, classifier = _get_estimators()

    if model is None or estimator is None or classifier is None:
        st.warning(
            "RA-SAFE model or calibration checkpoints not found.\n\n"
            "Run training first:\n"
            "`python training/train_ra_safe.py --epochs 12`"
        )
        return

    # 1. Performance Overview Banner
    metrics_file = RESULTS_DIR / "ra_safe_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            m = json.load(f)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Unfiltered Acc", f"{m['accuracy']:.1%}")
        c2.metric("Accepted Acc", f"{m['accepted_accuracy']:.1%}")
        c3.metric("Selective Risk", f"{m['selective_risk']:.1%}")
        c4.metric("Parameters", f"{m['n_params']:,}")
        c5.metric("Latency", f"{m['inference_ms_per_sample']:.2f} ms")

    st.markdown("---")

    # 2. Interactive Sample Prediction & Stress Test
    st.markdown("### 🔬 Live Inference & Degradation Stress Test")
    st.write(
        "Select a dataset row to inspect nominal inference, or inject real-time "
        "cyber/physical corruptions to observe how reliability and fusion weights adapt."
    )

    x_all, y_all = _get_sample_data()
    n_total = len(x_all)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        row_idx = st.number_input("Row Index", min_value=0, max_value=n_total - 1, value=13750, step=25)
    with col_ctrl2:
        degradation_mode = st.selectbox(
            "Inject Degradation",
            [
                "nominal (Clean Input)",
                "cyber_packet_loss (20% Loss)",
                "cyber_jamming (RF Jamming / SNR Collapse)",
                "phys_noise (IMU Vibration Noise)",
                "phys_stuck (Sensor Freeze)",
                "simultaneous (Packet Loss + Sensor Noise)",
            ],
        )
    with col_ctrl3:
        intensity = st.slider("Degradation Intensity", 0.1, 3.0, 1.0, 0.1)

    # Prepare inputs
    x_sample = x_all[row_idx : row_idx + 1]
    y_true = int(y_all[row_idx])
    xc = x_sample[:, :21]
    xp = x_sample[:, 21:]

    # Parse degradation
    mode_key = degradation_mode.split()[0]
    if mode_key != "nominal":
        xc_sim, xp_sim = simulate_degradation(xc, xp, mode=mode_key, intensity=intensity, seed=42)
    else:
        xc_sim, xp_sim = xc, xp

    # Compute Quality
    qc, qp = estimator.estimate(xc_sim, xp_sim)

    # Forward Pass
    xc_in = xc_sim.reshape(1, 21, 1)
    xp_in = xp_sim.reshape(1, 11, 1)
    outputs = model.predict([xc_in, xp_in, qc, qp], verbose=0)

    raw_probs = outputs[0][0]
    weights = outputs[1][0]  # [w_c, w_p]
    r_pair = outputs[2][0]   # [R_c, R_p]
    u_pair = outputs[3][0]   # [U_c, U_p]
    logits = outputs[6][0]

    # Selective Decision
    decision_info = classifier.predict_sample(
        logits=logits,
        r_c=float(r_pair[0]),
        r_p=float(r_pair[1]),
        w_c=float(weights[0]),
        w_p=float(weights[1]),
        class_names=[ATTACK_CLASS_NAMES[i] for i in range(NUM_ATTACK_CLASSES)],
    )

    # Display Decision Cards
    st.markdown("#### 📡 Decision & Reliability Dashboard")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)

    is_accepted = decision_info["decision"] == "ACCEPTED"
    pred_display = ATTACK_DISPLAY_NAMES.get(decision_info["raw_predicted_class_id"], "Unknown")
    true_display = ATTACK_DISPLAY_NAMES.get(y_true, "Unknown")

    with d_col1:
        if is_accepted:
            st.success(f"### Decision\n**ACCEPTED**")
        else:
            st.error(f"### Decision\n**ABSTAINED**")
        st.caption(f"Ground Truth: **{true_display}**")

    with d_col2:
        st.metric("Predicted Class", pred_display if is_accepted else "Uncertain (Abstained)")
        st.metric("Calibrated Confidence", f"{decision_info['confidence']:.1%}")

    with d_col3:
        st.metric("Cyber Reliability (R_c)", f"{decision_info['cyber_reliability']:.2f}")
        st.metric("Physical Reliability (R_p)", f"{decision_info['phys_reliability']:.2f}")

    with d_col4:
        st.metric("Cyber Fusion Weight (w_c)", f"{decision_info['cyber_weight']:.2f}")
        st.metric("Physical Fusion Weight (w_p)", f"{decision_info['phys_weight']:.2f}")

    # Operator Playbook
    if is_accepted:
        st.info(f"**Operator Recommendation:**\n{decision_info['operator_recommendation']}")
    else:
        st.warning(f"**Operator Action Playbook (Abstention Triggered):**\n{decision_info['operator_recommendation']}")

    # Probability Chart
    st.markdown("#### Class Probabilities (Calibrated via Temperature Scaling)")
    prob_df = pd.DataFrame(
        {
            "Attack Class": [ATTACK_DISPLAY_NAMES[i] for i in range(NUM_ATTACK_CLASSES)],
            "Probability": decision_info["calibrated_probabilities"],
        }
    ).set_index("Attack Class")
    st.bar_chart(prob_df)

    # 3. Model Diagnostic Artifacts
    st.markdown("---")
    st.markdown("### 📊 Model Diagnostic Artifacts")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Training Curves", "Adaptive Fusion Weights"])

    with tab1:
        cm_path = RESULTS_DIR / "confusion_ra_safe.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="RA-SAFE Held-Out Test Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found.")

    with tab2:
        hist_path = RESULTS_DIR / "history_ra_safe.png"
        if hist_path.exists():
            st.image(str(hist_path), caption="RA-SAFE Accuracy & Loss Convergence", use_container_width=True)
        else:
            st.info("Training history plot not found.")

    with tab3:
        gate_path = RESULTS_DIR / "gate_weights_ra_safe.png"
        if gate_path.exists():
            st.image(str(gate_path), caption="Mean Cyber vs Physical Fusion Weights by Class", use_container_width=True)
        else:
            st.info("Gate weights plot not found.")
