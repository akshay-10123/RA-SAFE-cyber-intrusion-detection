"""Streamlit UI — phase 2 live streaming demo."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.alert_panel import render_alert_card  # noqa: E402
from scripts.attack_info import get_attack_info, is_intrusion  # noqa: E402
from scripts.attack_labels import ATTACK_CLASS_NAMES, ATTACK_DISPLAY_NAMES, NUM_ATTACK_CLASSES  # noqa: E402
from scripts.multiclass_data import (  # noqa: E402
    available_stream_models,
    frame_to_xy,
    load_fusion_frame,
    model_path,
    reshape_for_cnn,
)


@st.cache_resource(show_spinner="Loading multi-class model…")
def _load_model(fusion: str):
    import tensorflow as tf

    path = model_path(fusion)
    if not path.exists():
        return None
    return tf.keras.models.load_model(path)


@st.cache_data(show_spinner="Loading stream dataset…")
def _load_xy(fusion: str):
    df = load_fusion_frame(fusion)
    return frame_to_xy(df)


def _init_session() -> None:
    defaults = {
        "stream_running": False,
        "stream_pos": 0,
        "stream_log": [],
        "stream_fusion": "central_combined",
        "stream_start": 13700,
        "stream_end": 13900,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _predict_row(model, x_row: np.ndarray) -> tuple[int, float, np.ndarray, float]:
    t0 = time.perf_counter()
    proba = model.predict(reshape_for_cnn(x_row), verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    pred = int(np.argmax(proba))
    conf = float(proba[pred])
    return pred, conf, proba, latency_ms


def show_live_streaming() -> None:
    _init_session()
    st.subheader("Live streaming IDS (simulated)")
    st.caption(
        "Replays standardized CSV rows like a real-time telemetry feed. "
        "Uses the saved multi-class 1D-CNN model from phase 1."
    )

    models = available_stream_models()
    if not models:
        st.warning(
            "No saved multi-class model found.\n\n"
            'Train and save: `python "1D-CNN -based IDS/multiclass_cnn.py" '
            "--fusion central_combined --epochs 10`"
        )
        return

    fusion = st.selectbox(
        "Fusion / model",
        models,
        index=models.index(st.session_state.stream_fusion)
        if st.session_state.stream_fusion in models
        else 0,
        key="stream_fusion_select",
    )
    st.session_state.stream_fusion = fusion

    model = _load_model(fusion)
    if model is None:
        st.error(f"Missing model file: `{model_path(fusion)}`")
        return

    try:
        x_all, y_all = _load_xy(fusion)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    max_idx = len(x_all) - 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start = st.number_input("Start row", 0, max_idx, int(st.session_state.stream_start))
    with c2:
        end_default = min(int(start) + 500, max_idx)
        end = st.number_input("End row", int(start), max_idx, end_default)
    with c3:
        delay_ms = st.slider("Delay (ms)", 0, 500, 80)
    with c4:
        st.number_input("Dataset rows", value=len(x_all), disabled=True)

    st.session_state.stream_start = int(start)
    st.session_state.stream_end = int(end)

    if st.session_state.stream_pos < int(start) or st.session_state.stream_pos > int(end):
        st.session_state.stream_pos = int(start)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Start stream", type="primary"):
            st.session_state.stream_running = True
            if st.session_state.stream_pos >= int(end):
                st.session_state.stream_pos = int(start)
                st.session_state.stream_log = []
    with b2:
        if st.button("Pause"):
            st.session_state.stream_running = False
    with b3:
        if st.button("Step +1"):
            st.session_state.stream_running = False
            st.session_state.stream_pos = min(st.session_state.stream_pos + 1, int(end))
    with b4:
        if st.button("Reset"):
            st.session_state.stream_running = False
            st.session_state.stream_pos = int(start)
            st.session_state.stream_log = []

    pos = st.session_state.stream_pos
    if int(start) <= pos <= int(end):
        pred, conf, proba, latency_ms = _predict_row(model, x_all[pos])
        true_id = int(y_all[pos])
        match = pred == true_id

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stream row", f"{pos:,} / {end:,}")
        m2.metric("Predicted", ATTACK_DISPLAY_NAMES.get(pred, ATTACK_CLASS_NAMES[pred]))
        m3.metric("Confidence", f"{conf:.1%}")
        m4.metric("Inference", f"{latency_ms:.0f} ms")

        true_label = ATTACK_DISPLAY_NAMES.get(true_id, ATTACK_CLASS_NAMES[true_id])
        if match:
            st.success(f"Ground truth: **{true_label}** — prediction matches.")
        else:
            st.error(f"Ground truth: **{true_label}** — model predicted **{ATTACK_DISPLAY_NAMES.get(pred, '?')}**.")

        st.markdown("#### Explainable alert")
        render_alert_card(pred, conf, show_benign=True)

        prob_df = pd.DataFrame(
            {"Class": [ATTACK_DISPLAY_NAMES[i] for i in range(NUM_ATTACK_CLASSES)], "Probability": proba}
        ).set_index("Class")
        st.bar_chart(prob_df)

        pred_info = get_attack_info(pred)
        entry = {
            "row": pos,
            "true": ATTACK_CLASS_NAMES[true_id],
            "predicted": ATTACK_CLASS_NAMES[pred],
            "severity": pred_info.severity,
            "alert": is_intrusion(pred, conf),
            "confidence": conf,
            "latency_ms": latency_ms,
            "match": match,
        }
        if not st.session_state.stream_log or st.session_state.stream_log[-1]["row"] != pos:
            st.session_state.stream_log.append(entry)

    if st.session_state.stream_log:
        log_df = pd.DataFrame(st.session_state.stream_log)
        rolling_acc = log_df["match"].mean()
        avg_lat = log_df["latency_ms"].mean()
        s1, s2, s3 = st.columns(3)
        s1.metric("Stream accuracy", f"{rolling_acc:.1%}")
        s2.metric("Samples seen", len(log_df))
        s3.metric("Avg latency", f"{avg_lat:.0f} ms")
        st.markdown("#### Stream log (latest 50)")
        st.dataframe(log_df.tail(50), use_container_width=True, hide_index=True)

    if st.session_state.stream_running and pos < int(end):
        st.session_state.stream_pos = pos + 1
        time.sleep(delay_ms / 1000.0)
        st.rerun()
    elif st.session_state.stream_running and pos >= int(end):
        st.session_state.stream_running = False
        st.info("Stream finished — press Reset or Start to replay.")

    with st.expander("Demo tip for guide"):
        st.markdown(
            "Start row **13700** crosses from benign into **DoS** in the T-ITS layout. "
            "Use **central_combined** so all five attack types exist in the full dataset."
        )
