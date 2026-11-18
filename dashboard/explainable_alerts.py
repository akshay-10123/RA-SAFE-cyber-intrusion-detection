"""Phase 3 — explainable alert system dashboard page."""
from __future__ import annotations

import streamlit as st

from dashboard.alert_panel import render_alert_card, render_attack_catalog
from scripts.attack_info import CONFIDENCE_ALERT_THRESHOLD, all_attack_infos
from scripts.attack_labels import ATTACK_DISPLAY_NAMES, NUM_ATTACK_CLASSES


def show_explainable_alerts() -> None:
    st.subheader("Explainable alert system")
    st.caption(
        "Operator-friendly descriptions, severity, and mitigation steps for each detected attack type. "
        "Alerts appear automatically on the **Live streaming demo** when the model classifies traffic."
    )

    st.markdown("### Simulate an alert")
    st.write(f"Alerts fire when confidence ≥ **{CONFIDENCE_ALERT_THRESHOLD:.0%}** and class is not benign.")

    class_id = st.selectbox(
        "Predicted class",
        list(range(NUM_ATTACK_CLASSES)),
        format_func=lambda i: ATTACK_DISPLAY_NAMES[i],
        index=1,
    )
    confidence = st.slider("Model confidence", 0.0, 1.0, 0.92, 0.01)
    render_alert_card(class_id, confidence, show_benign=True)

    st.markdown("---")
    st.markdown("### Attack reference (all classes)")
    render_attack_catalog()

    st.markdown("---")
    st.info(
        "During a live stream, the same alert card is shown for the **current prediction** "
        "with real model confidence and inference latency."
    )
