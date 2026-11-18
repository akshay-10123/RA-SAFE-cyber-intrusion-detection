"""Dashboard page for CP-Attn-Lite fusion weights and metrics."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.results_loader import cp_attn_confusion_plot_path, load_multiclass_metrics
from scripts.attack_labels import ATTACK_DISPLAY_NAMES, ATTACK_CLASS_NAMES


def show_cp_attn() -> None:
    st.subheader("CP-Attn-Lite — dual-branch CNN")
    st.caption(
        "Cyber and physical features are encoded separately. A fusion gate learns "
        "how much each sample depends on the network branch versus the sensor branch."
    )

    rows = [r for r in load_multiclass_metrics() if r.model == "CP-Attn-Lite"]
    if not rows:
        st.warning(
            "No CP-Attn-Lite run yet.\n\n"
            '`python "1D-CNN -based IDS/cp_attn_cnn.py" --fusion central_combined --epochs 8`'
        )
        return

    table = pd.DataFrame(
        [
            {
                "Fusion": r.fusion,
                "Accuracy": r.accuracy,
                "Macro F1": r.f1_macro,
                "Params": r.n_params or 0,
                "ms / sample": r.inference_ms_per_sample or 0.0,
            }
            for r in rows
        ]
    )
    st.dataframe(
        table.style.format({"Accuracy": "{:.1%}", "Macro F1": "{:.1%}", "ms / sample": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    selected_key = st.selectbox("Run", [r.fusion for r in rows])
    row = next(r for r in rows if r.fusion == selected_key)

    c1, c2, c3 = st.columns(3)
    c1.metric("Macro F1", f"{row.f1_macro:.1%}")
    c2.metric("Parameters", f"{row.n_params:,}" if row.n_params else "—")
    c3.metric(
        "Inference",
        f"{row.inference_ms_per_sample:.2f} ms" if row.inference_ms_per_sample else "—",
    )

    plot = cp_attn_confusion_plot_path(selected_key)
    if plot:
        st.image(str(plot), caption="Confusion matrix", use_container_width=True)

    if row.fusion_by_class:
        st.markdown("#### Fusion gate — mean weight by true class")
        st.caption("Higher cyber weight means the network branch drove that class; higher physical weight means sensors did.")
        names = []
        cyber = []
        physical = []
        for key, stats in row.fusion_by_class.items():
            class_id = next((i for i, name in ATTACK_CLASS_NAMES.items() if name == key), None)
            names.append(ATTACK_DISPLAY_NAMES.get(class_id, key) if class_id is not None else key)
            cyber.append(stats["cyber_weight"])
            physical.append(stats["physical_weight"])
        chart = pd.DataFrame({"Cyber": cyber, "Physical": physical}, index=names)
        st.bar_chart(chart)
        st.dataframe(chart.style.format("{:.2f}"), use_container_width=True)
