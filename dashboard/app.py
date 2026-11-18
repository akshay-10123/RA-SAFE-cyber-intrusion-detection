"""
UAV Cyber-Physical IDS — Results Dashboard

Run:
  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.results_loader import (  # noqa: E402
    FUSION_OPTIONS,
    MODELS,
    all_discovered_plots,
    available_fusions,
    load_all_metrics,
    load_multiclass_metrics,
    multiclass_confusion_plot_path,
    plot_path,
)
from dashboard.cp_attn_view import show_cp_attn  # noqa: E402
from dashboard.explainable_alerts import show_explainable_alerts  # noqa: E402
from dashboard.live_streaming import show_live_streaming  # noqa: E402
from dashboard.ra_safe_view import show_ra_safe_detection  # noqa: E402
from dashboard.robustness_view import show_robustness_comparison  # noqa: E402
from scripts.attack_labels import ATTACK_DISPLAY_NAMES  # noqa: E402

st.set_page_config(
    page_title="UAV IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COMPARE_FUSIONS = ["drone1_cyber", "drone1_physical", "drone1_combined"]
ALGORITHMS = list(MODELS.keys())

FUSION_HELP = {
    "cyber": "Wi‑Fi / network features",
    "physical": "Drone telemetry",
    "combined": "Cyber + physical",
}


def fusion_label(fusion: str) -> str:
    return fusion.replace("drone1_", "").replace("_", " ").title()


def metrics_dataframe(metrics: list) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Algorithm": m.model,
                "Fusion": fusion_label(m.fusion),
                "Fusion key": m.fusion,
                "Accuracy": m.accuracy,
                "Precision": m.precision,
                "Recall": m.recall,
                "F1": m.f1,
                "AUC": m.auc,
            }
            for m in metrics
        ]
    )


def main() -> None:
    st.title("UAV Intrusion Detection — Results Dashboard")
    st.caption("All algorithms · cyber / physical / combined · plots & metrics")

    metrics = load_all_metrics()
    plots = all_discovered_plots()
    df = metrics_dataframe(metrics)

    page = st.sidebar.radio(
        "Navigate",
        [
            "All algorithms",
            "Multi-class attack ID",
            "CP-Attn-Lite",
            "RA-SAFE Detection",
            "RA-SAFE Robustness",
            "Live streaming demo",
            "Explainable alerts",
            "Compare fusions",
            "Model details",
            "All plots",
            "How to run",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.metric("Saved metric runs", len(metrics))
    st.sidebar.metric("Plot images", len(plots))
    if st.sidebar.button("Refresh page"):
        st.rerun()

    if page == "All algorithms":
        show_all_algorithms(df, metrics)
    elif page == "Multi-class attack ID":
        show_multiclass_attack_id()
    elif page == "CP-Attn-Lite":
        show_cp_attn()
    elif page == "RA-SAFE Detection":
        show_ra_safe_detection()
    elif page == "RA-SAFE Robustness":
        show_robustness_comparison()
    elif page == "Live streaming demo":
        show_live_streaming()
    elif page == "Explainable alerts":
        show_explainable_alerts()
    elif page == "Compare fusions":
        show_compare_fusions(df, metrics)
    elif page == "Model details":
        show_model_details(metrics)
    elif page == "All plots":
        show_gallery(plots)
    else:
        show_how_to_run()


def show_all_algorithms(df: pd.DataFrame, metrics: list) -> None:
    st.subheader("All algorithm results")

    if df.empty:
        st.warning(
            "No metrics yet. Run all experiments first:\n\n"
            "`python scripts/run_all_experiments.py`"
        )
        show_how_to_run()
        return

    compare_df = df[df["Fusion key"].isin(COMPARE_FUSIONS)]
    if compare_df.empty:
        st.info("Run models with drone1_cyber, drone1_physical, drone1_combined.")
        compare_df = df

    # Summary cards per algorithm
    cols = st.columns(len(ALGORITHMS))
    for col, algo in zip(cols, ALGORITHMS):
        sub = compare_df[compare_df["Algorithm"] == algo]
        with col:
            st.markdown(f"**{algo}**")
            st.write(f"{len(sub)} / 3 fusions done")
            if not sub.empty:
                st.write(f"Best F1: **{sub['F1'].max():.1%}**")

    st.markdown("### Metrics table (all runs)")
    display = compare_df.drop(columns=["Fusion key"])
    st.dataframe(
        display.style.format(
            {"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}", "AUC": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### F1 score — every algorithm × fusion")
    pivot = compare_df.pivot(index="Algorithm", columns="Fusion", values="F1")
    if not pivot.empty:
        st.bar_chart(pivot)
        st.dataframe(pivot.style.format("{:.1%}"), use_container_width=True)

    st.markdown("### Accuracy heatmap")
    pivot_acc = compare_df.pivot(index="Algorithm", columns="Fusion", values="Accuracy")
    if not pivot_acc.empty:
        st.dataframe(
            pivot_acc.style.background_gradient(cmap="Greens", axis=None).format("{:.1%}"),
            use_container_width=True,
        )

    missing = []
    for algo in ALGORITHMS:
        for fusion in COMPARE_FUSIONS:
            if not any(m.model == algo and m.fusion == fusion for m in metrics):
                missing.append(f"{algo} · {fusion_label(fusion)}")
    if missing:
        with st.expander(f"Missing runs ({len(missing)})"):
            for item in missing:
                st.write(f"- {item}")


def show_multiclass_attack_id() -> None:
    st.subheader("Multi-class cyber-physical IDS")
    st.caption(
        "Identifies attack type: benign, DoS, replay, evil twin, or FDI "
        "(binary IDS pages above are unchanged)."
    )

    rows = load_multiclass_metrics()
    if not rows:
        st.warning(
            "No multi-class results yet. Run:\n\n"
            "`python scripts/prepare_dataset.py`\n\n"
            "`python scripts/run_multiclass_experiments.py --quick`"
        )
        return

    table = pd.DataFrame(
        [
            {
                "Fusion": fusion_label(r.fusion),
                "Accuracy": r.accuracy,
                "F1 (macro)": r.f1_macro,
                "F1 (weighted)": r.f1_weighted,
                "Precision (macro)": r.precision_macro,
                "Recall (macro)": r.recall_macro,
                "AUC (macro)": r.auc_macro,
            }
            for r in rows
        ]
    )
    st.dataframe(
        table.style.format(
            {
                "Accuracy": "{:.1%}",
                "F1 (macro)": "{:.1%}",
                "F1 (weighted)": "{:.1%}",
                "Precision (macro)": "{:.1%}",
                "Recall (macro)": "{:.1%}",
                "AUC (macro)": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fusion_keys = [r.fusion for r in rows]
    selected = st.selectbox("Fusion detail", fusion_keys, format_func=fusion_label)
    row = next(r for r in rows if r.fusion == selected)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{row.accuracy:.1%}")
    c2.metric("Macro F1", f"{row.f1_macro:.1%}")
    c3.metric("Macro recall", f"{row.recall_macro:.1%}")
    c4.metric("Macro AUC", f"{row.auc_macro:.3f}")

    cm_path = multiclass_confusion_plot_path(selected)
    if cm_path:
        st.markdown("#### Confusion matrix")
        st.image(str(cm_path), use_container_width=True)

    if row.per_class:
        st.markdown("#### Per attack type")
        per_df = pd.DataFrame(row.per_class).T
        st.dataframe(
            per_df.style.format({"precision": "{:.1%}", "recall": "{:.1%}", "f1": "{:.1%}"}),
            use_container_width=True,
        )

    with st.expander("What each class means"):
        for i in sorted(ATTACK_DISPLAY_NAMES):
            st.write(f"- **{ATTACK_DISPLAY_NAMES[i]}**")


def show_compare_fusions(df: pd.DataFrame, metrics: list) -> None:
    st.subheader("Cyber vs Physical vs Combined")

    algo = st.selectbox("Algorithm", ALGORITHMS)
    sub = [m for m in metrics if m.model == algo and m.fusion in COMPARE_FUSIONS]

    if not sub:
        st.warning(f"No metrics for **{algo}** yet.")
        return

    chart_df = pd.DataFrame(
        {
            "Fusion": [fusion_label(m.fusion) for m in sub],
            "Accuracy": [m.accuracy for m in sub],
            "F1": [m.f1 for m in sub],
            "AUC": [m.auc for m in sub],
        }
    ).set_index("Fusion")
    st.bar_chart(chart_df)

    cols = st.columns(3)
    for col, fusion in zip(cols, COMPARE_FUSIONS):
        with col:
            st.markdown(f"**{fusion_label(fusion)}**")
            row = next((m for m in sub if m.fusion == fusion), None)
            if row:
                st.write(f"Acc {row.accuracy:.1%} · F1 {row.f1:.1%}")
            info = MODELS[algo]
            for title, template in info["plots"]:
                path = plot_path(algo, fusion, template)
                if path:
                    st.image(str(path), caption=title, use_container_width=True)


def show_model_details(metrics: list) -> None:
    st.subheader("Single model view")
    algo = st.selectbox("Algorithm", ALGORITHMS, key="detail_algo")
    fusions = available_fusions(algo) or FUSION_OPTIONS
    fusion = st.selectbox("Fusion", fusions, key="detail_fusion")

    row = next((m for m in metrics if m.model == algo and m.fusion == fusion), None)
    if row:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{row.accuracy:.1%}")
        c2.metric("Precision", f"{row.precision:.1%}")
        c3.metric("Recall", f"{row.recall:.1%}")
        c4.metric("F1", f"{row.f1:.1%}")
        c5.metric("AUC", f"{row.auc:.3f}")
    else:
        st.info("Metrics not saved for this run yet.")

    for title, template in MODELS[algo]["plots"]:
        path = plot_path(algo, fusion, template)
        if path:
            st.markdown(f"#### {title}")
            st.image(str(path), use_container_width=True)


def show_gallery(plots: list) -> None:
    st.subheader("All plots")
    if not plots:
        st.warning("No plots found.")
        return
    filt_algo = st.multiselect("Algorithm", ALGORITHMS, default=ALGORITHMS)
    filtered = [p for p in plots if p["model"] in filt_algo]
    cols = st.columns(2)
    for i, item in enumerate(filtered):
        with cols[i % 2]:
            st.markdown(f"**{item['model']}** · `{item['fusion']}`")
            st.image(str(item["path"]), use_container_width=True)


def show_how_to_run() -> None:
    st.subheader("Run all algorithms (one command)")
    st.code(
        r"""cd "C:\Users\HP\OneDrive\Documents\Cyber-Physical-Intrusion-Detection-System-for-Unmanned-Aerial-Vehicles"
C:\uav-venv\Scripts\python.exe scripts\prepare_dataset.py
C:\uav-venv\Scripts\python.exe scripts\run_all_experiments.py
C:\uav-venv\Scripts\python.exe -m streamlit run dashboard/app.py""",
        language="powershell",
    )
    st.markdown("Quick test (fewer epochs): `python scripts/run_all_experiments.py --quick`")
    st.markdown("**Multi-class attack ID (phase 1):**")
    st.code(
        r"""python scripts\prepare_dataset.py
python scripts\run_multiclass_experiments.py --quick""",
        language="powershell",
    )
    st.markdown("**Live streaming demo (phase 2):**")
    st.code(
        r"""python "1D-CNN -based IDS\multiclass_cnn.py" --fusion central_combined --epochs 10
python scripts\live_stream_demo.py --fusion central_combined --start 13700 --steps 50
python -m streamlit run dashboard/app.py  →  Live streaming demo""",
        language="powershell",
    )
    st.markdown("**Explainable alerts (phase 3):** Dashboard → **Explainable alerts** or live on streaming page.")
    st.markdown("Full manual commands are in the **`commands`** file in the project root.")


if __name__ == "__main__":
    main()
