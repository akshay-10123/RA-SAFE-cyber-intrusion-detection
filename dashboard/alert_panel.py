"""Render explainable IDS alert cards in Streamlit."""
from __future__ import annotations

import streamlit as st

from scripts.attack_info import AttackInfo, alert_headline, get_attack_info, is_intrusion


def _severity_emoji(severity: str) -> str:
    return {
        "None": "🟢",
        "Low": "🟡",
        "Medium": "🟠",
        "High": "🔴",
        "Critical": "🛑",
    }.get(severity, "⚪")


def render_alert_card(class_id: int, confidence: float, *, show_benign: bool = True) -> None:
    info = get_attack_info(class_id)
    if info.class_id == 0 and not show_benign:
        return

    headline = alert_headline(class_id, confidence)
    intrusion = is_intrusion(class_id, confidence)

    if intrusion:
        st.error(f"**{headline}**")
    elif class_id == 0:
        st.success(f"**{headline}**")
    else:
        st.warning(f"**{headline}**")

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Attack type:** {info.title}")
    c2.markdown(f"**Layer:** {info.layer}")
    c3.markdown(f"**Severity:** {_severity_emoji(info.severity)} {info.severity}")

    st.markdown(f"**Confidence:** {confidence:.1%}")
    st.markdown(f"*{info.summary}*")
    st.markdown(info.description)

    with st.expander("Indicators (what the IDS looks for)", expanded=intrusion):
        for item in info.indicators:
            st.markdown(f"- {item}")

    with st.expander("Recommended operator actions", expanded=intrusion):
        for i, action in enumerate(info.recommended_actions, 1):
            st.markdown(f"{i}. {action}")


def render_attack_catalog() -> None:
    from scripts.attack_info import all_attack_infos

    for info in all_attack_infos():
        with st.expander(f"{info.title} — {info.severity}"):
            _render_info_static(info)


def _render_info_static(info: AttackInfo) -> None:
    st.markdown(f"**Layer:** {info.layer}")
    st.markdown(info.description)
    st.markdown("**Indicators:**")
    for item in info.indicators:
        st.markdown(f"- {item}")
    st.markdown("**Recommended actions:**")
    for i, action in enumerate(info.recommended_actions, 1):
        st.markdown(f"{i}. {action}")
