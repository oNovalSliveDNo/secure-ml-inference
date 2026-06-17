"""Local styling and color palette for the Streamlit UI."""

from __future__ import annotations

import streamlit as st

PALETTE = {
    "client": "#2563eb",
    "transport": "#7c3aed",
    "server": "#059669",
    "warning": "#d97706",
    "danger": "#dc2626",
    "surface": "#f8fafc",
    "border": "#dbeafe",
    "text": "#0f172a",
}

LOCAL_CSS = f"""
<style>
.protocol-card {{
    border: 1px solid {PALETTE["border"]};
    border-radius: 0.75rem;
    padding: 1rem;
    background: {PALETTE["surface"]};
    min-height: 8rem;
}}
.protocol-badge {{
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    color: white;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}}
.protocol-arrow {{
    text-align: center;
    font-size: 2rem;
    line-height: 1;
    color: {PALETTE["transport"]};
    padding-top: 2.2rem;
}}
.step-status {{ margin: 0.15rem 0; }}
.metric-card {{
    border: 1px solid {PALETTE["border"]};
    border-radius: 0.75rem;
    padding: 1rem;
    background: {PALETTE["surface"]};
    min-height: 7rem;
}}
.metric-card-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}}
.metric-card-title {{
    color: {PALETTE["text"]};
    font-size: 0.85rem;
    font-weight: 700;
}}
.metric-card-status {{
    border-radius: 999px;
    color: white;
    display: inline-block;
    flex-shrink: 0;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
}}
.metric-card-value {{
    color: {PALETTE["text"]};
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.15;
}}
.metric-card-delta {{
    color: {PALETTE["transport"]};
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: 0.35rem;
}}
.metric-card-explanation {{
    color: {PALETTE["text"]};
    font-size: 0.82rem;
    margin-top: 0.5rem;
    opacity: 0.8;
}}
</style>
"""


def apply_styles() -> None:
    """Apply local CSS for protocol cards and badges."""
    st.markdown(LOCAL_CSS, unsafe_allow_html=True)
