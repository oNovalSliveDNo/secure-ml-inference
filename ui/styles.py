"""Local styling and color palette for the Streamlit UI."""

from __future__ import annotations

import streamlit as st

PALETTE = {
    "client": "#2563eb",
    "transport": "#7c3aed",
    "server": "#059669",
    "warning": "#d97706",
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
</style>
"""


def apply_styles() -> None:
    """Apply local CSS for protocol cards and badges."""
    st.markdown(LOCAL_CSS, unsafe_allow_html=True)
