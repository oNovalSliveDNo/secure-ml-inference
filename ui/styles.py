"""Local styling and color palette for the Streamlit UI."""

from __future__ import annotations

import streamlit as st

PALETTE = {
    "client": "#22A06B",
    "client_bg": "#fff4f0",
    "transport": "#64748b",
    "transport_bg": "#f4f8fc",
    "server": "#F97360",
    "server_bg": "#effcf5",
    "warning": "#d97706",
    "danger": "#dc2626",
    "surface": "#f8fafc",
    "border": "#cbd5e1",
    "text": "#0f172a",
    "blue": "#2563eb",
}

LOCAL_CSS = f"""
<style>
.protocol-card, .metric-card, .operation-card {{
    border: 1px solid {PALETTE["border"]}; border-radius: .75rem; padding: .75rem;
    background: white; min-height: unset;
}}
.protocol-badge {{ display: inline-block; padding: .15rem .5rem; border-radius: 999px; color: white; font-size: .72rem; font-weight: 700; margin-bottom: .25rem; }}
.protocol-arrow {{ text-align: center; font-size: 1.6rem; line-height: 1; color: {PALETTE["transport"]}; padding-top: 1.5rem; }}
.progress-row {{ display:flex; align-items:center; gap:.35rem; margin:.4rem 0 .2rem; }}
.progress-line {{ flex: .35; height: 2px; background: #94a3b8; }}
.progress-step {{ display:flex; flex-direction:column; align-items:center; gap:.15rem; min-width:5.5rem; }}
.progress-step span {{ display:grid; place-items:center; width:1.45rem; height:1.45rem; border-radius:50%; background:#e2e8f0; color:#475569; font-weight:800; font-size:.78rem; }}
.progress-step small {{ font-size:.8rem; color:#334155; text-align:center; }}
.progress-step.active small {{ font-weight:700; }}
.current-step-badge {{ display:inline-block; margin:.35rem 0 .2rem; padding:.35rem .6rem; border:1px solid #bfdbfe; border-radius:.6rem; background:#eff6ff; color:{PALETTE["text"]}; font-size:.88rem; }}
.progress-step.done span {{ background:{PALETTE["server"]}; color:white; }}
.progress-step.active span {{ background:{PALETTE["blue"]}; color:white; box-shadow:0 0 0 3px #dbeafe; }}
div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stVerticalBlock"][data-st-key="client_zone"]) {{ background:{PALETTE["client_bg"]}; border-color:{PALETTE["client"]}; }}
div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stVerticalBlock"][data-st-key="channel_zone"]) {{ background:{PALETTE["transport_bg"]}; border-color:#94a3b8; }}
div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stVerticalBlock"][data-st-key="server_zone"]) {{ background:{PALETTE["server_bg"]}; border-color:{PALETTE["server"]}; }}
.active-zone {{ border:2px solid currentColor; box-shadow:0 8px 20px rgba(15,23,42,.10); border-radius:.75rem; padding:.35rem; margin-bottom:.35rem; font-weight:700; }}
.active-zone-client {{ background:#ecfdf5; color:{PALETTE["client"]}; box-shadow:0 8px 20px rgba(249,115,96,.14); }}
.active-zone-channel {{ background:#eff6ff; color:{PALETTE["blue"]}; box-shadow:0 8px 20px rgba(37,99,235,.14); }}
.active-zone-server {{ background:#fff1ed; color:{PALETTE["server"]}; box-shadow:0 8px 20px rgba(34,160,107,.14); }}
.channel-arrow {{ text-align:center; font-weight:800; font-size:1.05rem; margin:.4rem 0; color:{PALETTE["text"]}; white-space:nowrap; }}
.metric-card-header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:.5rem; margin-bottom:.45rem; }}
.metric-card-title {{ color:{PALETTE["text"]}; font-size:.82rem; font-weight:700; }}
.metric-card-status {{ border-radius:999px; color:white; font-size:.66rem; font-weight:700; padding:.12rem .42rem; }}
.metric-card-value {{ color:{PALETTE["text"]}; font-size:1.25rem; font-weight:800; line-height:1.15; }}
.metric-card-delta, .metric-card-explanation {{ color:#334155; font-size:.8rem; margin-top:.3rem; }}
.compact-kpi {{ display:flex; justify-content:space-between; gap:.5rem; padding:.25rem 0; border-bottom:1px solid #e2e8f0; font-size:.9rem; }}
.status-banner {{ border-radius:.65rem; padding:.55rem .7rem; font-weight:700; margin:.5rem 0; }}
.status-banner.green {{ background:#e6f4ea; color:#137333; }}
.status-banner.yellow {{ background:#fef7e0; color:#8a5a00; }}
.status-banner.red {{ background:#fce8e6; color:#a50e0e; }}
.operation-card code {{ display:block; margin:.35rem 0; background:#f1f5f9; padding:.35rem; border-radius:.35rem; }}
.operation-example {{ font-size:.86rem; color:#334155; }}
.operation-example-feature {{ font-weight:700; color:{PALETTE["text"]}; }}
.operation-example-technical {{ font-size:.74rem; color:#64748b; margin:.1rem 0 .25rem; }}

[data-testid="stToolbar"] {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
.important-number {{ font-size:1.35rem; font-weight:850; color:#0f172a; line-height:1.15; }}
.zone-title {{ display:block; border-top:4px solid currentColor; padding-top:.3rem; font-weight:850; letter-spacing:.02em; }}
.zone-title-client {{ color:#22A06B; }}
.zone-title-channel {{ color:#2563eb; text-align:center; }}
.zone-title-server {{ color:#F97360; }}
</style>
"""


def apply_styles() -> None:
    """Apply local CSS for protocol cards and badges."""
    st.markdown(LOCAL_CSS, unsafe_allow_html=True)
