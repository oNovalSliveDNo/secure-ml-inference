"""Reusable Streamlit components for the secure inference demo."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

import streamlit as st

from ui.styles import PALETTE


def render_badge(label: str, color: str) -> None:
    """Render a colored pill badge."""
    st.markdown(
        f"<span class='protocol-badge' style='background:{color}'>{label}</span>",
        unsafe_allow_html=True,
    )


def render_card(title: str, body: str, badge: str | None = None, color: str | None = None) -> None:
    """Render a compact explanatory card."""
    badge_html = ""
    if badge:
        badge_color = color or PALETTE["client"]
        badge_html = f"<span class='protocol-badge' style='background:{badge_color}'>{badge}</span>"
    st.markdown(
        f"<div class='protocol-card'>{badge_html}<h4>{title}</h4><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def render_side_header(title: str, subtitle: str, color: str) -> None:
    """Render a section header for a protocol side."""
    render_badge(title, color)
    st.caption(subtitle)


def render_arrow(label: str = "→") -> None:
    """Render a protocol direction arrow."""
    st.markdown(f"<div class='protocol-arrow'>{label}</div>", unsafe_allow_html=True)


def render_step_statuses(steps: Iterable[str], current_step: int) -> None:
    """Render wizard statuses for all protocol steps."""
    for idx, title in enumerate(steps, start=1):
        marker = "🟢" if current_step == idx else ("✅" if current_step > idx else "⚪")
        st.markdown(
            f"<div class='step-status'><strong>{marker} {title}</strong></div>",
            unsafe_allow_html=True,
        )


def render_metric_card(label: str, value: Any, delta: str | None = None) -> None:
    """Render a metric card using Streamlit's native metric component."""
    st.metric(label, value, delta=delta)
