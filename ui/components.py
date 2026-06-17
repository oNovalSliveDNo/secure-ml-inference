"""Reusable Streamlit components for the secure inference demo."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Literal

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


_METRIC_STATUS_TEXT: dict[str, str] = {
    "green": "В пределах допуска",
    "yellow": "Требует внимания",
    "red": "За пределами допуска",
    "neutral": "Информационный показатель",
}

_METRIC_STATUS_COLORS: dict[str, str] = {
    "green": PALETTE["server"],
    "yellow": PALETTE["warning"],
    "red": PALETTE["danger"],
    "neutral": PALETTE["transport"],
}


def render_metric_card(
    title: str,
    value: str,
    delta: str | None,
    status: Literal["green", "yellow", "red", "neutral"],
    explanation: str | None = None,
) -> None:
    """Render an escaped metric card with a status label."""
    status_text = _METRIC_STATUS_TEXT[status]
    status_color = _METRIC_STATUS_COLORS[status]
    title_html = html.escape(str(title))
    value_html = html.escape(str(value))
    delta_html = html.escape(str(delta)) if delta is not None else ""
    explanation_html = html.escape(str(explanation)) if explanation is not None else ""
    status_text_html = html.escape(status_text)
    status_color_html = html.escape(status_color)

    delta_block = f"<div class='metric-card-delta'>{delta_html}</div>" if delta_html else ""
    explanation_block = (
        f"<div class='metric-card-explanation'>{explanation_html}</div>" if explanation_html else ""
    )
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-card-header'>
                <span class='metric-card-title'>{title_html}</span>
                <span class='metric-card-status' style='background:{status_color_html}'>{status_text_html}</span>
            </div>
            <div class='metric-card-value'>{value_html}</div>
            {delta_block}
            {explanation_block}
        </div>
        """,
        unsafe_allow_html=True,
    )
