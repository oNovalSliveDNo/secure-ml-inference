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
        f"<span class='protocol-badge' style='background:{html.escape(color)}'>{html.escape(label)}</span>",
        unsafe_allow_html=True,
    )


def render_card(title: str, body: str, badge: str | None = None, color: str | None = None) -> None:
    """Render a compact explanatory card."""
    badge_html = ""
    if badge:
        badge_color = color or PALETTE["client"]
        badge_html = (
            f"<span class='protocol-badge' style='background:{html.escape(badge_color)}'>"
            f"{html.escape(badge)}</span>"
        )
    st.markdown(
        f"<div class='protocol-card'>{badge_html}<h4>{html.escape(title)}</h4>"
        f"<p>{html.escape(body)}</p></div>",
        unsafe_allow_html=True,
    )


def render_side_header(title: str, subtitle: str, color: str) -> None:
    """Render a section header for a protocol side."""
    render_badge(title, color)
    st.caption(subtitle)


def render_arrow(label: str = "→") -> None:
    """Render a protocol direction arrow."""
    st.markdown(f"<div class='protocol-arrow'>{html.escape(label)}</div>", unsafe_allow_html=True)


def render_step_statuses(steps: Iterable[str], current_step: int) -> None:
    """Render compact horizontal statuses for all protocol steps."""
    step_titles = list(steps)
    items: list[str] = []
    for idx, title in enumerate(step_titles, start=1):
        if current_step > idx:
            cls, mark = "done", "✓"
        elif current_step == idx:
            cls, mark = "active", str(idx)
        else:
            cls, mark = "todo", str(idx)
        items.append(
            f"<div class='progress-step {cls}'><span>{mark}</span><small>{html.escape(title)}</small></div>"
        )
    st.markdown(
        f"<div class='progress-row'>{'<div class=progress-line></div>'.join(items)}</div>",
        unsafe_allow_html=True,
    )
    current_index = max(min(current_step, 7), 1) - 1
    current_title = step_titles[current_index] if current_step else "ожидание запуска"
    st.markdown(
        "<div class='current-step-badge'>"
        f"Шаг {current_step if current_step else 0} из 7 — "
        f"<strong>{html.escape(current_title)}</strong>"
        "</div>",
        unsafe_allow_html=True,
    )


_STATUS_TEXT: dict[str, str] = {
    "green": "В пределах допуска",
    "yellow": "Требует внимания",
    "red": "За пределами допуска",
}
_STATUS_COLORS: dict[str, str] = {
    "green": PALETTE["server"],
    "yellow": PALETTE["warning"],
    "red": PALETTE["danger"],
}


def render_metric_card(
    title: str,
    value: str,
    delta: str | None,
    status: Literal["green", "yellow", "red", "neutral"],
    explanation: str | None = None,
) -> None:
    """Render an escaped metric card. Neutral cards have no status pill."""
    title_html = html.escape(str(title))
    value_html = html.escape(str(value))
    delta_html = html.escape(str(delta)) if delta is not None else ""
    explanation_html = html.escape(str(explanation)) if explanation is not None else ""
    status_block = ""
    if status != "neutral":
        status_block = (
            f"<span class='metric-card-status' style='background:{html.escape(_STATUS_COLORS[status])}'>"
            f"{html.escape(_STATUS_TEXT[status])}</span>"
        )

    delta_block = f"<div class='metric-card-delta'>{delta_html}</div>" if delta_html else ""
    explanation_block = (
        f"<div class='metric-card-explanation'>{explanation_html}</div>" if explanation_html else ""
    )
    st.markdown(
        f"""
        <div class='metric-card {status}'>
            <div class='metric-card-header'>
                <span class='metric-card-title'>{title_html}</span>{status_block}
            </div>
            <div class='metric-card-value'>{value_html}</div>
            {delta_block}{explanation_block}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_kpi(label: str, value: str) -> None:
    """Render a small inline KPI."""
    st.markdown(
        f"<div class='compact-kpi'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>",
        unsafe_allow_html=True,
    )


def render_status_banner(text: str, status: Literal["green", "yellow", "red"]) -> None:
    """Render one semantic status banner for a whole block."""
    st.markdown(
        f"<div class='status-banner {status}'>{html.escape(text)}</div>", unsafe_allow_html=True
    )


def render_operation_card(
    title: str, formula: str, example: str | None = None, *, example_is_html: bool = False
) -> None:
    """Render current operation with one formula and optional substitution."""
    if example and example_is_html:
        example_html = f"<div class='operation-example'>{example}</div>"
    else:
        example_html = (
            f"<div class='operation-example'>{html.escape(example)}</div>" if example else ""
        )
    st.markdown(
        f"<div class='operation-card'><strong>{html.escape(title)}</strong>"
        f"<code>{html.escape(formula)}</code>{example_html}</div>",
        unsafe_allow_html=True,
    )
