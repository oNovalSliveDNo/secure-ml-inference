"""Mathematical trace rendering for the encrypted inference protocol."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.encoding import encoded_plaintext_score


def render_calculation_trace(
    *, result: dict[str, Any], scenario: dict[str, Any], sample: pd.Series, scale: int
) -> None:
    """Render formulas, substitution tables, and an open control calculation."""
    if not {"x_scaled", "x_int"}.issubset(result):
        return

    st.subheader("Математика протокола")
    st.markdown(
        """
**Формулы.** Клиент масштабирует признаки `x`, кодирует их как `Xᵢ = round(scale · xᵢ)`,
сервер вычисляет линейную сумму над зашифрованными значениями, а клиент декодирует результат
как `z = score_int / scale²`.
        """
    )

    weights = np.asarray(scenario["w"], dtype=float)
    bias = float(scenario["b"])
    x_scaled = np.asarray(result["x_scaled"], dtype=float)
    x_int = np.asarray(result["x_int"], dtype=int)
    encoded_score = encoded_plaintext_score(x=x_scaled, w=weights, b=bias, scale=scale)

    substitution_df = pd.DataFrame(
        {
            "Признак": sample.index,
            "x после масштабирования": x_scaled,
            "X = round(scale · x)": x_int,
            "w": weights,
            "w · x": weights * x_scaled,
        }
    )
    st.markdown("**Таблица подстановок для выбранного объекта**")
    st.dataframe(substitution_df, width="stretch")

    control_df = pd.DataFrame(
        [
            {"Величина": "Σ wᵢxᵢ", "Значение": float(np.dot(weights, x_scaled))},
            {"Величина": "b", "Значение": bias},
            {
                "Величина": "Открытый контрольный z",
                "Значение": float(np.dot(weights, x_scaled) + bias),
            },
            {"Величина": "Открытый расчёт после кодирования", "Значение": float(encoded_score)},
            {"Величина": "Защищённый z после расшифровки", "Значение": result.get("z_secure")},
        ]
    )
    st.markdown("**Контрольный открытый расчёт**")
    st.dataframe(control_df, width="stretch")
