"""Mathematical trace rendering for the encrypted inference protocol."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.crypto import serialize_ciphertext
from app.encoding import decode_score, encode_bias, encode_weights


def _preview(value: Any, *, width: int = 48) -> str:
    """Return a short, safe preview for large cryptographic values."""
    text = str(value)
    return text if len(text) <= width else f"{text[:width]}..."


def _ciphertext_to_text(value: Any) -> str:
    """Serialize ciphertext objects while accepting already serialized values."""
    if isinstance(value, str):
        return value
    return serialize_ciphertext(value)


def _as_context(
    trace: dict[str, Any] | None,
    *,
    result: dict[str, Any] | None,
    scenario: dict[str, Any] | None,
    sample: pd.Series | None,
    scale: int | None,
) -> dict[str, Any] | None:
    """Normalize the new trace argument and the legacy keyword arguments."""
    if trace is not None:
        return trace
    if result is None or scenario is None or sample is None or scale is None:
        return None
    return {"result": result, "scenario": scenario, "sample": sample, "scale": scale}


def render_calculation_trace(
    trace: dict[str, Any] | None = None,
    *,
    result: dict[str, Any] | None = None,
    scenario: dict[str, Any] | None = None,
    sample: pd.Series | None = None,
    scale: int | None = None,
) -> None:
    """Render formulas, substitution tables, cryptographic values, and checks."""
    context = _as_context(trace, result=result, scenario=scenario, sample=sample, scale=scale)
    if context is None:
        return

    result = context["result"]
    scenario = context["scenario"]
    sample = context["sample"]
    scale = int(context["scale"])
    if not {"x_scaled", "x_int"}.issubset(result):
        return

    scaler = scenario.get("scaler") or getattr(result.get("client"), "scaler", None)
    if scaler is None or not all(hasattr(scaler, attr) for attr in ("mean_", "scale_")):
        st.warning("Трассировка масштабирования недоступна: StandardScaler не найден.")
        return

    feature_names = list(sample.index)
    x_raw = sample.to_numpy(dtype=float)
    means = np.asarray(scaler.mean_, dtype=float)
    sigmas = np.asarray(scaler.scale_, dtype=float)
    x_scaled = np.asarray(result["x_scaled"], dtype=float)
    x_int = np.asarray(result["x_int"], dtype=np.int64)
    weights = np.asarray(scenario["w"], dtype=float)
    w_int = encode_weights(weights, scale)
    bias = float(scenario["b"])
    b_int = encode_bias(bias, scale)
    scale_squared = scale * scale

    st.subheader("Математическая трассировка расчёта")
    tabs = st.tabs(
        [
            "Масштабирование",
            "Кодирование",
            "Шифрование",
            "Вычисление на сервере",
            "Расшифрование",
            "Контрольная проверка",
        ]
    )

    with tabs[0]:
        st.latex(r"x_i' = \frac{x_i - \mu_i}{\sigma_i}")
        st.dataframe(
            pd.DataFrame(
                {
                    "Признак": feature_names,
                    "Исходное значение": x_raw,
                    "Среднее μ": means,
                    "Масштаб σ": sigmas,
                    "Масштабированное значение": x_scaled,
                }
            ),
            width="stretch",
        )

    with tabs[1]:
        st.latex(r"x_i^{int} = round(x_i' \cdot S)")
        st.latex(r"w_i^{int} = round(w_i \cdot S)")
        st.latex(r"b^{int} = round(b \cdot S^2)")
        scaled_times_s = x_scaled * scale
        st.dataframe(
            pd.DataFrame(
                {
                    "Признак": feature_names,
                    "Масштабированное значение": x_scaled,
                    "S": scale,
                    "x' × S": scaled_times_s,
                    "Целое значение": x_int,
                    "Ошибка округления": x_int - scaled_times_s,
                }
            ),
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Величина": "S", "Значение": scale},
                    {"Величина": "S²", "Значение": scale_squared},
                    {"Величина": "Свободный член b", "Значение": bias},
                    {"Величина": "Закодированный свободный член", "Значение": b_int},
                ]
            ),
            width="stretch",
        )

    ciphertext_texts = [_ciphertext_to_text(value) for value in result.get("enc_x", [])]
    with tabs[2]:
        st.latex(r"c_i = Enc_{pk}(x_i^{int})")
        if ciphertext_texts:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Признак": feature_names,
                        "Целое значение": x_int,
                        "Зашифрованное значение": [_preview(value) for value in ciphertext_texts],
                        "Размер шифротекста, байт": [
                            len(value.encode("utf-8")) for value in ciphertext_texts
                        ],
                    }
                ),
                width="stretch",
            )
            with st.expander("Полные криптографические значения", expanded=False):
                st.warning(
                    "Закрытый ключ не выводится. Ниже показаны только шифротексты и открытый ключ."
                )
                public_key = getattr(result.get("client"), "public_key", None)
                if public_key is not None:
                    st.code(f"Открытый ключ n = {_preview(public_key.n, width=96)}")
                for feature, ciphertext in zip(feature_names, ciphertext_texts, strict=True):
                    st.text_area(str(feature), ciphertext, height=80)
        else:
            st.info("Шифротексты появятся после шага шифрования.")

    with tabs[3]:
        st.latex(r"Enc(z_{int}) = Enc(b_{int}) + \sum_i w_i^{int} \cdot Enc(x_i^{int})")
        st.warning(
            "Сервер получает только шифротексты и веса модели; открытые признаки здесь приведены только для аудита UI."
        )
        if ciphertext_texts:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Зашифрованное значение": [_preview(value) for value in ciphertext_texts],
                        "Закодированный вес": w_int,
                        "Операция": [
                            f"прибавить w_int[{i}] × Enc(x_int[{i}]) к аккумулятору"
                            for i in range(len(w_int))
                        ],
                        "Статус": "гомоморфный член добавлен",
                    }
                ),
                width="stretch",
            )

    with tabs[4]:
        if "score_int" in result:
            z_int = int(result["score_int"])
            decoded_z = decode_score(z_int, scale)
            st.latex(r"z_{int} = Dec_{sk}(Enc(z_{int}))")
            st.latex(r"z = \frac{z_{int}}{S^2}")
            st.write(
                f"Подстановка: z_int = {z_int}; S² = {scale_squared}; z = {z_int} / {scale_squared} = {decoded_z:.12g}"
            )
        else:
            st.info("Расшифрованное значение появится после шага расшифрования.")

    with tabs[5]:
        st.warning(
            "Контрольный открытый расчёт нужен только для проверки корректности. Сервер не получает открытые x_int."
        )
        products = x_int * w_int
        control_z_int = int(np.sum(products) + b_int)
        control_z = decode_score(control_z_int, scale)
        st.dataframe(
            pd.DataFrame(
                {
                    "Признак": feature_names,
                    "Целое значение": x_int,
                    "Закодированный вес": w_int,
                    "Произведение": products,
                }
            ),
            width="stretch",
        )
        decrypted_score_int = result.get("score_int")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Величина": "Сумма произведений", "Значение": int(np.sum(products))},
                    {"Величина": "Закодированный свободный член", "Значение": b_int},
                    {"Величина": "Итоговое целое значение", "Значение": control_z_int},
                    {"Величина": "Итоговый прогноз", "Значение": control_z},
                    {"Величина": "Расшифрованное PHE-значение", "Значение": decrypted_score_int},
                    {
                        "Величина": "Совпадает с расшифрованным результатом",
                        "Значение": decrypted_score_int == control_z_int,
                    },
                ]
            ),
            width="stretch",
        )
