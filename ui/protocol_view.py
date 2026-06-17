"""Protocol and architecture panels for the Streamlit UI."""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.config import KEY_LENGTH
from app.crypto import serialize_ciphertext
from ui.components import render_arrow, render_card, render_compact_kpi, render_operation_card
from ui.metrics_helpers import format_class_label
from ui.model_labels import get_model_label, get_model_technical_name
from ui.styles import PALETTE

FEATURE_LABELS: dict[str, str] = {
    # Breast Cancer Wisconsin features
    "mean radius": "средний радиус",
    "mean texture": "средняя текстура",
    "mean perimeter": "средний периметр",
    "mean area": "средняя площадь",
    "mean smoothness": "средняя гладкость",
    "mean compactness": "средняя компактность",
    "mean concavity": "средняя вогнутость",
    "mean concave points": "среднее число вогнутых точек",
    "mean symmetry": "средняя симметрия",
    "mean fractal dimension": "средняя фрактальная размерность",
    "radius error": "ошибка радиуса",
    "texture error": "ошибка текстуры",
    "worst radius": "наибольший радиус",
    "worst texture": "наибольшая текстура",
    # Diabetes features from sklearn.datasets.load_diabetes
    "age": "возраст пациента",
    "sex": "пол пациента",
    "bmi": "индекс массы тела",
    "bp": "среднее артериальное давление",
    "s1": "общий холестерин",
    "s2": "липопротеины низкой плотности",
    "s3": "липопротеины высокой плотности",
    "s4": "отношение общего холестерина к ЛПВП",
    "s5": "логарифм уровня триглицеридов",
    "s6": "уровень глюкозы в крови",
}


def _feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _preview(value: Any, chars: int = 48) -> str:
    text = str(value)
    return text if len(text) <= chars else f"{text[:chars]}..."


def _ciphertext_preview(value: Any, chars: int = 64) -> str:
    if callable(getattr(value, "ciphertext", None)):
        try:
            text = serialize_ciphertext(value)
        except (AttributeError, TypeError, ValueError):
            text = str(value.ciphertext())
    else:
        text = str(value)

    if len(text) <= chars:
        return text
    if chars <= 3:
        return text[:chars]

    edge_chars = (chars - 3) // 2
    leading_chars = edge_chars + ((chars - 3) % 2)
    return f"{text[:leading_chars]}...{text[-edge_chars:]}"


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}".replace(",", " ").replace(".", ",")
    except (TypeError, ValueError):
        return "—"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def _fmt_bytes(value: Any) -> str:
    try:
        byte_count = int(value)
    except (TypeError, ValueError):
        return "—"
    abs_count = abs(byte_count)
    if abs_count % 10 == 1 and abs_count % 100 != 11:
        unit = "байт"
    elif 2 <= abs_count % 10 <= 4 and not 12 <= abs_count % 100 <= 14:
        unit = "байта"
    else:
        unit = "байт"
    return f"{byte_count:,}".replace(",", " ") + f" {unit}"


def _feature_example(
    sample: pd.Series, scenario: dict[str, Any], result: dict[str, Any]
) -> tuple[str, str, str, str, str, str] | None:
    scaler = scenario.get("scaler")
    if scaler is None or not all(hasattr(scaler, attr) for attr in ("mean_", "scale_")):
        return None
    idx = 0
    name = str(sample.index[idx])
    raw = float(sample.iloc[idx])
    mean = float(scaler.mean_[idx])
    sigma = float(scaler.scale_[idx])
    scaled = float(result.get("x_scaled", np.array([(raw - mean) / sigma]))[idx])
    return name, _feature_label(name), _fmt(raw), _fmt(mean), _fmt(sigma), _fmt(scaled)


def _active(step: int, zone: str) -> bool:
    return (
        (zone == "client" and step in {1, 2, 3, 6})
        or (zone == "channel" and step in {4, 5})
        or (zone == "server" and step == 5)
    )


ACTIVE_NOTES = {
    (1, "client"): "Масштабирование выполняется на клиенте",
    (2, "client"): "Кодирование выполняется на клиенте",
    (3, "client"): "Клиент шифрует признаки",
    (4, "channel"): "Зашифрованный запрос передаётся серверу",
    (5, "server"): "Сервер выполняет гомоморфное вычисление",
    (5, "channel"): "Зашифрованный результат возвращается клиенту",
    (6, "client"): "Клиент расшифровывает результат",
}


def _active_note(step: int, zone: str) -> None:
    note = ACTIVE_NOTES.get((step, zone))
    if note:
        st.markdown(
            f"<div class='active-zone active-zone-{zone}'>{note}</div>",
            unsafe_allow_html=True,
        )


def build_server_panel_data(
    *,
    result: dict[str, Any],
    scenario: dict[str, Any],
    sample: pd.Series,
    scale: int,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Prepare only server-visible protocol data for the server panel."""
    request_payload = dict(result.get("request_payload", {}))
    feature_count = int(request_payload.get("feature_count", len(sample)))
    encrypted_features = request_payload.get("encrypted_features", [])
    encrypted_score = result.get("encrypted_score")
    return {
        "scenario_id": scenario_id or request_payload.get("scenario_id", "—"),
        "coefficient_count": len(scenario.get("w", [])),
        "scale": scale,
        "request_payload": request_payload,
        "encrypted_feature_previews": [
            _ciphertext_preview(value, 36) for value in encrypted_features[:5]
        ],
        "feature_count": feature_count,
        "multiplication_count": feature_count,
        "addition_count": feature_count,
        "encrypted_score_preview": None
        if encrypted_score is None
        else _ciphertext_preview(encrypted_score, 80),
        "server_compute_ms": result.get("server_compute_ms"),
    }


def render_protocol_exchange_layout(
    *,
    result: dict[str, Any],
    scenario: dict[str, Any],
    sample: pd.Series,
    current_step: int,
    scale: int,
    scenario_id: str | None = None,
    detailed: bool = False,
) -> None:
    """Render compact client, channel, and server protocol zones."""
    client_col, channel_col, server_col = st.columns([4.4, 1.6, 4.4])
    feature_names = [str(name) for name in sample.index]

    server_data = build_server_panel_data(
        result=result, scenario=scenario, sample=sample, scale=scale, scenario_id=scenario_id
    )

    with client_col, st.container(key="client_zone", border=True):
        _active_note(current_step, "client")
        st.markdown(
            "<span class='zone-title zone-title-client'>🔐 КЛИЕНТ</span>", unsafe_allow_html=True
        )
        st.caption("Владелец данных и закрытого ключа")
        render_compact_kpi("Выбран объект", str(result.get("human_sample_idx", "—")))
        render_compact_kpi("Количество признаков", str(len(sample)))
        st.write("✓ Исходные данные локальны  ")
        st.write("✓ Закрытый ключ не передаётся")
        ex = _feature_example(sample, scenario, result)

        def render_scaling_step(*, show_table: bool) -> None:
            if not ex:
                return
            technical_name, readable_label, raw, mean, sigma, scaled = ex
            example_html = (
                f"<div class='operation-example-feature'>Признак: {html.escape(readable_label)}</div>"
                f"<div class='operation-example-technical'>{html.escape(technical_name)}</div>"
                f"<div>({raw} − {mean}) / {sigma} = {scaled}</div>"
            )
            render_operation_card(
                "Масштабирование признаков",
                "x' = (x − μ) / σ",
                example_html,
                example_is_html=True,
            )
            st.write(f"Обработано признаков: {len(sample)}")
            if show_table:
                with st.expander("Технические подробности масштабирования", expanded=detailed):
                    scaler = scenario["scaler"]
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Признак": feature_names,
                                "Исходное значение": sample.values,
                                "Среднее μ": scaler.mean_,
                                "Масштаб σ": scaler.scale_,
                                "Масштабированное значение": result.get("x_scaled", []),
                            }
                        ),
                        width="stretch",
                    )

        def render_encoding_step(*, show_table: bool) -> None:
            if "x_int" not in result:
                return
            value = int(result["x_int"][0])
            scaled_value = float(result["x_scaled"][0])
            render_operation_card(
                "Кодирование с фиксированной точкой",
                "x_int = round(x' × S)",
                f"{_fmt(scaled_value)} × {_fmt_int(scale)} → {_fmt_int(value)}",
            )
            st.write(f"Масштаб S: {_fmt_int(scale)}")
            st.write(f"Закодировано значений: {len(result['x_int'])}")
            if show_table:
                with st.expander("Технические подробности кодирования", expanded=detailed):
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Признак": feature_names,
                                "Масштабированное значение": result["x_scaled"],
                                "Целое значение": result["x_int"],
                            }
                        ),
                        width="stretch",
                    )

        def render_encryption_step(*, show_table: bool) -> None:
            if "enc_x" not in result:
                return
            render_operation_card("Шифрование Paillier", "c_i = Enc_pk(x_int)")
            st.write(f"Длина ключа: {KEY_LENGTH} бит")
            st.write(f"Зашифровано признаков: {len(result['enc_x'])}")
            st.write("Закрытый ключ остаётся у клиента")
            st.write("Пример зашифрованного значения")
            st.code(_ciphertext_preview(result["enc_x"][0], 64))
            if show_table:
                with st.expander("Технические подробности шифрования", expanded=detailed):
                    client = result.get("client")
                    if client is not None:
                        st.code(f"Открытый ключ n = {_preview(client.public_key.n, 120)}")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Признак": feature_names,
                                "Пример зашифрованного значения": [
                                    _ciphertext_preview(v, 80) for v in result["enc_x"]
                                ],
                            }
                        ),
                        width="stretch",
                    )

        def render_decryption_step() -> None:
            if "z_secure" not in result:
                return
            render_operation_card("Результат получен и расшифрован", "z = Dec_sk(Enc(z_int)) / S²")
            st.write(f"Расшифрованное целое значение: `{_fmt_int(result['score_int'])}`")
            st.write(f"Итоговый прогноз: `{_fmt(result['z_secure'], 6)}`")
            if scenario_id == "classification":
                st.write(f"Вероятность класса: `{_fmt(result.get('prob_secure'), 4)}`")
                st.write(f"Предсказанный класс: `{format_class_label(result.get('pred_secure'))}`")

        completed_client_steps = [
            (1, "✓ Масштабирование", render_scaling_step),
            (2, "✓ Кодирование", render_encoding_step),
            (3, "✓ Шифрование", render_encryption_step),
        ]
        completed_statuses = [
            label for step, label, _ in completed_client_steps if current_step > step
        ]
        if completed_statuses:
            st.caption(" · ".join(completed_statuses))
            with st.expander("Ранее выполненные действия клиента", expanded=False):
                for step, _, render_step in completed_client_steps:
                    if current_step > step:
                        render_step(show_table=detailed)

        if current_step == 1:
            render_scaling_step(show_table=detailed)
        elif current_step == 2:
            render_encoding_step(show_table=detailed)
        elif current_step == 3:
            render_encryption_step(show_table=detailed)
        elif current_step == 6:
            render_decryption_step()

    with channel_col, st.container(key="channel_zone", border=True):
        _active_note(current_step, "channel")
        st.markdown(
            "<span class='zone-title zone-title-channel'>⇄ КАНАЛ ПЕРЕДАЧИ</span>",
            unsafe_allow_html=True,
        )
        request_size = result.get("encrypted_bytes")
        status_code = result.get("status_code")
        if current_step < 4:
            st.caption("Ожидание")
            st.caption("Запрос ещё не сформирован")
        elif current_step == 4:
            st.markdown("<div class='channel-arrow'>КЛИЕНТ → СЕРВЕР</div>", unsafe_allow_html=True)
            st.write("Enc(x)")
            st.write(f"{len(sample)} значений")
            st.caption(f"Размер запроса: {_fmt_bytes(request_size)}")
            with st.expander("Технические подробности", expanded=detailed):
                st.write("Открытый ключ передаётся вместе с запросом")
                st.write("Служебные параметры включены в payload")
                st.write(f"HTTP: {status_code if status_code is not None else '—'}")
        else:
            st.markdown("<div class='channel-arrow'>КЛИЕНТ ← СЕРВЕР</div>", unsafe_allow_html=True)
            st.write("Enc(z)")
            st.write("1 значение")
            if status_code == 200:
                st.caption("Запрос успешно обработан")
            else:
                st.caption(f"HTTP: {status_code if status_code is not None else '—'}")
            with st.expander("Технические подробности", expanded=detailed):
                st.write(f"Размер запроса: {_fmt_bytes(request_size)}")
                st.write("Ответ содержит один зашифрованный результат")
                st.write(f"HTTP: {status_code if status_code is not None else '—'}")

    with server_col, st.container(key="server_zone", border=True):
        _active_note(current_step, "server")
        st.markdown(
            "<span class='zone-title zone-title-server'>⚙ СЕРВЕР</span>", unsafe_allow_html=True
        )
        st.caption("Владелец модели")
        st.write(f"Сценарий: {'классификация' if scenario_id == 'classification' else 'регрессия'}")
        st.write(f"Модель: {get_model_label(scenario_id)}")
        st.caption(get_model_technical_name(scenario_id))
        st.write(f"Коэффициентов: {server_data['coefficient_count']}")
        st.write(f"Масштаб кодирования: {_fmt_int(scale)}")
        if current_step >= 4:
            st.write(f"Получено зашифрованных признаков: {server_data['feature_count']}")
            st.write("Открытые значения признаков: недоступны")
            st.write("Закрытый ключ: недоступен")
        if current_step >= 5:
            render_operation_card(
                "Гомоморфное вычисление", "Enc(z_int) = Enc(b_int) + Σ w_int · Enc(x_int)"
            )
            st.write(f"Умножений шифротекста на коэффициент: {server_data['multiplication_count']}")
            st.write(f"Сложений с аккумулятором: {server_data['addition_count']}")
            st.write(f"Время вычисления: {_fmt(server_data['server_compute_ms'])} мс")
            with st.expander("Подробности серверного вычисления", expanded=detailed):
                st.dataframe(
                    pd.DataFrame(
                        {"Зашифрованный признак": server_data["encrypted_feature_previews"]}
                    ),
                    width="stretch",
                )
                if server_data["encrypted_score_preview"]:
                    st.code(f"Зашифрованный результат: {server_data['encrypted_score_preview']}")


def render_client_transport_server_panels() -> None:
    """Render client, transport, and server panels for the encrypted protocol."""
    client_col, arrow_1, transport_col, arrow_2, server_col = st.columns([3, 1, 3, 1, 3])
    with client_col:
        render_card(
            "Клиент",
            "Масштабирует, кодирует, шифрует признаки и расшифровывает результат.",
            "CLIENT",
            PALETTE["client"],
        )
    with arrow_1:
        render_arrow("→")
    with transport_col:
        render_card(
            "Транспорт",
            "Передаёт открытый ключ, зашифрованные признаки и служебные параметры.",
            "HTTPS/API",
            PALETTE["transport"],
        )
    with arrow_2:
        render_arrow("→")
    with server_col:
        render_card(
            "Сервер",
            "Вычисляет линейную модель над шифротекстами и возвращает Enc(z).",
            "SERVER",
            PALETTE["server"],
        )


def show_architecture() -> None:
    """Render architecture and threat model without image schemes."""
    st.header("Архитектура и модель угроз")
    render_client_transport_server_panels()
    st.markdown("""
### Как работает система
1. **Клиент подготавливает данные.** Признаки масштабируются, кодируются и шифруются локально.
2. **Сервер получает только зашифрованные признаки.** Исходные значения и закрытый ключ недоступны серверу.
3. **Сервер выполняет расчёт над зашифрованными данными.** Используются веса заранее обученной модели.
4. **Клиент расшифровывает результат.** Итоговый прогноз получает только клиент.
""")
