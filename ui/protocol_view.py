"""Protocol and architecture panels for the Streamlit UI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from ui.components import render_arrow, render_badge, render_card, render_side_header
from ui.styles import PALETTE

if TYPE_CHECKING:
    from collections.abc import Sequence


def _preview(value: Any, chars: int = 48) -> str:
    """Return a compact, safe text preview for long protocol values."""
    text = str(value)
    return text if len(text) <= chars else f"{text[:chars]}..."


def _render_badges(labels: Sequence[tuple[str, str]]) -> None:
    """Render a row-like set of protocol badges."""
    for label, color in labels:
        render_badge(label, color)


def _show_if_present(label: str, value: Any, *, code: bool = False) -> None:
    """Show a labeled value only when it is available in the protocol state."""
    if value is None:
        return
    st.caption(label)
    if code:
        st.code(value)
    else:
        st.write(value)


def _scaler_parameters(scaler: Any, feature_names: Sequence[str]) -> pd.DataFrame | None:
    """Build a compact table with scaler parameters when they are available."""
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        return None
    return pd.DataFrame(
        {
            "Признак": feature_names,
            "mean": mean,
            "scale": scale,
        }
    )


def render_protocol_exchange_layout(
    *,
    result: dict[str, Any],
    scenario: dict[str, Any],
    sample: pd.Series,
    current_step: int,
    scale: int,
    scenario_id: str | None = None,
) -> None:
    """Render client, transport, and server-visible protocol data in three columns."""
    client_col, transport_col, server_col = st.columns([5, 1, 5])
    feature_names = [str(name) for name in sample.index]

    with client_col:
        render_side_header("КЛИЕНТ", "Владелец данных и закрытого ключа", PALETTE["client"])
        _render_badges(
            [
                ("открыто на клиенте", PALETTE["client"]),
                ("зашифровано", PALETTE["transport"]),
                ("не отправляется", PALETTE["warning"]),
                ("секрет клиента", PALETTE["warning"]),
            ]
        )
        st.markdown("**Исходные признаки**")
        st.dataframe(
            pd.DataFrame({"Признак": feature_names, "Значение": sample.values}), width="stretch"
        )

        scaler_df = _scaler_parameters(scenario.get("scaler"), feature_names)
        if scaler_df is not None:
            st.markdown("**Scaler параметры**")
            st.dataframe(scaler_df, width="stretch")
        if "x_scaled" in result:
            st.markdown("**Scaled features**")
            st.dataframe(
                pd.DataFrame({"Признак": feature_names, "x_scaled": result["x_scaled"]}),
                width="stretch",
            )
        if "x_int" in result:
            st.markdown("**Fixed-point encoding**")
            st.dataframe(
                pd.DataFrame({"Признак": feature_names, "x_int": result["x_int"]}),
                width="stretch",
            )
        client = result.get("client")
        if client is not None:
            _show_if_present(
                "Public key preview", f"n = {_preview(client.public_key.n)}", code=True
            )
            _show_if_present("Private key", "есть у клиента; значение не отображается")
        if "enc_x" in result:
            st.markdown("**Ciphertext previews**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Признак": feature_names,
                        "Enc(x)": [_preview(value, 36) for value in result["enc_x"]],
                    }
                ),
                width="stretch",
            )
        encrypted_score = result.get("encrypted_score")
        _show_if_present(
            "Encrypted result",
            None if encrypted_score is None else _preview(encrypted_score, 80),
            code=True,
        )
        _show_if_present("Decrypted integer", result.get("score_int"))
        _show_if_present("Decoded prediction", result.get("z_secure"))
        _show_if_present("Postprocessing", scenario.get("postprocess"))

    with transport_col:
        if current_step == 4:
            st.markdown("**Enc(x), public key ───────────────→**")
        elif current_step == 5:
            st.markdown("**Enc(z) ←───────────────**")
        else:
            st.markdown("───────────────")
        if "request_payload" in result:
            st.metric("Payload size", result.get("encrypted_bytes", "—"))
            st.metric("Feature count", result["request_payload"].get("feature_count", "—"))
        else:
            st.metric("Payload size", "—")
            st.metric("Feature count", len(sample))
        st.metric("HTTP status", result.get("status_code", "—"))
        http_elapsed = result.get("http_elapsed_ms")
        st.metric("HTTP roundtrip", "—" if http_elapsed is None else f"{http_elapsed:.1f} мс")

    with server_col:
        render_side_header("СЕРВЕР", "Только данные, доступные серверу", PALETTE["server"])
        visible_scenario_id = scenario_id or result.get("request_payload", {}).get(
            "scenario_id", "—"
        )
        st.write(f"Scenario id: `{visible_scenario_id}`")
        st.write(f"Coefficient count: `{len(scenario.get('w', []))}`")
        st.write(f"Scale: `{scale}`")
        st.write("Encoded weights: есть на сервере; значения не раскрываются в клиентском UI")
        if "request_payload" in result:
            st.markdown("**Ciphertext feature previews**")
            encrypted_features = result["request_payload"].get("encrypted_features", [])
            st.dataframe(
                pd.DataFrame(
                    {"Enc(x) preview": [_preview(value, 36) for value in encrypted_features[:5]]}
                ),
                width="stretch",
            )
        feature_count = result.get("request_payload", {}).get("feature_count", len(sample))
        st.write(
            "Operation counts: "
            f"умножений ciphertext×weight = {feature_count}; сложений = {max(feature_count - 1, 0)}"
        )
        encrypted_score = result.get("encrypted_score")
        _show_if_present(
            "Encrypted bias/result",
            None if encrypted_score is None else _preview(encrypted_score, 80),
            code=True,
        )
        server_compute = result.get("server_compute_ms")
        st.metric(
            "Server compute time", "—" if server_compute is None else f"{server_compute:.1f} мс"
        )


def render_client_transport_server_panels() -> None:
    """Render client, transport, and server panels for the encrypted protocol."""
    client_col, arrow_1, transport_col, arrow_2, server_col = st.columns([3, 1, 3, 1, 3])
    with client_col:
        render_card(
            "Клиент",
            "Масштабирует признаки, кодирует их в целые числа, генерирует ключи, шифрует данные и расшифровывает результат.",
            "CLIENT",
            PALETTE["client"],
        )
    with arrow_1:
        render_arrow("→")
    with transport_col:
        render_card(
            "Транспорт",
            "Передаёт открытый ключ, зашифрованные признаки, масштаб кодирования и служебные метаданные запроса.",
            "HTTPS/API",
            PALETTE["transport"],
        )
    with arrow_2:
        render_arrow("→")
    with server_col:
        render_card(
            "Сервер",
            "Выполняет линейное вычисление модели над шифротекстами и возвращает зашифрованный score.",
            "SERVER",
            PALETTE["server"],
        )


def show_architecture() -> None:
    """Render architecture and threat model without image schemes."""
    st.header("Архитектура и модель угроз")
    render_client_transport_server_panels()
    st.markdown(
        """
### Как работает система

1. **Клиент подготавливает данные.** Пользователь выбирает объект из тестовой выборки. На стороне клиента признаки масштабируются, переводятся в целые числа и шифруются.
2. **Сервер получает только зашифрованные признаки.** Сервер не видит исходные значения признаков. Он получает открытую часть ключа, набор зашифрованных чисел и служебные параметры.
3. **Сервер выполняет расчёт над зашифрованными данными.** Сервер использует заранее обученную модель и возвращает клиенту зашифрованный результат линейного вычисления.
4. **Клиент расшифровывает результат.** Закрытый ключ хранится только у клиента. После расшифровки клиент получает итоговое значение.

### Что защищается

- исходные признаки клиента;
- промежуточный результат расчёта до расшифровки;
- закрытый ключ, который не покидает сторону клиента.

### Что видит сервер

- открытую часть ключа;
- зашифрованные признаки;
- количество признаков;
- выбранный тип задачи;
- служебные параметры кодирования.

### Что сервер не видит

- исходные значения признаков;
- закрытый ключ клиента;
- расшифрованный результат вычисления.

### Модель угроз

В демонстрации рассматривается сервер, который корректно выполняет протокол, но может пытаться извлечь информацию из полученных данных.

### Ограничения демонстрации

- модель сервера не скрывается от самого сервера;
- факт обращения к серверу и технические метаданные не защищаются;
- активные атаки, компрометация клиента и атаки по побочным каналам не рассматриваются;
- прототип предназначен для демонстрации и экспериментальной оценки, а не для промышленного внедрения без дополнительной защиты.
        """
    )
