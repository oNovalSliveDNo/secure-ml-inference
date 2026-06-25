"""Static checks for compact Russian Streamlit demo UI."""

from __future__ import annotations

from pathlib import Path

from ui.protocol_view import build_server_panel_data

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_start_button_and_removed_legacy_labels() -> None:
    app = read("ui/streamlit_app.py")
    assert "Начать демонстрацию" in app
    assert "Запустить демонстрацию" not in app
    assert "Начать заново" not in app
    assert "Выполнить до конца" not in app
    assert "Сбросить" in app


def test_first_click_executes_scaling_and_human_sample_numbering() -> None:
    app = read("ui/streamlit_app.py")
    step0 = app.index("if step == 0:")
    step1 = app.index("if step == 1:")
    assert "client.preprocess" in app[step0:step1]
    assert 'wizard_state["step"] = 1' in app[step0:step1]
    assert "Объект {sample_idx + 1} из {len(x_test)}" in app


def test_server_addition_count_equals_feature_count() -> None:
    data = build_server_panel_data(
        result={"request_payload": {"feature_count": 30, "encrypted_features": []}},
        scenario={"w": [0] * 30},
        sample=[0] * 30,
        scale=10_000,
        scenario_id="classification",
    )
    assert data["addition_count"] == 30


def test_detailed_trace_is_closed_by_default_and_duplicate_step_expanders_removed() -> None:
    app = read("ui/streamlit_app.py")
    assert '"Технические расчёты и промежуточные значения", expanded=False' in app
    assert 'with st.expander("Шаг 1. Масштабирование' not in app
    assert "Исходные признаки (видны только клиенту)" not in app


def test_neutral_metric_card_has_no_information_badge() -> None:
    components = read("ui/components.py")
    assert "Информационный показатель" not in components
    assert 'if status != "neutral"' in components


def test_required_english_labels_absent_from_ui() -> None:
    combined = "\n".join(
        read(path)
        for path in [
            "ui/streamlit_app.py",
            "ui/protocol_view.py",
            "ui/metrics_view.py",
            "ui/calculation_trace.py",
        ]
    )
    forbidden = [
        "Payload size",
        "Feature count",
        "HTTP status",
        "HTTP roundtrip",
        "Scenario id",
        "Coefficient count",
        "Encoded weights",
        "Operation counts",
        "Server compute time",
        "Sample-level метрики",
        "Baseline prediction",
        "Median abs error",
        "p90 abs error",
        "Encoded plaintext",
        "PHE prediction",
        "Tolerance",
        "Overhead ratio",
        "Plaintext request size",
        "Encrypted request size",
        "Encryption time",
        "Decryption time",
        "Aggregate regression panel",
        "Private key",
        "Public key preview",
    ]
    for label in forbidden:
        assert label not in combined


def test_payload_construction_contains_no_plaintext_features() -> None:
    app = read("ui/streamlit_app.py")
    payload_start = app.index("payload = {")
    payload_end = app.index("}", payload_start)
    payload_block = app[payload_start:payload_end]
    assert "encrypted_features" in payload_block
    assert "sample" not in payload_block
    assert "x_scaled" not in payload_block
    assert "x_int" not in payload_block
