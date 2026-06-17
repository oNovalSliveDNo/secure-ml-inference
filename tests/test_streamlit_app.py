"""Smoke and state-reset tests for the Streamlit UI."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from ui.ui_models import ProtocolState

APP_FILE = "ui/streamlit_app.py"


def _run_app() -> AppTest:
    app = AppTest.from_file(APP_FILE, default_timeout=30)
    app.run()
    assert not app.exception
    return app


def test_classification_scenario_renders_without_exception() -> None:
    app = _run_app()

    assert app.radio[0].value == "classification"
    assert app.session_state["demo_state"]["scenario_id"] == "classification"


def test_regression_scenario_renders_without_exception() -> None:
    app = _run_app()

    app.radio[0].set_value("regression").run()

    assert not app.exception
    assert app.radio[0].value == "regression"
    assert app.session_state["demo_state"]["scenario_id"] == "regression"


def test_sample_index_change_resets_protocol_state() -> None:
    app = _run_app()
    app.session_state["demo_state"] = ProtocolState(
        scenario_id="classification",
        sample_idx=0,
        step=5,
        result={"encrypted_score": "ciphertext", "x_int": [1, 2, 3]},
    ).to_session_dict()

    app.slider[0].set_value(1).run()

    assert not app.exception
    expected = ProtocolState(
        scenario_id="classification",
        sample_idx=1,
    ).to_session_dict()
    expected["result"]["human_sample_idx"] = 2
    assert app.session_state["demo_state"] == expected
    assert "protocol_trace" not in app.session_state
    assert "protocol_result" not in app.session_state


def test_scenario_change_resets_protocol_state() -> None:
    app = _run_app()
    app.session_state["demo_state"] = ProtocolState(
        scenario_id="classification",
        sample_idx=0,
        step=5,
        result={"encrypted_score": "ciphertext", "x_int": [1, 2, 3]},
    ).to_session_dict()

    app.radio[0].set_value("regression").run()

    assert not app.exception
    expected = ProtocolState(
        scenario_id="regression",
        sample_idx=0,
    ).to_session_dict()
    expected["result"]["human_sample_idx"] = 1
    assert app.session_state["demo_state"] == expected
    assert "protocol_trace" not in app.session_state
    assert "protocol_result" not in app.session_state
