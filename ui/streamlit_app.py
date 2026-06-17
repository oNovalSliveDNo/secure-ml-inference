# ui/streamlit_app.py
"""Интерфейс Streamlit для демонстрации защищённого предсказания модели."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.datasets import load_diabetes

from app.client import Client
from app.config import KEY_LENGTH, RANDOM_STATE, SCALE, TEST_SIZE, THRESHOLD
from app.crypto import decrypt_score, deserialize_ciphertext, serialize_ciphertext
from app.data import load_dataset, split_dataset
from app.encoding import decode_score, encoded_plaintext_score
from app.metrics import measure_payload_size
from app.model import extract_linear_params, load_model
from ui.calculation_trace import render_calculation_trace
from ui.components import render_step_statuses
from ui.metrics_view import render_sample_level_metrics, show_metrics_dashboard
from ui.protocol_view import render_protocol_exchange_layout, show_architecture
from ui.styles import apply_styles
from ui.ui_models import ProtocolState

st.set_page_config(page_title="Защищённое предсказание модели — демонстрация", layout="wide")

MODEL_PATH = Path("results/models/model.pkl")
REGRESSION_MODEL_PATH = Path("results/models/regression_model.pkl")
TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")
API_URL = os.getenv("API_URL", "http://localhost:8000")


def _extract_server_compute_ms(payload: dict[str, Any]) -> float | None:
    """Extract server compute time from API response payload."""
    for key in ("server_compute_ms", "compute_ms", "server_ms"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


@st.cache_resource
def load_resources() -> dict[str, Any]:
    """Load artifacts for all supported scenarios once per app process."""
    scenarios: dict[str, dict[str, Any]] = {}

    if MODEL_PATH.exists():
        cls_model = load_model(str(MODEL_PATH))
        cls_features, cls_target = load_dataset()
        _, cls_x_test, _, cls_y_test = split_dataset(
            features=cls_features,
            target=cls_target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        w_cls, b_cls = extract_linear_params(cls_model)
        scenarios["classification"] = {
            "title": "Классификация: диагностика опухоли",
            "task_type": "classification",
            "model": cls_model,
            "scaler": cls_model.named_steps["scaler"],
            "x_test": cls_x_test.reset_index(drop=True),
            "y_test": cls_y_test.reset_index(drop=True),
            "w": w_cls,
            "b": b_cls,
            "postprocess": "sigmoid_and_class",
        }

    if REGRESSION_MODEL_PATH.exists():
        reg_model = load_model(str(REGRESSION_MODEL_PATH))
        reg_ds = load_diabetes(as_frame=True)
        reg_features: pd.DataFrame = reg_ds.data
        reg_target: pd.Series = reg_ds.target
        _, reg_x_test, _, reg_y_test = split_dataset(
            features=reg_features,
            target=reg_target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            use_stratify=False,
        )
        scaler = reg_model.named_steps["scaler"]
        ridge = reg_model.named_steps["regressor"]
        w_reg = ridge.coef_.ravel()
        b_reg = float(ridge.intercept_)
        scenarios["regression"] = {
            "title": "Регрессия: числовой медицинский прогноз",
            "task_type": "regression",
            "model": reg_model,
            "scaler": scaler,
            "x_test": reg_x_test.reset_index(drop=True),
            "y_test": reg_y_test.reset_index(drop=True),
            "w": w_reg,
            "b": b_reg,
            "postprocess": "numeric_prediction",
        }

    if not scenarios:
        raise FileNotFoundError(
            f"Артефакты моделей не найдены ({MODEL_PATH} / {REGRESSION_MODEL_PATH})."
        )
    return {"scenarios": scenarios}


def show_live_protocol_demo(resources: dict[str, Any]) -> None:
    """Render step-by-step interactive protocol demo."""
    st.header("Пошаговая демонстрация")

    # Модель загружается исключительно для демонстрационного сравнения.
    # В защищённом протоколе клиент не использует веса модели.
    st.info(
        "Модель на клиенте загружена только для сравнения результатов. "
        "В реальном протоколе веса модели находятся на сервере."
    )

    scenarios = resources["scenarios"]
    scenario_id = st.radio(
        "Тип задачи",
        options=list(scenarios.keys()),
        format_func=lambda sid: scenarios[sid]["title"],
        horizontal=True,
    )
    scenario = scenarios[scenario_id]
    x_test = scenario["x_test"]
    y_test = scenario["y_test"]
    model = scenario["model"]

    sample_idx = st.slider("Шаг 0. Выберите индекс тестового образца", 0, len(x_test) - 1, 0)
    sample = x_test.iloc[sample_idx]

    st.session_state.setdefault(
        "demo_state",
        ProtocolState(scenario_id=scenario_id, sample_idx=sample_idx).to_session_dict(),
    )
    if (
        st.session_state.demo_state["scenario_id"] != scenario_id
        or st.session_state.demo_state.get("sample_idx") != sample_idx
    ):
        st.session_state.demo_state = ProtocolState(
            scenario_id=scenario_id, sample_idx=sample_idx
        ).to_session_dict()

    wizard_state: dict[str, Any] = st.session_state.demo_state
    result: dict[str, Any] = wizard_state["result"]
    current_step = int(wizard_state["step"])

    def reset_demo() -> None:
        st.session_state.demo_state = ProtocolState(
            scenario_id=scenario_id, sample_idx=sample_idx
        ).to_session_dict()

    def execute_next_step() -> bool:
        """Execute exactly one pending operation and advance the demo state by at most one."""
        step = int(wizard_state["step"])
        if step >= 7:
            return True

        if step == 0:
            wizard_state["step"] = 1
            return True

        if step == 1:
            if "x_scaled" not in result:
                client = Client(scaler=scenario["scaler"], scale=SCALE, key_length=KEY_LENGTH)
                t0 = time.perf_counter()
                x_scaled = client.preprocess(sample.to_numpy(dtype=float).reshape(1, -1))
                result.update(
                    {
                        "client": client,
                        "x_scaled": x_scaled.flatten(),
                        "scaling_ms": (time.perf_counter() - t0) * 1000.0,
                    }
                )
            wizard_state["step"] = 2
            return True

        if step == 2:
            if "x_int" not in result:
                result["x_int"] = result["client"].encode(result["x_scaled"])
            wizard_state["step"] = 3
            return True

        if step == 3:
            if "enc_x" not in result:
                t1 = time.perf_counter()
                result["enc_x"] = result["client"].encrypt(result["x_int"])
                result["encrypt_ms"] = (time.perf_counter() - t1) * 1000.0
            wizard_state["step"] = 4
            return True

        if step == 4:
            if "encrypted_score" not in result:
                client = result["client"]
                payload = {
                    "public_key_n": str(client.public_key.n),
                    "encrypted_features": [serialize_ciphertext(v) for v in result["enc_x"]],
                    "scale": SCALE,
                    "scenario_id": scenario_id,
                    "feature_count": len(result["enc_x"]),
                }
                t_http0 = time.perf_counter()
                try:
                    response = requests.post(f"{API_URL}/infer/encrypted", json=payload, timeout=30)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    st.error(f"Ошибка при обращении к серверу ({API_URL}): {exc}")
                    return False
                http_elapsed_ms = (time.perf_counter() - t_http0) * 1000.0
                response_payload = response.json()
                encrypted_score_str = response_payload.get("encrypted_score")
                if not isinstance(encrypted_score_str, str):
                    st.error("Некорректный ответ сервера: отсутствует зашифрованный результат.")
                    return False
                result.update(
                    {
                        "request_payload": payload,
                        "status_code": response.status_code,
                        "server_compute_ms": _extract_server_compute_ms(response_payload),
                        "http_elapsed_ms": http_elapsed_ms,
                        "encrypted_score": encrypted_score_str,
                    }
                )
            wizard_state["step"] = 5
            return True

        if step == 5:
            if "comparison_df" not in result:
                client = result["client"]
                t_dec0 = time.perf_counter()
                enc_score = deserialize_ciphertext(client.public_key, result["encrypted_score"])
                score_int = decrypt_score(client.private_key, enc_score)
                result.update(
                    {
                        "score_int": score_int,
                        "z_secure": decode_score(score_int=score_int, scale=SCALE),
                        "decrypt_ms": (time.perf_counter() - t_dec0) * 1000.0,
                    }
                )
                x_raw = sample.to_numpy(dtype=float).reshape(1, -1)
                x_scaled = np.array(result["x_scaled"])
                w, b = scenario["w"], scenario["b"]
                z_encoded = float(encoded_plaintext_score(x=x_scaled, w=w, b=b, scale=SCALE))
                baseline_pred = model.predict(x_raw)
                baseline_value = float(baseline_pred[0])
                result["true_label"] = float(y_test.iloc[sample_idx])
                result["baseline_value"] = baseline_value
                result["z_encoded"] = z_encoded
                if scenario_id == "classification":
                    z_baseline = (
                        float(model.decision_function(x_raw)[0])
                        if hasattr(model, "decision_function")
                        else baseline_value
                    )
                    prob_baseline = float(model.predict_proba(x_raw)[:, 1][0])
                    pred_baseline = int(prob_baseline >= THRESHOLD)
                    prob_encoded = float(1.0 / (1.0 + np.exp(-z_encoded)))
                    pred_encoded = int(prob_encoded >= THRESHOLD)
                    prob_secure = float(1.0 / (1.0 + np.exp(-result["z_secure"])))
                    pred_secure = int(prob_secure >= THRESHOLD)
                    result.update(
                        {
                            "pred_baseline": pred_baseline,
                            "pred_encoded": pred_encoded,
                            "pred_secure": pred_secure,
                            "prob_baseline": prob_baseline,
                            "prob_encoded": prob_encoded,
                            "prob_secure": prob_secure,
                        }
                    )
                    result["comparison_df"] = pd.DataFrame(
                        [
                            {
                                "Метод": "Обычная модель без защиты",
                                "Линейный результат z": z_baseline,
                                "Вероятность класса 1": prob_baseline,
                                "Класс": pred_baseline,
                            },
                            {
                                "Метод": "Открытый расчёт после кодирования",
                                "Линейный результат z": z_encoded,
                                "Вероятность класса 1": prob_encoded,
                                "Класс": pred_encoded,
                            },
                            {
                                "Метод": "Защищённый расчёт по зашифрованным данным",
                                "Линейный результат z": result["z_secure"],
                                "Вероятность класса 1": prob_secure,
                                "Класс": pred_secure,
                            },
                            {
                                "Метод": "Истинная метка",
                                "Линейный результат z": None,
                                "Вероятность класса 1": None,
                                "Класс": int(y_test.iloc[sample_idx]),
                            },
                        ]
                    )
                else:
                    pred_baseline_value = float(baseline_value)
                    pred_encoded_value = float(z_encoded)
                    pred_secure_value = float(result["z_secure"])
                    result.update(
                        {
                            "pred_baseline": pred_baseline_value,
                            "pred_encoded": pred_encoded_value,
                            "pred_secure": pred_secure_value,
                        }
                    )
                    result["comparison_df"] = pd.DataFrame(
                        [
                            {"Метод": "Обычная модель без защиты", "Прогноз": pred_baseline_value},
                            {
                                "Метод": "Открытый расчёт после кодирования",
                                "Прогноз": pred_encoded_value,
                            },
                            {
                                "Метод": "Защищённый расчёт по зашифрованным данным",
                                "Прогноз": pred_secure_value,
                            },
                            {
                                "Метод": "Истинное значение",
                                "Прогноз": float(y_test.iloc[sample_idx]),
                            },
                        ]
                    )
            wizard_state["step"] = 6
            return True

        if step == 6:
            if "encrypted_bytes" not in result:
                payload = result["request_payload"]
                result["encrypted_bytes"] = measure_payload_size(payload)
                result["plaintext_bytes"] = measure_payload_size(
                    sample.to_numpy(dtype=float).tolist()
                )
                result["overhead_ratio"] = result["encrypted_bytes"] / max(
                    result["plaintext_bytes"], 1
                )
            wizard_state["step"] = 7
            return True

        return True

    controls = st.columns(4)
    with controls[0]:
        start_clicked = st.button("Запустить демонстрацию", disabled=current_step != 0)
    with controls[1]:
        next_clicked = st.button("Следующий шаг", disabled=current_step >= 7)
    with controls[2]:
        finish_clicked = st.button("Выполнить до конца", disabled=current_step >= 7)
    with controls[3]:
        restart_clicked = st.button("Начать заново")

    if restart_clicked:
        reset_demo()
        st.rerun()
    if (start_clicked or next_clicked) and execute_next_step():
        st.rerun()
    if finish_clicked:
        while int(wizard_state["step"]) < 7:
            if not execute_next_step():
                break
        st.rerun()

    current_step = int(wizard_state["step"])
    st.subheader(f"Шаг {current_step} из 7")

    step1_col, step1_note = st.columns([3, 2])
    with step1_col:
        st.subheader("Исходные признаки (видны только клиенту)")
        st.dataframe(
            pd.DataFrame({"Признак": sample.index, "Значение": sample.values}),
            width="stretch",
        )
    with step1_note:
        st.warning("Эти данные не отправляются на сервер.")

    render_step_statuses(
        [
            "Шаг 1. Масштабирование",
            "Шаг 2. Кодирование",
            "Шаг 3. Генерация ключей и шифрование",
            "Шаг 4. Отправка запроса",
            "Шаг 5. Серверное вычисление и возврат ответа",
            "Шаг 6. Расшифрование и постобработка",
            "Шаг 7. Итоговые метрики",
        ],
        current_step,
    )

    render_protocol_exchange_layout(
        result=result,
        scenario=scenario,
        sample=sample,
        current_step=current_step,
        scale=SCALE,
        scenario_id=scenario_id,
    )

    if current_step >= 1:
        with st.expander("Шаг 1. Масштабирование", expanded=current_step == 1):
            if "x_scaled" in result:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Признак": sample.index,
                            "Исходное значение": sample.values,
                            "После масштабирования": result["x_scaled"],
                        }
                    ),
                    width="stretch",
                )
            else:
                st.info("Нажмите «Следующий шаг», чтобы выполнить масштабирование.")

    if current_step >= 2:
        with st.expander("Шаг 2. Кодирование", expanded=current_step == 2):
            if "x_int" in result:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "После масштабирования": result["x_scaled"],
                            "Целое число": result["x_int"],
                        }
                    ),
                    width="stretch",
                )

    if current_step >= 3:
        with st.expander("Шаг 3. Генерация ключей и шифрование", expanded=current_step == 3):
            if "enc_x" in result:
                client = result["client"]
                st.caption("Открытая часть ключа, которая передаётся серверу:")
                st.code(f"n = {str(client.public_key.n)[:48]}...")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Целое число": result["x_int"],
                            "Зашифрованное значение": [
                                serialize_ciphertext(v)[:36] + "..." for v in result["enc_x"]
                            ],
                            "Размер (байт)": [
                                len(serialize_ciphertext(v).encode("utf-8"))
                                for v in result["enc_x"]
                            ],
                        }
                    ),
                    width="stretch",
                )

    if current_step >= 4:
        with st.expander("Шаг 4. Отправка запроса", expanded=current_step == 4):
            if "request_payload" in result:
                st.json(
                    {
                        "Открытая часть ключа": result["request_payload"]["public_key_n"][:32]
                        + "...",
                        "Первые зашифрованные признаки": [
                            c[:32] + "..."
                            for c in result["request_payload"]["encrypted_features"][:3]
                        ],
                        "Масштаб кодирования": result["request_payload"]["scale"],
                        "Количество признаков": result["request_payload"]["feature_count"],
                    }
                )
            else:
                st.info("Запрос будет отправлен на следующем шаге.")

    if current_step >= 5:
        with st.expander(
            "Шаг 5. Серверное вычисление и возврат ответа", expanded=current_step == 5
        ):
            if "encrypted_score" in result:
                st.write(
                    f"Статус ответа сервера: {result['status_code']}; "
                    f"время вычисления на сервере: {result['server_compute_ms']} мс"
                )
                st.code(result["encrypted_score"][:80] + "...")

    if current_step >= 6:
        with st.expander("Шаг 6. Расшифрование и постобработка", expanded=current_step == 6):
            if "z_secure" in result:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Этап": "Зашифрованный результат",
                                "Значение": result["encrypted_score"][:80] + "...",
                            },
                            {"Этап": "Целочисленный результат", "Значение": result["score_int"]},
                            {
                                "Этап": "Расшифрованное значение z",
                                "Значение": f"{result['z_secure']:.6f}",
                            },
                        ]
                    ),
                    width="stretch",
                )
            if "comparison_df" in result:
                st.dataframe(result["comparison_df"], width="stretch")

    if current_step >= 7 and "comparison_df" in result:
        with st.expander("Шаг 7. Итоговые метрики", expanded=True):
            if "encrypted_bytes" not in result:
                payload = result["request_payload"]
                result["encrypted_bytes"] = measure_payload_size(payload)
                result["plaintext_bytes"] = measure_payload_size(
                    sample.to_numpy(dtype=float).tolist()
                )
                result["overhead_ratio"] = result["encrypted_bytes"] / max(
                    result["plaintext_bytes"], 1
                )
            st.subheader("Шаг 7. Итоговые метрики")

    render_calculation_trace(result=result, scenario=scenario, sample=sample, scale=SCALE)
    render_sample_level_metrics(result, scenario_id)


def main() -> None:
    """Run the Streamlit application."""
    apply_styles()
    st.title("Защищённое предсказание модели — пошаговая демонстрация")
    try:
        resources = load_resources()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tab1, tab2, tab3 = st.tabs(
        ["Пошаговая демонстрация", "Результаты экспериментов", "Архитектура и модель угроз"]
    )

    with tab1:
        show_live_protocol_demo(resources)
    with tab2:
        show_metrics_dashboard(TABLES_DIR, PLOTS_DIR)
    with tab3:
        show_architecture()


if __name__ == "__main__":
    main()
