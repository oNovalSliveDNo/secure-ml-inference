# ui/streamlit_app.py
"""Streamlit demo UI for secure ML inference."""

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

st.set_page_config(page_title="Защищённый ML-инференс — Демонстрация", layout="wide")

MODEL_PATH = Path("results/models/model.pkl")
REGRESSION_MODEL_PATH = Path("results/models/regression_model.pkl")
TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")
SCHEMES_DIR = Path("docs/schemes")
API_URL = os.getenv("API_URL", "http://localhost:8000")


def _extract_server_compute_ms(payload: dict[str, Any]) -> float | None:
    """Extract server compute time from API response payload."""
    for key in ("server_compute_ms", "compute_ms", "server_ms"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _get_table_explanation(csv_name: str, df: pd.DataFrame) -> str:
    """Return a concise Russian explanation for metrics table."""
    lower_name = csv_name.lower()
    numeric_df = df.select_dtypes(include="number")

    if "accuracy" in lower_name or "acc" in lower_name:
        if numeric_df.shape[1] >= 2:
            delta = abs(float(numeric_df.iloc[:, 0].mean()) - float(numeric_df.iloc[:, 1].mean()))
            if delta < 1e-6:
                return "Точность модели не изменилась при переходе к зашифрованному инференсу."
        return "Таблица демонстрирует сопоставимое качество модели в открытом и защищённом режимах."

    if "latency" in lower_name or "time" in lower_name or "timing" in lower_name:
        return "Таблица показывает структуру задержек протокола и вклад каждого этапа вычислений."

    if "overhead" in lower_name or "payload" in lower_name:
        return "Таблица отражает сетевые издержки и дополнительный объём данных из-за шифрования."

    if not numeric_df.empty:
        return "Таблица фиксирует количественные результаты экспериментов для проверки воспроизводимости."
    return "Таблица содержит вспомогательные экспериментальные сведения и контекст измерений."


def _get_plot_explanation(plot_name: str) -> str:
    """Return a concise Russian explanation for plot."""
    lower_name = plot_name.lower()
    if "latency" in lower_name or "time" in lower_name or "timing" in lower_name:
        return "График показывает, как меняется время выполнения при росте вычислительной нагрузки."
    if "feature" in lower_name or "dim" in lower_name:
        return "График иллюстрирует зависимость затрат протокола от числа признаков."
    if "accuracy" in lower_name or "auc" in lower_name:
        return "График подтверждает сохранение качества модели в защищённом контуре инференса."
    if "payload" in lower_name or "size" in lower_name or "overhead" in lower_name:
        return "График показывает рост сетевой нагрузки, вызванный передачей шифртекстов."
    return "График визуализирует экспериментальные результаты и подтверждает наблюдаемую динамику метрик."


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
            "title": "Классификация (Breast Cancer)",
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
            "title": "Регрессия (Diabetes)",
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
    st.header("Демонстрация протокола")

    # Модель загружается исключительно для демонстрационного сравнения.
    # В защищённом протоколе клиент не использует веса модели.
    st.info(
        "Модель на клиенте загружена только для сравнения результатов. "
        "В реальном протоколе веса модели находятся на сервере."
    )

    scenarios = resources["scenarios"]
    scenario_id = st.radio(
        "Сценарий инференса",
        options=list(scenarios.keys()),
        format_func=lambda sid: scenarios[sid]["title"],
        horizontal=True,
    )
    scenario = scenarios[scenario_id]
    x_test = scenario["x_test"]
    y_test = scenario["y_test"]
    model = scenario["model"]
    label_names = {0: "malignant (злокачественная)", 1: "benign (доброкачественная)"}

    sample_idx = st.slider("Шаг 0. Выберите индекс тестового образца", 0, len(x_test) - 1, 0)
    sample = x_test.iloc[sample_idx]

    step1_col, step1_note = st.columns([3, 2])
    with step1_col:
        st.subheader("Исходные признаки (видны только клиенту)")
        st.dataframe(
            pd.DataFrame({"Признак": sample.index, "Значение": sample.values}),
            width="stretch",
        )
    with step1_note:
        st.warning("Эти данные не отправляются на сервер.")

    st.session_state.setdefault("demo_state", {"scenario_id": scenario_id, "step": 1, "result": {}})
    if st.session_state.demo_state["scenario_id"] != scenario_id:
        st.session_state.demo_state = {"scenario_id": scenario_id, "step": 1,
                                       "result": {}}
        if st.session_state.demo_state.get("sample_idx") != sample_idx:
            st.session_state.demo_state = {"scenario_id": scenario_id, "step": 1,
                                           "result": {}, "sample_idx": sample_idx}

    wizard_state: dict[str, Any] = st.session_state.demo_state
    result: dict[str, Any] = wizard_state["result"]
    current_step = wizard_state["step"]

    if st.button("Начать заново"):
        st.session_state.demo_state = {"scenario_id": scenario_id, "step": 1,
                                       "result": {}, "sample_idx": sample_idx}
        st.rerun()

    for idx, title in enumerate(
            [
                "Шаг 1. Масштабирование",
                "Шаг 2. Кодирование",
                "Шаг 3. Генерация ключей и шифрование",
                "Шаг 4. Отправка на сервер",
                "Шаг 5. Расшифровка",
                "Шаг 6. Постобработка и сравнение",
                "Шаг 7. Итоговая сводка",
            ],
            start=1,
    ):
        marker = "🟢" if current_step == idx else ("✅" if current_step > idx else "⚪")
        st.markdown(f"**{marker} {title}**")

    if current_step >= 1:
        with st.expander("Шаг 1. Масштабирование", expanded=current_step == 1):
            if "x_scaled" in result:
                pass
            elif st.button("Масштабировать признаки", type="primary"):
                client = Client(scaler=scenario["scaler"], scale=SCALE,
                                key_length=KEY_LENGTH)
                t0 = time.perf_counter()
                x_scaled = client.preprocess(sample.to_numpy(dtype=float).reshape(1, -1))
                result.update({"client": client, "x_scaled": x_scaled.flatten(),
                               "scaling_ms": (time.perf_counter() - t0) * 1000.0})
                wizard_state["step"] = 2
                st.rerun()
            if "x_scaled" in result:
                st.dataframe(pd.DataFrame({"Признак": sample.index, "Raw": sample.values,
                                           "Scaled": result["x_scaled"]}),
                             width="stretch")

    if current_step >= 2:
        with st.expander("Шаг 2. Кодирование", expanded=current_step == 2):
            if "x_int" not in result and st.button("Закодировать в целые числа",
                                                   type="primary"):
                client = result["client"]
                x_scaled = np.array(result["x_scaled"]).reshape(1, -1)
                result["x_int"] = client.encode(x_scaled)
                wizard_state["step"] = 3
                st.rerun()
            if "x_int" in result:
                st.dataframe(pd.DataFrame(
                    {"Scaled": result["x_scaled"], "Encoded (int)": result["x_int"]}),
                             width="stretch")

    if current_step >= 3:
        with st.expander("Шаг 3. Генерация ключей и шифрование",
                         expanded=current_step == 3):
            if "enc_x" not in result and st.button("Сгенерировать ключи и зашифровать",
                                                   type="primary"):
                client = result["client"]
                t1 = time.perf_counter()
                result["enc_x"] = client.encrypt(result["x_int"])
                result["encrypt_ms"] = (time.perf_counter() - t1) * 1000.0
                wizard_state["step"] = 4
                st.rerun()
            if "enc_x" in result:
                client = result["client"]
                st.code(f"public_key_n: {str(client.public_key.n)[:48]}...")
                st.dataframe(pd.DataFrame({"Encoded": result["x_int"], "Encrypted": [
                    serialize_ciphertext(v)[:36] + "..." for v in result["enc_x"]],
                                           "Размер (байт)": [
                                               len(serialize_ciphertext(v).encode(
                                                   'utf-8')) for v in result["enc_x"]]}),
                             width="stretch")

    if current_step >= 4:
        with st.expander("Шаг 4. Отправка на сервер", expanded=current_step == 4):
            if "encrypted_score" not in result and st.button(
                    "Отправить зашифрованный запрос", type="primary"):
                client = result["client"]
                payload = {"public_key_n": str(client.public_key.n),
                           "encrypted_features": [serialize_ciphertext(v) for v in
                                                  result["enc_x"]], "scale": SCALE,
                           "scenario_id": scenario_id,
                           "feature_count": len(result["enc_x"])}
                t_http0 = time.perf_counter()
                try:
                    response = requests.post(f"{API_URL}/infer/encrypted", json=payload,
                                             timeout=30)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    st.error(f"Ошибка при обращении к API ({API_URL}): {exc}")
                    return
                http_elapsed_ms = (time.perf_counter() - t_http0) * 1000.0
                response_payload = response.json()
                encrypted_score_str = response_payload.get("encrypted_score")
                if not isinstance(encrypted_score_str, str):
                    st.error("Некорректный ответ API: отсутствует поле encrypted_score.")
                    return
                result.update(
                    {"request_payload": payload, "status_code": response.status_code,
                     "server_compute_ms": _extract_server_compute_ms(response_payload),
                     "http_elapsed_ms": http_elapsed_ms,
                     "encrypted_score": encrypted_score_str})
                wizard_state["step"] = 5
                st.rerun()
            if "request_payload" in result:
                st.json({"public_key_n": result["request_payload"]["public_key_n"][
                                         :32] + "...",
                         "encrypted_features": [c[:32] + "..." for c in
                                                result["request_payload"][
                                                    "encrypted_features"][:3]],
                         "scale": result["request_payload"]["scale"],
                         "feature_count": result["request_payload"]["feature_count"]})
                st.write(
                    f"HTTP-статус: {result['status_code']}, server_compute_ms: {result['server_compute_ms']}")

    if current_step >= 5:
        with st.expander("Шаг 5. Расшифровка", expanded=current_step == 5):
            if "z_secure" not in result and st.button("Расшифровать результат",
                                                      type="primary"):
                client = result["client"]
                t_dec0 = time.perf_counter()
                enc_score = deserialize_ciphertext(client.public_key,
                                                   result["encrypted_score"])
                score_int = decrypt_score(client.private_key, enc_score)
                result.update({"score_int": score_int,
                               "z_secure": decode_score(score_int=score_int, scale=SCALE),
                               "decrypt_ms": (time.perf_counter() - t_dec0) * 1000.0})
                wizard_state["step"] = 6
                st.rerun()
            if "z_secure" in result:
                st.dataframe(pd.DataFrame([{"Этап": "Encrypted score",
                                            "Значение": result["encrypted_score"][
                                                        :80] + "..."},
                                           {"Этап": "score_int",
                                            "Значение": result["score_int"]},
                                           {"Этап": "z",
                                            "Значение": f"{result['z_secure']:.6f}"}]),
                             width="stretch")

    if current_step >= 6:
        with st.expander("Шаг 6. Постобработка и сравнение", expanded=current_step == 6):
            if "comparison_df" not in result and st.button("Получить прогноз и сравнить",
                                                           type="primary"):
                x_raw = sample.to_numpy(dtype=float).reshape(1, -1)
                x_scaled = np.array(result["x_scaled"]).reshape(1, -1)
                w, b = scenario["w"], scenario["b"]
                z_encoded = float(
                    encoded_plaintext_score(x=x_scaled, w=w, b=b, scale=SCALE))
                baseline_pred = model.predict(x_raw)
                baseline_value = float(baseline_pred[0])
                result["true_label"] = float(y_test.iloc[sample_idx])
                result["baseline_value"] = baseline_value
                result["z_encoded"] = z_encoded
                if scenario_id == "classification":
                    z_baseline = float(model.decision_function(x_raw)[0]) if hasattr(
                        model, "decision_function") else baseline_value
                    prob_baseline = float(model.predict_proba(x_raw)[:, 1][0])
                    pred_baseline = int(prob_baseline >= THRESHOLD)
                    prob_encoded = float(1.0 / (1.0 + np.exp(-z_encoded)))
                    pred_encoded = int(prob_encoded >= THRESHOLD)
                    prob_secure = float(1.0 / (1.0 + np.exp(-result["z_secure"])))
                    pred_secure = int(prob_secure >= THRESHOLD)
                    result.update(
                        {"pred_baseline": pred_baseline, "pred_encoded": pred_encoded,
                         "pred_secure": pred_secure, "prob_baseline": prob_baseline,
                         "prob_encoded": prob_encoded, "prob_secure": prob_secure})
                    result["comparison_df"] = pd.DataFrame([{"Метод": "Baseline",
                                                             "Linear score": z_baseline,
                                                             "Probability": prob_baseline,
                                                             "Class": pred_baseline},
                                                            {"Метод": "Encoded plaintext",
                                                             "Linear score": z_encoded,
                                                             "Probability": prob_encoded,
                                                             "Class": pred_encoded},
                                                            {"Метод": "PHE inference",
                                                             "Linear score": result[
                                                                 "z_secure"],
                                                             "Probability": prob_secure,
                                                             "Class": pred_secure},
                                                            {"Метод": "Истинная метка",
                                                             "Linear score": None,
                                                             "Probability": None,
                                                             "Class": int(y_test.iloc[
                                                                              sample_idx])}])
                else:
                    pred_baseline = baseline_value
                    pred_encoded = z_encoded
                    pred_secure = result["z_secure"]
                    result.update(
                        {"pred_baseline": pred_baseline, "pred_encoded": pred_encoded,
                         "pred_secure": pred_secure})
                    result["comparison_df"] = pd.DataFrame(
                        [{"Метод": "Baseline", "Predicted value": pred_baseline},
                         {"Метод": "Encoded plaintext", "Predicted value": pred_encoded},
                         {"Метод": "PHE inference", "Predicted value": pred_secure},
                         {"Метод": "Истинное значение",
                          "Predicted value": float(y_test.iloc[sample_idx])}])
                payload = result["request_payload"]
                result["encrypted_bytes"] = measure_payload_size(payload)
                result["plaintext_bytes"] = measure_payload_size(
                    sample.to_numpy(dtype=float).tolist())
                result["overhead_ratio"] = result["encrypted_bytes"] / max(
                    result["plaintext_bytes"], 1)
                wizard_state["step"] = 7
                st.rerun()
            if "comparison_df" in result:
                st.dataframe(result["comparison_df"], width="stretch")

    if current_step >= 7 and "comparison_df" in result:
        st.subheader("Шаг 7. Итоговая сводка")

    k1, k2, k3, k4, k5 = st.columns(5)
    if not result:
        k1.metric("Server sees plaintext", "NO")
        k2.metric("Prediction matches baseline", "—")
        k3.metric("Payload overhead", "—")
        k4.metric("Encrypted payload", "—")
        k5.metric("Plaintext payload", "—")
        st.info("Для расчёта KPI выполните шаг 2 и запустите защищённый инференс.")
        return

    match_label = "ДА" if scenario_id == "classification" and result.get("pred_secure") == result.get("pred_baseline") else "НЕТ"
    k1.metric("Server sees plaintext", "NO")
    k2.metric("Prediction matches baseline", match_label)
    k3.metric("Payload overhead", f"{result['overhead_ratio']:.2f}x")
    k4.metric("Encrypted payload", f"{result['encrypted_bytes']} B")
    k5.metric("Plaintext payload", f"{result['plaintext_bytes']} B")

    if scenario_id == "regression":
        delta_phe_baseline = abs(
            float(result["pred_secure"]) - float(result["pred_baseline"]))
        delta_phe_encoded = abs(
            float(result["pred_secure"]) - float(result["pred_encoded"]))
        r1, r2, r3 = st.columns(3)
        r1.metric("Разность PHE и Baseline", f"{delta_phe_baseline:.6f}")
        r2.metric("Разность PHE и Encoded", f"{delta_phe_encoded:.6f}")
        r3.metric("Близость к baseline", "ДА" if delta_phe_baseline < 0.01 else "НЕТ")

    scheme_files = sorted(SCHEMES_DIR.glob("*.png"))
    if scheme_files:
        st.subheader("Схемы протокола")
        for scheme in scheme_files:
            st.image(str(scheme), caption=scheme.name, width="stretch")


def show_metrics_dashboard() -> None:
    """Render experiment evidence with explanatory comments."""
    st.header("Экспериментальные свидетельства")

    csv_files = sorted(TABLES_DIR.glob("*.csv"))
    plot_files = sorted(PLOTS_DIR.glob("*.png"))

    if not csv_files and not plot_files:
        st.info(
            "Экспериментальные артефакты не найдены. Запустите сценарии в директории experiments/, чтобы сформировать таблицы и графики."
        )
        return

    if not csv_files:
        st.info(
            "CSV-файлы с метриками отсутствуют в results/tables/. Запустите эксперименты для формирования табличных результатов."
        )

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        st.subheader(f"Таблица: {csv_file.name}")
        st.dataframe(df, width="stretch")

        if csv_file.name == "latency_metrics.csv" and {"stage", "mean_ms"}.issubset(df.columns):
            totals = df.set_index("stage")["mean_ms"].to_dict()
            total_with_keygen = totals.get("total")
            total_without_keygen = totals.get("total_without_keygen")
            if total_with_keygen is not None and total_without_keygen is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Общее время (с keygen), ms", f"{total_with_keygen:.2f}")
                with col2:
                    st.metric("Общее время (без keygen), ms", f"{total_without_keygen:.2f}")

        st.markdown(f"**Интерпретация:** {_get_table_explanation(csv_file.name, df)}")

    if not plot_files:
        st.info(
            "Файлы графиков отсутствуют в results/plots/. Запустите эксперименты визуализации результатов."
        )

    for plot_file in plot_files:
        st.subheader(f"График: {plot_file.name}")
        st.image(str(plot_file), width="stretch")
        st.markdown(f"**Вывод:** {_get_plot_explanation(plot_file.name)}")


def show_architecture() -> None:
    """Render architecture description tab."""
    st.header("Архитектура и модель угроз")
    st.markdown(
        """
### Поток защищённого инференса
1. **Клиент**: исходные признаки → стандартизация → кодирование с фиксированной точкой (`SCALE`) → шифрование Пайе.
2. **Сервер**: получает только зашифрованные признаки и открытый ключ, вычисляет зашифрованный линейный score `Enc(z_int)`.
3. **Клиент**: расшифровывает `z_int`, преобразует в вещественное `z`, применяет сигмоиду и пороговое правило.

### Граница доверия
- Сервер никогда не видит открытые признаки и не имеет доступа к закрытому ключу.
- Клиент не передаёт параметры модели на сервер; сервер хранит только закодированные веса.

### Модель угроз
- Предполагается честный, но любопытный сервер: сервер корректно исполняет протокол, но пытается извлечь сведения из наблюдаемых данных.
- Канал связи может наблюдаться внешним нарушителем, поэтому передаются только шифртексты и служебные открытые параметры.
- Закрытый ключ хранится только у клиента, что исключает восстановление признаков и линейного score на стороне сервера.

### Формат сообщений
- **Запрос**: `public_key_n`, `encrypted_features[]`, `scale`
- **Ответ**: `encrypted_score`
        """
    )

    st.subheader("Схемы для презентации протокола")
    required_schemes = [
        "protocol_flow.png",
        "threat_model.png",
        "math_flow.png",
        "plaintext_vs_encoded_vs_phe.png",
    ]
    missing_schemes = [name for name in required_schemes if not (SCHEMES_DIR / name).exists()]
    if missing_schemes:
        st.warning(
            "Не найдены схемы: "
            + ", ".join(missing_schemes)
            + ". Запустите `python experiments/generate_schemes.py`."
        )
    else:
        st.success("Все схемы найдены и готовы для встраивания.")

    for scheme_name in required_schemes:
        scheme_path = SCHEMES_DIR / scheme_name
        if scheme_path.exists():
            st.image(str(scheme_path), caption=scheme_name, width="stretch")


def main() -> None:
    """Run the Streamlit application."""
    st.title("Защищённый ML-инференс — Live Protocol Demo")
    try:
        resources = load_resources()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tab1, tab2, tab3 = st.tabs(
        ["Демонстрация протокола", "Экспериментальные свидетельства", "Архитектура и модель угроз"]
    )

    with tab1:
        show_live_protocol_demo(resources)
    with tab2:
        show_metrics_dashboard()
    with tab3:
        show_architecture()


if __name__ == "__main__":
    main()
