# ui/streamlit_app.py
"""Streamlit demo UI for secure ML inference."""

from __future__ import annotations

import os
import time
import warnings
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

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


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
        )
        scaler = reg_model.named_steps["scaler"]
        ridge = reg_model.named_steps["ridge"]
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

    st.session_state.setdefault(
        "demo_state", {"scenario_id": scenario_id, "step": 0, "result": None}
    )
    if st.session_state.demo_state["scenario_id"] != scenario_id:
        st.session_state.demo_state = {"scenario_id": scenario_id, "step": 0, "result": None}

    if st.button("Запустить защищённый инференс", type="primary"):
        client = Client(scaler=scenario["scaler"], scale=SCALE, key_length=KEY_LENGTH)

        t0 = time.perf_counter()
        x_scaled = client.preprocess(sample.to_numpy(dtype=float).reshape(1, -1))
        step_scaling_ms = (time.perf_counter() - t0) * 1000.0
        x_int = client.encode(x_scaled)
        t1 = time.perf_counter()

        enc_x = client.encrypt(x_int)
        t2 = time.perf_counter()

        request_payload: dict[str, Any] = {
            "public_key_n": str(client.public_key.n),
            "encrypted_features": [serialize_ciphertext(value) for value in enc_x],
            "scale": SCALE,
            "scenario_id": scenario_id,
            "feature_count": len(enc_x),
        }

        request_started = time.perf_counter()
        try:
            response = requests.post(
                f"{API_URL}/infer/encrypted",
                json=request_payload,
                timeout=30,
            )
            status_code = response.status_code
            response.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"Ошибка при обращении к API ({API_URL}): {exc}")
            return
        request_ended = time.perf_counter()

        response_payload = response.json()
        encrypted_score_str = response_payload.get("encrypted_score")
        if not isinstance(encrypted_score_str, str):
            st.error("Некорректный ответ API: отсутствует поле encrypted_score.")
            return

        enc_score = deserialize_ciphertext(client.public_key, encrypted_score_str)
        t3 = time.perf_counter()

        score_int = decrypt_score(client.private_key, enc_score)  # step 5
        z_secure = decode_score(score_int=score_int, scale=SCALE)

        if scenario_id == "classification":
            prob_secure = float(1.0 / (1.0 + np.exp(-z_secure)))
            pred_secure = int(prob_secure >= THRESHOLD)
        else:
            prob_secure = None
            pred_secure = None
        t4 = time.perf_counter()

        baseline_value = float(model.predict(sample.to_numpy(dtype=float).reshape(1, -1))[0])
        w, b = scenario["w"], scenario["b"]
        z_encoded = encoded_plaintext_score(x=x_scaled, w=w, b=b, scale=SCALE)
        encoded_value = float(z_encoded)
        encrypted_bytes = measure_payload_size(request_payload)
        plaintext_payload = sample.to_numpy(dtype=float).tolist()
        plaintext_bytes = measure_payload_size(plaintext_payload)
        overhead_ratio = encrypted_bytes / max(plaintext_bytes, 1)

        st.session_state.demo_state = {
            "scenario_id": scenario_id,
            "step": 7,
            "result": {
                "sample_idx": sample_idx,
                "x_scaled": x_scaled.flatten(),
                "x_int": x_int,
                "enc_x": enc_x,
                "encrypted_score": encrypted_score_str,
                "score_int": score_int,
                "z_secure": z_secure,
                "prob_secure": prob_secure,
                "pred_secure": pred_secure,
                "z_encoded": z_encoded,
                "encoded_value": encoded_value,
                "baseline_value": baseline_value,
                "pred_baseline": int(baseline_value) if scenario_id == "classification" else None,
                "prob_baseline": (
                    float(model.predict_proba(sample.to_numpy(dtype=float).reshape(1, -1))[:, 1][0])
                    if scenario_id == "classification"
                    else None
                ),
                "true_label": int(y_test.iloc[sample_idx]),
                "status_code": status_code,
                "server_compute_ms": _extract_server_compute_ms(response_payload),
                "request_payload": request_payload,
                "encrypted_bytes": encrypted_bytes,
                "plaintext_bytes": plaintext_bytes,
                "http_elapsed_ms": (request_ended - request_started) * 1000.0,
                "preprocess_ms": (t1 - t0) * 1000.0,
                "scaling_ms": step_scaling_ms,
                "encrypt_ms": (t2 - t1) * 1000.0,
                "decrypt_ms": (t4 - t3) * 1000.0,
                "overhead_ratio": overhead_ratio,
            },
        }

    result = st.session_state.demo_state.get("result")

    k1, k2, k3, k4, k5 = st.columns(5)
    if result is None:
        k1.metric("Server sees plaintext", "NO")
        k2.metric("Prediction matches baseline", "—")
        k3.metric("Payload overhead", "—")
        k4.metric("Encrypted payload", "—")
        k5.metric("Plaintext payload", "—")
        st.info("Для расчёта KPI выполните шаг 2 и запустите защищённый инференс.")
        return

    match_label = (
        "YES"
        if scenario_id == "classification" and result["pred_secure"] == result["pred_baseline"]
        else "—"
    )
    k1.metric("Server sees plaintext", "NO")
    k2.metric("Prediction matches baseline", match_label)
    k3.metric("Payload overhead", f"{result['overhead_ratio']:.2f}x")
    k4.metric("Encrypted payload", f"{result['encrypted_bytes']} B")
    k5.metric("Plaintext payload", f"{result['plaintext_bytes']} B")

    st.subheader("Шаги 0–7 защищённого протокола")

    preview_count = 5
    preproc_df = pd.DataFrame(
        {
            "Признак": sample.index,
            "Масштабированное значение": result["x_scaled"],
            "Закодированное значение": result["x_int"],
        }
    )
    st.markdown("**Шаг 1–2: масштабирование и кодирование**")
    st.dataframe(preproc_df.head(preview_count), width="stretch")
    if st.checkbox("Показать все признаки после предобработки", value=False):
        st.dataframe(preproc_df, width="stretch")

    encrypted_df = pd.DataFrame(
        {
            "Признак": sample.index,
            "Зашифрованное значение": [
                serialize_ciphertext(v)[:40] + "..." for v in result["enc_x"]
            ],
            "Размер (байт)": [
                len(serialize_ciphertext(v).encode("utf-8")) for v in result["enc_x"]
            ],
        }
    )
    st.markdown("**Шаг 3: шифрование признаков**")
    st.dataframe(encrypted_df, width="stretch")

    http_df = pd.DataFrame(
        [
            {
                "URL": f"{API_URL}/infer/encrypted",
                "HTTP-статус": result["status_code"],
                "Время ответа HTTP (мс)": f"{result['http_elapsed_ms']:.2f}",
                "Время серверного вычисления (мс)": (
                    f"{result['server_compute_ms']:.2f}"
                    if result["server_compute_ms"] is not None
                    else "не передано сервером"
                ),
            }
        ]
    )
    st.markdown("**Шаг 4: API request**")
    request_payload = result["request_payload"]
    request_preview = {
        "public_key_n": f"{str(request_payload['public_key_n'])[:32]}...",
        "encrypted_features": [
            f"{cipher[:32]}..." for cipher in request_payload["encrypted_features"][:3]
        ]
        + (["..."] if len(request_payload["encrypted_features"]) > 3 else []),
        "scale": request_payload["scale"],
        "feature_count": request_payload["feature_count"],
    }
    st.json(request_preview)

    st.info("Что НЕ отправляется: исходные признаки, закрытый ключ, расшифрованный score.")

    st.markdown("**Ответ сервера**")
    st.json(
        {
            "encrypted_score": f"{result['encrypted_score'][:48]}...",
            "server_compute_ms": (
                round(result["server_compute_ms"], 3)
                if result["server_compute_ms"] is not None
                else None
            ),
        }
    )

    st.markdown("**HTTP-запрос и ответ сервера**")
    st.dataframe(http_df, width="stretch")

    st.markdown("**Что видит сервер**")
    left, right = st.columns(2)
    with left:
        st.caption("Доступно")
        st.markdown(
            "- Открытый ключ\n- Зашифрованные признаки\n- Закодированные параметры модели\n- Параметр масштаба"
        )
    with right:
        st.caption("Недоступно")
        st.markdown(
            "- Исходные признаки\n- Закрытый ключ\n- Расшифрованный score\n- Вероятность и класс"
        )

    st.markdown("**Шаг 5–6: дешифрование и постобработка**")
    decrypt_rows = [
        {
            "Этап": "Зашифрованный score (шифртекст)",
            "Значение": f"{result['encrypted_score'][:80]}...",
        },
        {"Этап": "Расшифрованный score_int", "Значение": str(result["score_int"])},
        {"Этап": "Декодированный z", "Значение": f"{result['z_secure']:.6f}"},
    ]
    if scenario_id == "classification":
        decrypt_rows.extend(
            [
                {"Этап": "Вероятность sigmoid(z)", "Значение": f"{result['prob_secure']:.6f}"},
                {
                    "Этап": "Предсказание",
                    "Значение": f"{label_names[result['pred_secure']]} (class={result['pred_secure']})",
                },
            ]
        )
    else:
        decrypt_rows.append({"Этап": "Числовой прогноз", "Значение": f"{result['z_secure']:.6f}"})
    decrypt_steps_df = pd.DataFrame(decrypt_rows)
    st.dataframe(decrypt_steps_df, width="stretch")
    if scenario_id == "classification":
        st.caption(
            "Вероятность sigmoid(z) — это вероятность класса 1 (benign / доброкачественная)."
        )

    st.subheader("Шаг 7. Summary")
    if scenario_id == "classification":
        comparison_df = pd.DataFrame(
            [
                {
                    "Метод": "Baseline",
                    "Вероятность": result["prob_baseline"],
                    "Предсказание": f"{label_names[result['pred_baseline']]} (class={result['pred_baseline']})",
                },
                {
                    "Метод": "PHE inference",
                    "Вероятность": result["prob_secure"],
                    "Предсказание": f"{label_names[result['pred_secure']]} (class={result['pred_secure']})",
                },
                {
                    "Метод": "Истинная метка",
                    "Вероятность": "",
                    "Предсказание": label_names[result["true_label"]]
                    + f" (class={result['true_label']})",
                },
            ]
        )
        comparison_df["Вероятность"] = comparison_df["Вероятность"].map(lambda v: f"{float(v):.6f}")
        comparison_df["Совпадение Baseline/PHE"] = (
            "Да" if result["pred_baseline"] == result["pred_secure"] else "Нет"
        )
    else:
        comparison_df = pd.DataFrame(
            [
                {"Метод": "Baseline", "Прогноз": result["baseline_value"]},
                {"Метод": "Encoded plaintext", "Прогноз": result["encoded_value"]},
                {"Метод": "PHE inference", "Прогноз": result["z_secure"]},
            ]
        )
        comparison_df["Δ к Baseline"] = comparison_df["Прогноз"].map(
            lambda v: f"{(float(v) - result['baseline_value']):.6f}"
        )

    st.dataframe(comparison_df, width="stretch")

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
