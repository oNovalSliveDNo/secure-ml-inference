# ui/streamlit_app.py
"""Streamlit demo UI for secure ML inference."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.client import Client
from app.config import KEY_LENGTH, RANDOM_STATE, SCALE, TEST_SIZE
from app.data import load_dataset, split_dataset
from app.encoding import encode_bias, encode_weights
from app.model import extract_linear_params, load_model
from app.server import Server

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


st.set_page_config(page_title="Защищённый ML-инференс — Демонстрация", layout="wide")

MODEL_PATH = Path("results/models/model.pkl")
TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")

st.set_page_config(page_title="Защищённый ML-инференс — Демонстрация", layout="wide")


@st.cache_resource
def load_resources() -> dict[str, Any]:
    """Load model artifacts and test split once per app process.

    Returns:
        Dict with model, scaler, test features/labels and encoded params.

    Raises:
        FileNotFoundError: If the trained model artifact is missing.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Артефакт модели не найден по пути {MODEL_PATH}. Сначала запустите experiments/01_train_baseline.py."
        )

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, y_test = split_dataset(
        features=features,
        target=target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = model.named_steps["scaler"]
    w, b = extract_linear_params(model)

    return {
        "model": model,
        "scaler": scaler,
        "x_test": x_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "w_int": encode_weights(w=w, scale=SCALE),
        "b_int": encode_bias(b=b, scale=SCALE),
    }


def show_demo_inference(resources: dict[str, Any]) -> None:
    """Render interactive secure inference demo tab.

    Args:
        resources: Cached app resources from ``load_resources``.
    """
    st.header("Демонстрация защищённого инференса")

    x_test = resources["x_test"]
    y_test = resources["y_test"]
    model = resources["model"]

    sample_idx = st.slider("Индекс тестового образца", 0, len(x_test) - 1, 0)
    sample = x_test.iloc[sample_idx]

    st.subheader("Исходные признаки пациента")
    st.dataframe(pd.DataFrame({"Признак": sample.index, "Значение": sample.values}), width='stretch')

    if st.button("Запустить защищённый инференс", type="primary"):
        client = Client(scaler=resources["scaler"], scale=SCALE, key_length=KEY_LENGTH)
        server = Server(
            w_int=resources["w_int"], b_int=resources["b_int"], public_key=client.public_key
        )

        t0 = time.perf_counter()
        x_scaled = client.preprocess(sample.to_numpy(dtype=float).reshape(1, -1))
        x_int = client.encode(x_scaled)
        t1 = time.perf_counter()

        enc_x = client.encrypt(x_int)
        t2 = time.perf_counter()

        enc_score = server.compute_encrypted_score(enc_x)
        t3 = time.perf_counter()

        pred_secure, prob_secure = client.decrypt_and_predict(enc_score)
        t4 = time.perf_counter()

        pred_baseline = int(model.predict(sample.to_numpy(dtype=float).reshape(1, -1))[0])
        prob_baseline = float(
            model.predict_proba(sample.to_numpy(dtype=float).reshape(1, -1))[:, 1][0]
        )

        st.subheader("Выполнение протокола")
        c1, c2, c3 = st.columns(3)
        c1.metric("Количество зашифрованных признаков", len(enc_x))
        c2.metric("Суммарная длина шифртекстов (симв.)", sum(len(str(value.ciphertext())) for value in enc_x))
        c3.metric("Общее время протокола (мс)", f"{(t4 - t0) * 1000.0:.2f}")

        timing_df = pd.DataFrame(
            {
                "Этап": ["Предобработка + кодирование", "Шифрование", "Вычисление на сервере", "Расшифрование и прогноз"],
                "Время (мс)": [
                    (t1 - t0) * 1000.0,
                    (t2 - t1) * 1000.0,
                    (t3 - t2) * 1000.0,
                    (t4 - t3) * 1000.0,
                ],
            }
        )
        st.dataframe(timing_df, width='stretch')

        st.subheader("Результат прогнозирования")
        st.write(f"**Истинная метка:** {int(y_test.iloc[sample_idx])}")
        st.write(f"**Защищённый прогноз:** {pred_secure} (p={prob_secure:.6f})")
        st.write(f"**Базовый прогноз (plaintext):** {pred_baseline} (p={prob_baseline:.6f})")
        st.write(
            f"**Совпадение с базовым прогнозом:** {'✅ yes' if pred_secure == pred_baseline else '❌ no'}"
        )


def show_protocol_view(resources: dict[str, Any]) -> None:
    """Render protocol knowledge separation and encrypted samples.

    Args:
        resources: Cached app resources.
    """
    st.header("Представление протокола")

    client_table = pd.DataFrame(
        {
            "Сторона клиента": [
                "Исходные признаки (x)",
                "Стандартизатор (scaler)",
                "Закрытый ключ",
                "Открытый ключ",
                "Закодированные/зашифрованные признаки",
                "Результат расшифрования",
            ]
        }
    )
    server_table = pd.DataFrame(
        {
            "Сторона сервера": [
                "Закодированные веса модели (w_int)",
                "Закодированное смещение (b_int)",
                "Открытый ключ",
                "Только зашифрованные признаки",
                "Зашифрованный линейный score",
                "Открытые признаки не получает",
            ]
        }
    )
    left, right = st.columns(2)
    left.dataframe(client_table, width='stretch')
    right.dataframe(server_table, width='stretch')

    demo_sample = resources["x_test"].iloc[0].to_numpy(dtype=float)
    client = Client(scaler=resources["scaler"], scale=SCALE, key_length=KEY_LENGTH)
    encrypted = client.encrypt(client.encode(client.preprocess(demo_sample.reshape(1, -1))))

    st.subheader("Пример зашифрованных значений (первые три признака)")
    st.code(
        "\n".join(str(value.ciphertext())[:120] + "..." for value in encrypted[:3]), language="text"
    )


def show_metrics_dashboard() -> None:
    """Render metrics tables and plots from experiment artifacts."""
    st.header("Панель метрик")

    csv_files = sorted(TABLES_DIR.glob("*.csv"))

    if not csv_files:
        st.info("CSV-файлы с метриками не найдены в results/tables/. Сначала запустите эксперименты № 04-07.")

    for csv_file in csv_files:
        st.subheader(csv_file.name)
        st.dataframe(pd.read_csv(csv_file), width='stretch')

    plot_files = sorted(PLOTS_DIR.glob("*.png"))

    if not plot_files:
        st.info("Графики не найдены в results/plots/.")

    for plot_file in plot_files:
        st.subheader(plot_file.name)
        st.image(str(plot_file), width='stretch')


def show_architecture() -> None:
    """Render architecture description tab."""
    st.header("Архитектура системы")
    st.markdown(
        """
### Поток защищённого инференса
1. **Клиент**: исходные признаки → стандартизация → кодирование с фиксированной точкой (`SCALE`) → шифрование Пайе.
2. **Сервер**: получает только зашифрованные признаки и открытый ключ, вычисляет зашифрованный линейный score `Enc(z_int)`.
3. **Клиент**: расшифровывает `z_int`, преобразует в вещественное `z`, применяет сигмоиду и пороговое правило.

### Граница доверия
- Сервер никогда не видит открытые признаки и не имеет доступа к закрытому ключу.
- Клиент не передаёт параметры модели на сервер; сервер хранит только закодированные веса.

### Формат сообщений
- **Запрос**: `public_key_n`, `encrypted_features[]`, `scale`
- **Ответ**: `encrypted_score`
        """
    )


def main() -> None:
    """Run the Streamlit application."""
    st.title("Защищённый ML-инференс — Клиент Streamlit")
    try:
        resources = load_resources()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Демонстрация инференса", "Представление протокола", "Панель метрик", "Архитектура"]
    )

    with tab1:
        show_demo_inference(resources)
    with tab2:
        show_protocol_view(resources)
    with tab3:
        show_metrics_dashboard()
    with tab4:
        show_architecture()


if __name__ == "__main__":
    main()
