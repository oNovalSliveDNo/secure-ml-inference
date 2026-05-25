"""Эксперимент 09: оценка влияния длины ключа Paillier на одном объекте Breast Cancer."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from app.config import N_BENCHMARK_RUNS, RANDOM_STATE, SCALE, TEST_SIZE
from app.crypto import decrypt_score, encrypt_vector, generate_keys
from app.data import load_dataset, split_dataset
from app.encoding import encode_bias, encode_vector, encode_weights
from app.linear_scorer import Server
from app.model import extract_linear_params, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODEL_PATH = Path("results/models/model.pkl")
TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")

KEY_LENGTH_CSV_PATH = TABLES_DIR / "key_length_metrics.csv"
KEY_LENGTH_PLOT_PATH = PLOTS_DIR / "key_length_comparison.png"

KEY_LENGTHS = [
    512,
    1024,
    2048,
]  # 512 бит используется только как небезопасный демонстрационный режим.

CSV_HEADERS = [
    "key_length",
    "keygen_ms_mean",
    "encryption_ms_mean",
    "server_init_ms_mean",
    "server_ms_mean",
    "decryption_ms_mean",
    "total_with_keygen_ms_mean",
    "total_without_keygen_ms_mean",
    "request_size_bytes_mean",
]


def _mean(values: list[float]) -> float:
    """Вычислить среднее значение."""
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize(values: list[float]) -> tuple[float, float, float]:
    """Вычислить среднее значение, стандартное отклонение и медиану."""
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)), float(np.median(arr))


def main() -> None:
    """Выполнить эксперимент по влиянию длины ключа и сохранить результаты."""
    logger.info("Запуск эксперимента по влиянию длины ключа Paillier...")

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, _ = split_dataset(
        features=features,
        target=target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = model.named_steps["scaler"]
    w, b = extract_linear_params(model)

    w_int = encode_weights(w=w, scale=SCALE)
    b_int = encode_bias(b=b, scale=SCALE)

    sample = x_test.iloc[[0]]
    x_scaled = scaler.transform(sample)[0]
    encoded_sample = [int(value) for value in encode_vector(x=x_scaled, scale=SCALE).tolist()]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []

    for key_length in KEY_LENGTHS:
        keygen_ms_values: list[float] = []
        encryption_ms_values: list[float] = []
        server_init_ms_values: list[float] = []
        server_ms_values: list[float] = []
        decryption_ms_values: list[float] = []
        total_with_keygen_ms_values: list[float] = []
        total_without_keygen_ms_values: list[float] = []
        request_sizes: list[float] = []

        for _ in range(N_BENCHMARK_RUNS):
            t_total_start = time.perf_counter()

            t0 = time.perf_counter()
            public_key, private_key = generate_keys(n_length=key_length)
            t1 = time.perf_counter()
            keygen_ms_values.append((t1 - t0) * 1000.0)

            t0 = time.perf_counter()
            server = Server(w_int=w_int, b_int=b_int, public_key=public_key)
            t1 = time.perf_counter()
            server_init_ms_values.append((t1 - t0) * 1000.0)

            t0 = time.perf_counter()
            encrypted_vector = encrypt_vector(public_key=public_key, x_int=encoded_sample)
            t1 = time.perf_counter()
            encryption_ms_values.append((t1 - t0) * 1000.0)

            request_payload = {
                "encrypted_features": [str(value.ciphertext()) for value in encrypted_vector],
                "public_key_n": str(public_key.n),
                "scale": SCALE,
            }
            request_sizes.append(
                float(len(json.dumps(request_payload, ensure_ascii=False).encode("utf-8")))
            )

            t0 = time.perf_counter()
            encrypted_score = server.compute_encrypted_score(encrypted_vector)
            t1 = time.perf_counter()
            server_ms_values.append((t1 - t0) * 1000.0)

            t0 = time.perf_counter()
            _ = decrypt_score(private_key=private_key, encrypted_score=encrypted_score)
            t1 = time.perf_counter()
            decryption_ms_values.append((t1 - t0) * 1000.0)

            t_total_end = time.perf_counter()
            total_with_keygen_ms_values.append((t_total_end - t_total_start) * 1000.0)
            total_without_keygen_ms_values.append(
                encryption_ms_values[-1]
                + server_init_ms_values[-1]
                + server_ms_values[-1]
                + decryption_ms_values[-1]
            )

        logger.info("Длина ключа %d бит обработана.", key_length)
        logger.info(
            "  Генерация ключей, среднее/ст. отклонение/медиана: %s", _summarize(keygen_ms_values)
        )
        logger.info(
            "  Шифрование, среднее/ст. отклонение/медиана: %s", _summarize(encryption_ms_values)
        )
        logger.info(
            "  Инициализация сервера, среднее/ст. отклонение/медиана: %s",
            _summarize(server_init_ms_values),
        )
        logger.info(
            "  Серверное вычисление, среднее/ст. отклонение/медиана: %s",
            _summarize(server_ms_values),
        )
        logger.info(
            "  Расшифрование, среднее/ст. отклонение/медиана: %s", _summarize(decryption_ms_values)
        )

        rows.append(
            {
                "key_length": key_length,
                "keygen_ms_mean": _mean(keygen_ms_values),
                "encryption_ms_mean": _mean(encryption_ms_values),
                "server_init_ms_mean": _mean(server_init_ms_values),
                "server_ms_mean": _mean(server_ms_values),
                "decryption_ms_mean": _mean(decryption_ms_values),
                "total_with_keygen_ms_mean": _mean(total_with_keygen_ms_values),
                "total_without_keygen_ms_mean": _mean(total_without_keygen_ms_values),
                "request_size_bytes_mean": _mean(request_sizes),
            }
        )

    with KEY_LENGTH_CSV_PATH.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    key_lengths = [int(row["key_length"]) for row in rows]
    total_with_keygen = [float(row["total_with_keygen_ms_mean"]) for row in rows]
    total_without_keygen = [float(row["total_without_keygen_ms_mean"]) for row in rows]
    encryption_time = [float(row["encryption_ms_mean"]) for row in rows]
    request_size = [float(row["request_size_bytes_mean"]) for row in rows]

    fig, axis_left = plt.subplots(figsize=(9, 5))
    axis_right = axis_left.twinx()

    axis_left.plot(
        key_lengths,
        total_with_keygen,
        marker="o",
        linewidth=2,
        label="Общее время с генерацией ключей, мс",
    )
    axis_left.plot(
        key_lengths,
        total_without_keygen,
        marker="o",
        linewidth=2,
        label="Общее время без генерации ключей, мс",
    )
    axis_left.plot(
        key_lengths,
        encryption_time,
        marker="o",
        linewidth=2,
        label="Время шифрования, мс",
    )
    axis_right.plot(
        key_lengths,
        request_size,
        marker="s",
        linestyle="--",
        linewidth=2,
        color="tab:purple",
        label="Размер запроса, байт",
    )

    axis_left.set_title("Влияние длины ключа Paillier на время и размер запроса")
    axis_left.set_xlabel("Длина ключа, бит")
    axis_left.set_ylabel("Время, мс")
    axis_right.set_ylabel("Размер запроса, байт")
    axis_left.set_xticks(key_lengths)
    axis_left.grid(True, alpha=0.3)

    left_handles, left_labels = axis_left.get_legend_handles_labels()
    right_handles, right_labels = axis_right.get_legend_handles_labels()
    axis_left.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(KEY_LENGTH_PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    logger.info("Метрики сохранены в %s", KEY_LENGTH_CSV_PATH)
    logger.info("График сохранён в %s", KEY_LENGTH_PLOT_PATH)
    logger.info("Эксперимент по влиянию длины ключа завершён.")


if __name__ == "__main__":
    main()
