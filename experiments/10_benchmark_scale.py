"""Эксперимент 10: оценка влияния SCALE на точность кодирования и совпадение с PHE."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from app.client import Client
from app.config import RANDOM_STATE, TEST_SIZE
from app.encoding import encode_bias, encode_weights, encoded_plaintext_score
from app.inference import encoded_plaintext_inference, phe_inference_batch, plaintext_inference
from app.linear_scorer import Server
from app.model import compute_manual_score, extract_linear_params, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODEL_PATH = Path("results/models/model.pkl")
TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")

SCALE_CSV_PATH = TABLES_DIR / "scale_metrics.csv"
SCALE_PLOT_PATH = PLOTS_DIR / "scale_impact.png"

SCALE_VALUES = [100, 1000, 10000, 100000]
PHE_SUBSET_SIZE = 15
PHE_KEY_LENGTH = 512

CSV_HEADERS = [
    "scale",
    "encoded_match_rate",
    "mean_abs_score_error",
    "phe_match_rate",
]


def main() -> None:
    """Выполнить эксперимент по влиянию SCALE и сохранить таблицу результатов."""
    from app.data import load_dataset, split_dataset

    logger.info("Запуск эксперимента по влиянию SCALE...")

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, _ = split_dataset(
        features=features,
        target=target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    baseline_pred, _ = plaintext_inference(model=model, x_test=x_test)

    scaler = model.named_steps["scaler"]
    w, b = extract_linear_params(model)
    x_scaled = scaler.transform(x_test)

    # Ошибка оценивается по линейному результату до сигмоиды,
    # чтобы не смешивать ошибку кодирования с нелинейной постобработкой.
    manual_scores = compute_manual_score(x=x_scaled, w=w, b=b)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []

    for scale in SCALE_VALUES:
        encoded_pred, _ = encoded_plaintext_inference(
            x_scaled=x_scaled,
            w=w,
            b=b,
            scale=scale,
            threshold=0.5,
        )

        encoded_scores = np.asarray(
            [
                encoded_plaintext_score(
                    x=sample,
                    w=w,
                    b=b,
                    scale=scale,
                )
                for sample in x_scaled
            ],
            dtype=np.float64,
        )

        encoded_match_rate = float(np.mean(encoded_pred == baseline_pred))
        mean_abs_score_error = float(np.mean(np.abs(encoded_scores - manual_scores)))

        client = Client(
            scaler=scaler,
            scale=scale,
            key_length=PHE_KEY_LENGTH,
        )
        server = Server(
            w_int=encode_weights(w=w, scale=scale),
            b_int=encode_bias(b=b, scale=scale),
            public_key=client.public_key,
        )

        x_subset = x_test.iloc[:PHE_SUBSET_SIZE]
        encoded_subset = encoded_pred[:PHE_SUBSET_SIZE]

        phe_pred, _ = phe_inference_batch(
            client=client,
            server=server,
            x_raw=x_subset,
        )
        phe_match_rate = float(np.mean(phe_pred == encoded_subset))

        rows.append(
            {
                "scale": scale,
                "encoded_match_rate": encoded_match_rate,
                "mean_abs_score_error": mean_abs_score_error,
                "phe_match_rate": phe_match_rate,
            }
        )

        logger.info(
            "SCALE=%d: доля совпадений кодированного режима=%.6f, "
            "средняя абсолютная ошибка=%.6f, доля совпадений PHE=%.6f",
            scale,
            encoded_match_rate,
            mean_abs_score_error,
            phe_match_rate,
        )

    with SCALE_CSV_PATH.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    scales = [int(row["scale"]) for row in rows]
    encoded_match_rates = [float(row["encoded_match_rate"]) for row in rows]
    mean_abs_errors = [float(row["mean_abs_score_error"]) for row in rows]

    fig, axis_left = plt.subplots(figsize=(9, 5))
    axis_right = axis_left.twinx()

    axis_left.plot(
        scales,
        encoded_match_rates,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="Доля совпадений",
    )
    axis_right.plot(
        scales,
        mean_abs_errors,
        marker="s",
        linestyle="--",
        linewidth=2,
        color="tab:red",
        label="Средняя абсолютная ошибка",
    )

    axis_left.set_title("Влияние коэффициента SCALE на ошибку кодирования")
    axis_left.set_xlabel("Коэффициент масштабирования SCALE")
    axis_left.set_ylabel("Доля совпадений")
    axis_right.set_ylabel("Средняя абсолютная ошибка линейного результата")
    axis_left.set_xscale("log")
    axis_left.set_xticks(scales)
    axis_left.set_xticklabels([str(scale) for scale in scales])
    axis_left.set_ylim(0.0, 1.02)
    axis_left.grid(True, alpha=0.3)

    left_handles, left_labels = axis_left.get_legend_handles_labels()
    right_handles, right_labels = axis_right.get_legend_handles_labels()
    axis_left.legend(left_handles + right_handles, left_labels + right_labels, loc="lower right")

    fig.tight_layout()
    fig.savefig(SCALE_PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    logger.info("Метрики сохранены в %s", SCALE_CSV_PATH)
    logger.info("График сохранён в %s", SCALE_PLOT_PATH)
    logger.info("Эксперимент по влиянию SCALE завершён.")


if __name__ == "__main__":
    main()
