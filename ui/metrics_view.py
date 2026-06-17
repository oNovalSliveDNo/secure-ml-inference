"""Metrics views for sample-level and aggregate experiment evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components import render_metric_card


def _get_table_explanation(csv_name: str, df: pd.DataFrame) -> str:
    """Return a concise Russian explanation for metrics table."""
    lower_name = csv_name.lower()
    numeric_df = df.select_dtypes(include="number")
    if "accuracy" in lower_name or "acc" in lower_name:
        if numeric_df.shape[1] >= 2:
            delta = abs(float(numeric_df.iloc[:, 0].mean()) - float(numeric_df.iloc[:, 1].mean()))
            if delta < 1e-6:
                return "Точность модели не изменилась при переходе к защищённому режиму."
        return "Таблица показывает, что качество обычного и защищённого расчёта сопоставимо."
    if "latency" in lower_name or "time" in lower_name or "timing" in lower_name:
        return "Таблица показывает, сколько времени занимает каждый этап работы системы."
    if "overhead" in lower_name or "payload" in lower_name:
        return "Таблица показывает, насколько увеличивается объём передаваемых данных из-за шифрования."
    if not numeric_df.empty:
        return "Таблица содержит численные результаты экспериментов."
    return "Таблица содержит дополнительные сведения по экспериментам."


def _get_plot_explanation(plot_name: str) -> str:
    """Return a concise Russian explanation for plot."""
    lower_name = plot_name.lower()
    if "latency" in lower_name or "time" in lower_name or "timing" in lower_name:
        return "График показывает, как меняется время работы системы при увеличении нагрузки."
    if "feature" in lower_name or "dim" in lower_name:
        return "График показывает, как число признаков влияет на время работы и размер данных."
    if "accuracy" in lower_name or "auc" in lower_name:
        return "График показывает, что качество модели сохраняется в защищённом режиме."
    if "payload" in lower_name or "size" in lower_name or "overhead" in lower_name:
        return "График показывает рост объёма передаваемых данных из-за шифрования."
    return "График показывает экспериментальные результаты в наглядном виде."


def _get_artifact_title(file_name: str) -> str:
    """Return a readable Russian title for result files."""
    lower_name = file_name.lower()
    title_by_keyword = {
        "quality": "качество предсказаний",
        "accuracy": "качество предсказаний",
        "latency": "время выполнения",
        "time": "время выполнения",
        "payload": "размер передаваемых данных",
        "overhead": "дополнительные затраты",
        "feature": "влияние числа признаков",
        "dataset": "сравнение наборов данных",
        "key": "влияние длины ключа",
        "scale": "влияние масштаба кодирования",
        "roundtrip": "полный запрос через сервер",
        "regression": "результаты регрессии",
        "classification": "результаты классификации",
    }
    for keyword, title in title_by_keyword.items():
        if keyword in lower_name:
            return title
    return file_name


_COLUMN_TRANSLATIONS = {
    "method": "Метод",
    "mode": "Режим",
    "scenario": "Тип задачи",
    "task": "Тип задачи",
    "dataset": "Набор данных",
    "stage": "Этап",
    "metric": "Метрика",
    "value": "Значение",
    "mean": "Среднее значение",
    "std": "Стандартное отклонение",
    "min": "Минимум",
    "max": "Максимум",
    "mean_ms": "Среднее время, мс",
    "std_ms": "Отклонение, мс",
    "median_ms": "Медианное время, мс",
    "total_ms": "Общее время, мс",
    "encryption_ms": "Время шифрования, мс",
    "decryption_ms": "Время расшифровки, мс",
    "server_compute_ms": "Время расчёта на сервере, мс",
    "http_elapsed_ms": "Полное время запроса, мс",
    "accuracy": "Доля верных ответов",
    "precision": "Точность положительного класса",
    "recall": "Полнота",
    "f1": "F1-мера",
    "roc_auc": "ROC-AUC",
    "match_rate": "Доля совпадений",
    "mae": "Средняя абсолютная ошибка",
    "mse": "Среднеквадратичная ошибка",
    "rmse": "Корень из среднеквадратичной ошибки",
    "r2": "Коэффициент детерминации R²",
    "plaintext_bytes": "Размер обычного запроса, байт",
    "encrypted_bytes": "Размер зашифрованного запроса, байт",
    "payload_bytes": "Размер данных, байт",
    "request_bytes": "Размер запроса, байт",
    "response_bytes": "Размер ответа, байт",
    "overhead_ratio": "Увеличение размера",
    "feature_count": "Число признаков",
    "key_length": "Длина ключа",
    "scale": "Масштаб кодирования",
    "sample_count": "Число объектов",
}
_VALUE_TRANSLATIONS = {
    "baseline": "Обычная модель без защиты",
    "plaintext": "Обычный расчёт",
    "plaintext_baseline": "Обычная модель без защиты",
    "encoded": "Открытый расчёт после кодирования",
    "encoded_plaintext": "Открытый расчёт после кодирования",
    "phe": "Защищённый расчёт",
    "phe_inference": "Защищённый расчёт по зашифрованным данным",
    "encrypted": "Зашифрованный режим",
    "classification": "Классификация",
    "regression": "Регрессия",
    "keygen": "Генерация ключей",
    "key_generation": "Генерация ключей",
    "scaling": "Масштабирование признаков",
    "encoding": "Кодирование признаков",
    "encryption": "Шифрование признаков",
    "server_compute": "Расчёт на сервере",
    "decryption": "Расшифровка результата",
    "postprocessing": "Получение итогового прогноза",
    "total": "Общее время",
    "total_without_keygen": "Общее время без генерации ключей",
}


def _prepare_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Translate common experiment table labels for the Streamlit page."""
    display_df = df.rename(
        columns={column: _COLUMN_TRANSLATIONS.get(column, column) for column in df.columns}
    ).copy()
    for column in display_df.select_dtypes(include="object").columns:
        display_df[column] = display_df[column].map(
            lambda value: _VALUE_TRANSLATIONS.get(value, value) if isinstance(value, str) else value
        )
    return display_df


def render_sample_level_metrics(result: dict[str, Any], scenario_id: str) -> None:
    """Render metrics for the currently selected sample."""
    k1, k2, k3, k4, k5 = st.columns(5)
    if "overhead_ratio" not in result:
        for col, label in zip(
            [k1, k2, k3, k4, k5],
            [
                "Сервер видит исходные признаки",
                "Совпадение с обычной моделью",
                "Увеличение размера запроса",
                "Зашифрованный запрос",
                "Обычный запрос",
            ],
            strict=True,
        ):
            with col:
                render_metric_card(label, "НЕТ" if col == k1 else "—")
        return
    with k1:
        render_metric_card("Сервер видит исходные признаки", "НЕТ")
    if scenario_id == "classification":
        match_label = "ДА" if result.get("pred_secure") == result.get("pred_baseline") else "НЕТ"
        with k2:
            render_metric_card("Совпадение с обычной моделью", match_label)
    else:
        delta = abs(float(result["pred_secure"]) - float(result["pred_baseline"]))
        with k2:
            render_metric_card(
                "Прогноз близок к обычной модели",
                "ДА" if delta < 0.01 else "НЕТ",
                f"Δ = {delta:.4f}",
            )
    with k3:
        render_metric_card("Увеличение размера запроса", f"{result['overhead_ratio']:.2f}x")
    with k4:
        render_metric_card("Зашифрованный запрос", f"{result['encrypted_bytes']} байт")
    with k5:
        render_metric_card("Обычный запрос", f"{result['plaintext_bytes']} байт")
    if scenario_id == "regression":
        delta_baseline = abs(float(result["pred_secure"]) - float(result["pred_baseline"]))
        delta_encoded = abs(float(result["pred_secure"]) - float(result["pred_encoded"]))
        r1, r2 = st.columns(2)
        with r1:
            render_metric_card("Разность защищённого и обычного прогноза", f"{delta_baseline:.6f}")
        with r2:
            render_metric_card(
                "Разность защищённого и кодированного прогноза", f"{delta_encoded:.6f}"
            )


def show_metrics_dashboard(tables_dir: Path, plots_dir: Path) -> None:
    """Render aggregate-level experiment evidence with explanatory comments."""
    st.header("Результаты экспериментов")
    csv_files = sorted(tables_dir.glob("*.csv"))
    plot_files = sorted(plots_dir.glob("*.png"))
    if not csv_files and not plot_files:
        st.info(
            "Результаты экспериментов не найдены. Запустите сценарии в директории experiments/, чтобы сформировать таблицы и графики."
        )
        return
    if not csv_files:
        st.info(
            "Файлы с таблицами метрик отсутствуют в results/tables/. Запустите эксперименты для формирования табличных результатов."
        )
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            st.warning(f"Не удалось прочитать {csv_file.name}: {exc}")
            continue
        st.subheader(f"Таблица: {_get_artifact_title(csv_file.name)}")
        st.caption(f"Файл с результатами: `{csv_file.name}`")
        st.dataframe(_prepare_display_dataframe(df), width="stretch")
        if csv_file.name == "latency_metrics.csv" and {"stage", "mean_ms"}.issubset(df.columns):
            totals = df.set_index("stage")["mean_ms"].to_dict()
            if totals.get("total") is not None and totals.get("total_without_keygen") is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Общее время с генерацией ключей, мс", f"{totals['total']:.2f}")
                with col2:
                    st.metric(
                        "Общее время без генерации ключей, мс",
                        f"{totals['total_without_keygen']:.2f}",
                    )
        st.markdown(f"**Интерпретация:** {_get_table_explanation(csv_file.name, df)}")
    if not plot_files:
        st.info(
            "Файлы с графиками отсутствуют в results/plots/. Запустите эксперименты визуализации результатов."
        )
    for plot_file in plot_files:
        st.subheader(f"График: {_get_artifact_title(plot_file.name)}")
        st.caption(f"Файл с графиком: `{plot_file.name}`")
        st.image(str(plot_file), width="stretch")
        st.markdown(f"**Вывод:** {_get_plot_explanation(plot_file.name)}")
