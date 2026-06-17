"""Metrics views for sample-level and aggregate experiment evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components import render_metric_card

DEFAULT_FIDELITY_TOLERANCE = 0.01


def _to_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Any, digits: int = 6) -> str:
    """Format a numeric value for compact metric display."""
    number = _to_float(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def _format_bytes(value: Any) -> str:
    """Format a byte count for metric display."""
    number = _to_float(value)
    if number is None:
        return "—"
    return f"{int(number)} байт"


def _format_ms(value: Any) -> str:
    """Format milliseconds for metric display."""
    number = _to_float(value)
    if number is None:
        return "—"
    return f"{number:.2f} мс"


def classify_sample_error_status(
    abs_error: float | None, median_abs_error: float | None, p90_abs_error: float | None
) -> tuple[str, str]:
    """Classify selected regression-sample baseline error against baseline-error distribution."""
    if abs_error is None or median_abs_error is None or p90_abs_error is None:
        return "Недостаточно данных", "off"
    if abs_error <= median_abs_error:
        return "Ошибка не выше медианной", "normal"
    if abs_error <= p90_abs_error:
        return "Ошибка между медианой и p90", "warning"
    return "Ошибка выше p90", "critical"


def classify_fidelity_status(max_delta: float | None, tolerance: float | None) -> tuple[str, str]:
    """Classify encoding/PHE fidelity against a numeric tolerance."""
    if max_delta is None or tolerance is None:
        return "Недостаточно данных", "off"
    if max_delta <= tolerance:
        return "В пределах допуска", "normal"
    return "Выше допуска", "critical"


def _render_status(label: str, status: tuple[str, str]) -> None:
    """Render a status line without implying model quality loss."""
    status_text, level = status
    icon = {"normal": "✅", "warning": "⚠️", "critical": "❌"}.get(level, "ℹ️")
    st.markdown(f"**{label}:** {icon} {status_text}")


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
    """Render sample-level evidence split into model quality, fidelity, and privacy/overhead."""
    st.markdown("### Sample-level метрики")
    if "overhead_ratio" not in result:
        st.info("Итоговые sample-level метрики появятся после выполнения всех шагов протокола.")
        return
    baseline_value = result.get("pred_baseline", result.get("baseline_value"))
    encoded_value = result.get("pred_encoded", result.get("z_encoded"))
    secure_value = result.get("pred_secure", result.get("z_secure"))
    true_value = result.get("true_label")

    with st.container(border=True):
        st.subheader("1. Качество исходной ML-модели")
        if scenario_id == "classification":
            true_class = int(true_value) if true_value is not None else "—"
            baseline_class = int(baseline_value) if baseline_value is not None else "—"
            probability = result.get("prob_baseline")
            correctness = (
                "ДА"
                if true_value is not None
                and baseline_value is not None
                and int(true_value) == int(baseline_value)
                else "НЕТ"
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Истинный класс", true_class)
            with c2:
                render_metric_card("Класс baseline", baseline_class)
            with c3:
                render_metric_card("Вероятность baseline", _format_number(probability, digits=4))
            with c4:
                render_metric_card("Baseline корректен", correctness)
        else:
            true_number = _to_float(true_value)
            baseline_number = _to_float(baseline_value)
            abs_error = (
                abs(baseline_number - true_number)
                if baseline_number is not None and true_number is not None
                else None
            )
            relative_error = (
                abs_error / abs(true_number)
                if abs_error is not None and true_number is not None and true_number != 0.0
                else None
            )
            median_abs_error = _to_float(result.get("baseline_median_abs_error"))
            p90_abs_error = _to_float(result.get("baseline_p90_abs_error"))
            error_percentile = _to_float(result.get("baseline_error_percentile"))
            _render_status(
                "Статус ошибки baseline",
                classify_sample_error_status(abs_error, median_abs_error, p90_abs_error),
            )
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                render_metric_card("Истинное значение", _format_number(true_number, digits=4))
            with r2:
                render_metric_card("Baseline prediction", _format_number(baseline_number, digits=4))
            with r3:
                render_metric_card("Абсолютная ошибка", _format_number(abs_error, digits=4))
            with r4:
                render_metric_card(
                    "Относительная ошибка",
                    "—" if relative_error is None else f"{relative_error:.2%}",
                )
            r5, r6, r7 = st.columns(3)
            with r5:
                render_metric_card("Median abs error", _format_number(median_abs_error, digits=4))
            with r6:
                render_metric_card("p90 abs error", _format_number(p90_abs_error, digits=4))
            with r7:
                percentile_value = "—" if error_percentile is None else f"{error_percentile:.1f}%"
                render_metric_card("Percentile ошибки объекта", percentile_value)

    with st.container(border=True):
        st.subheader("2. Влияние кодирования и шифрования")
        fidelity_baseline_value = (
            result.get("prob_baseline") if scenario_id == "classification" else baseline_value
        )
        fidelity_encoded_value = (
            result.get("prob_encoded") if scenario_id == "classification" else encoded_value
        )
        fidelity_secure_value = (
            result.get("prob_secure") if scenario_id == "classification" else secure_value
        )
        baseline_number = _to_float(fidelity_baseline_value)
        encoded_number = _to_float(fidelity_encoded_value)
        secure_number = _to_float(fidelity_secure_value)
        delta_encoded_baseline = (
            abs(encoded_number - baseline_number)
            if encoded_number is not None and baseline_number is not None
            else None
        )
        delta_secure_baseline = (
            abs(secure_number - baseline_number)
            if secure_number is not None and baseline_number is not None
            else None
        )
        delta_secure_encoded = (
            abs(secure_number - encoded_number)
            if secure_number is not None and encoded_number is not None
            else None
        )
        tolerance = _to_float(result.get("fidelity_tolerance", DEFAULT_FIDELITY_TOLERANCE))
        deltas = [
            d
            for d in (delta_encoded_baseline, delta_secure_baseline, delta_secure_encoded)
            if d is not None
        ]
        max_delta = max(deltas) if deltas else None
        margin = tolerance - max_delta if tolerance is not None and max_delta is not None else None
        _render_status("Статус fidelity", classify_fidelity_status(max_delta, tolerance))
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            render_metric_card("Baseline", _format_number(baseline_number, digits=6))
        with f2:
            render_metric_card("Encoded plaintext", _format_number(encoded_number, digits=6))
        with f3:
            render_metric_card("PHE prediction", _format_number(secure_number, digits=6))
        with f4:
            render_metric_card("Tolerance", _format_number(tolerance, digits=6))
        f5, f6, f7, f8 = st.columns(4)
        with f5:
            render_metric_card(
                "Δ encoded vs baseline", _format_number(delta_encoded_baseline, digits=6)
            )
        with f6:
            render_metric_card("Δ PHE vs baseline", _format_number(delta_secure_baseline, digits=6))
        with f7:
            render_metric_card("Δ PHE vs encoded", _format_number(delta_secure_encoded, digits=6))
        with f8:
            render_metric_card("Запас до tolerance", _format_number(margin, digits=6))

    with st.container(border=True):
        st.subheader("3. Конфиденциальность и накладные расходы")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_metric_card("Сервер видит исходные признаки", "НЕТ")
        with p2:
            render_metric_card("Закрытый ключ передан серверу", "НЕТ")
        with p3:
            render_metric_card("Запрос зашифрован", "ДА")
        with p4:
            render_metric_card(
                "Overhead ratio",
                f"{_to_float(result.get('overhead_ratio')):.2f}x"
                if _to_float(result.get("overhead_ratio")) is not None
                else "—",
            )
        p5, p6, p7, p8 = st.columns(4)
        with p5:
            render_metric_card(
                "Plaintext request size", _format_bytes(result.get("plaintext_bytes"))
            )
        with p6:
            render_metric_card(
                "Encrypted request size", _format_bytes(result.get("encrypted_bytes"))
            )
        with p7:
            render_metric_card("Encryption time", _format_ms(result.get("encrypt_ms")))
        with p8:
            render_metric_card("Server compute time", _format_ms(result.get("server_compute_ms")))
        p9, p10 = st.columns(2)
        with p9:
            render_metric_card("Decryption time", _format_ms(result.get("decrypt_ms")))
        with p10:
            render_metric_card("HTTP roundtrip", _format_ms(result.get("http_elapsed_ms")))


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
