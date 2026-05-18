# secure-ml-inference

**Конфиденциальный ML-инференс табличной модели с частично гомоморфным шифрованием**

Прототип демонстрирует защищённый ML-инференс линейных моделей на основе частично гомоморфного шифрования Paillier. Реализованы два сценария: бинарная классификация (LogisticRegression на Breast Cancer) и регрессия (Ridge на Diabetes). Сервер выполняет только зашифрованное линейное вычисление, клиент расшифровывает и выполняет постобработку.

Проект разработан в рамках выпускной квалификационной работы по теме конфиденциального машинного обучения. Основное внимание уделяется этапу online-инференса заранее обученной модели.

## Краткий обзор

- **classification**: `sklearn.datasets.load_breast_cancer`, `StandardScaler + LogisticRegression`
- **regression**: `sklearn.datasets.load_diabetes`, `StandardScaler + Ridge`
- **Шифрование**: Частично гомоморфное шифрование Paillier (`python-paillier`)
- **Протокол**: Клиент шифрует стандартизованные и кодированные в целые числа признаки; сервер вычисляет зашифрованный линейный score; клиент расшифровывает и применяет сигмоиду.
- **Реализация**: Core-логика на Python, серверное API на FastAPI, клиентское демо на Streamlit, все компоненты контейнеризованы через Docker.

## Цели

- Подтвердить, что качество классификации сохраняется при переходе от открытых данных к зашифрованным.
- Измерить вычислительные и сетевые издержки, связанные с применением частично гомоморфного шифрования.
- Создать воспроизводимый исследовательский прототип с чётким разделением ролей клиента и сервера.

## Архитектура (высокоуровневая)

```
Клиент (Streamlit UI)
  │
  ├─ Применяет StandardScaler
  ├─ Кодирует признаки в fixed-point целые
  ├─ Генерирует Paillier-ключи
  ├─ Шифрует кодированный вектор
  │
  └─ Отправляет Enc(x) ────► Сервер (FastAPI)
                              │
                              ├─ Хранит закодированные веса w_int и b_int
                              ├─ Вычисляет Enc(z) = Σ Enc(x_i)*w_i + Enc(b)
                              └─ Возвращает Enc(z)
Клиент:
  ├─ Расшифровывает z
  ├─ Вычисляет сигмоиду и порог
  └─ Получает предсказание класса
```

Подробнее см. [docs/architecture.md](docs/architecture.md).

## Структура проекта

Основные модули и артефакты:

- `app/` – ядро бизнес-логики (ML, кодирование, шифрование, инференс, метрики)
- `app/linear_scorer.py` – общий линейный scorer для classification/regression-сценариев
- `app/schemas.py` – схемы сценариев и метаданных моделей
- `api/` – серверное FastAPI-приложение
- `ui/` – клиентское Streamlit-приложение
- `experiments/` – скрипты для проведения и регистрации экспериментов
- `results/` – сгенерированные артефакты (модели, таблицы, графики)
- `results/tables/regression_quality_metrics.csv` – метрики качества regression-сценария
- `results/tables/api_roundtrip_metrics.csv` – метрики полного API roundtrip
- `docs/` – архитектурная документация, модель угроз, протокол экспериментов
- `docs/schemes/` – схемы и иллюстрации для документации

Детализированная иерархия описана в файле `STRUCTURE.md`.

## Быстрый старт (локально)

### Требования

- Python 3.10+
- pip

### Установка

```bash
git clone <repo-url>
cd secure-ml-inference
python -m venv venv
source venv/bin/activate   # или .\venv\Scripts\activate на Windows
pip install -r requirements.txt
```

### Обучение модели

```bash
python experiments/01_train_baseline.py
```

После выполнения в `results/models/` появятся `model.pkl`, `scaler.pkl`, `weights.json`, `metadata.json`.

### Запуск экспериментов

> ⚠️ Если запускать отдельные скрипты напрямую (`python experiments/...`), возможна ошибка импорта
> `ModuleNotFoundError: No module named 'app'`, если не задан `PYTHONPATH`.

Перед запуском отдельных экспериментов установите `PYTHONPATH` в корень репозитория:

- Linux/macOS:
  ```bash
  export PYTHONPATH="$PWD"
  ```
- Windows (PowerShell):
  ```powershell
  $env:PYTHONPATH = "."
  ```
- Windows (cmd.exe):
  ```bat
  set PYTHONPATH=%CD%
  ```

Рекомендуемый способ запуска — пакетные скрипты `run_experiments.sh` / `run_experiments.bat`: они уже выставляют `PYTHONPATH` автоматически.

Classification (Breast Cancer + LogisticRegression):

```bash
python experiments/01_train_baseline.py
python experiments/02_validate_manual_inference.py
python experiments/03_run_encoded_inference.py
python experiments/04_run_phe_inference.py
python experiments/05_benchmark_latency.py
python experiments/06_benchmark_payload.py
python experiments/07_benchmark_feature_scaling.py
python experiments/08_benchmark_datasets.py
python experiments/09_benchmark_key_lengths.py
python experiments/10_benchmark_scale.py
python experiments/11_benchmark_api_roundtrip.py
```

Regression (Diabetes + Ridge):

```bash
python experiments/12_train_regression_baseline.py
python experiments/13_run_phe_regression.py
```

Примечание для эксперимента 11: требуется предварительно запущенный сервер FastAPI. Эксперимент может занять несколько минут в зависимости от CPU и длины ключа. Поле `server_compute_ms` отражает только время гомоморфного вычисления на сервере, а полная задержка (включая сериализацию, передачу и клиентские этапы) измеряется в этом же эксперименте и сохраняется в `results/tables/api_roundtrip_metrics.csv`.

Альтернатива (запуск как модуля, также требует запуска из корня проекта):

```bash
python -m experiments.01_train_baseline
```

Результаты сохраняются в `results/tables/` и `results/plots/`.

Рекомендуемый пакетный запуск:

- Windows:
  ```bat
  run_experiments.bat
  ```
- Linux/macOS (если в репозитории добавлен/добавляется shell-скрипт):
  ```bash
  ./run_experiments.sh
  ```

### Запуск двух сценариев (classification / regression)

Система поддерживает два `scenario_id`, синхронизированных между API и UI:

- `classification` — бинарная классификация Breast Cancer (артефакты: `results/models/model.pkl`, `results/models/scaler.pkl`, `results/models/weights.json`, `results/models/metadata.json`).
- `regression` — регрессия Diabetes (артефакты: `results/models/regression_model.pkl`, `results/models/regression_scaler.pkl`, `results/models/regression_weights.json`, `results/models/regression_metadata.json`).

Запуск regression-сценария:

```bash
python experiments/12_train_regression_baseline.py
python experiments/13_run_phe_regression.py
```

Проверка итоговых метрик regression-сценария:

```bash
cat results/tables/regression_quality_metrics.csv
```

### Обновлённый pipeline экспериментов (01–13)

1. `01_train_baseline.py`
2. `02_validate_manual_inference.py`
3. `03_run_encoded_inference.py`
4. `04_run_phe_inference.py`
5. `05_benchmark_latency.py`
6. `06_benchmark_payload.py`
7. `07_benchmark_feature_scaling.py`
8. `08_benchmark_datasets.py`
9. `09_benchmark_key_lengths.py`
10. `10_benchmark_scale.py`
11. `11_benchmark_api_roundtrip.py`
12. `12_train_regression_baseline.py`
13. `13_run_phe_regression.py`

Для полного запуска используйте `run_experiments.sh` / `run_experiments.bat`; они включают шаги 01–13 и настраивают `PYTHONPATH`.

### Unit-тесты и инструменты разработки

Перед запуском тестов и линтеров установите dev-зависимости:

```bash
pip install -e ".[dev]"
```

После этого доступны команды:

```bash
ruff format .
ruff check .
mypy .
pytest
```

Пример успешного запуска тестов: `17 passed`.

### Запуск API и UI по отдельности

#### API (FastAPI)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Документация: http://localhost:8000/docs

#### UI (Streamlit)

```bash
streamlit run ui/streamlit_app.py --server.port 8501
```

UI — одностраничное демо-приложение для ввода признаков, запуска зашифрованного инференса и просмотра результата предсказания. В демонстрационном интерфейсе клиент дополнительно загружает baseline-модель исключительно для визуального сравнения; это не часть защищённого протокола. Модельные веса остаются на сервере.

## Запуск через Docker

```bash
docker compose up --build
```

Compose поднимает два сервиса:
- `api` — FastAPI (`uvicorn api.main:app`) на порту `8000`
- `ui` — Streamlit (`ui/streamlit_app.py`) на порту `8501`, запускается после `api` (`depends_on`)

Проверка доступности:
- API health: http://localhost:8000/health
- Swagger UI: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

## Учебная reference-реализация Paillier

Для учебного разбора алгоритма и тестируемой минимальной реализации см.:

- [docs/paillier_algorithm.md](docs/paillier_algorithm.md)
- [app/paillier_reference.py](app/paillier_reference.py)
- [tests/test_paillier_reference.py](tests/test_paillier_reference.py)

## Модель угроз

Сервер считается честным, но любопытным (honest-but-curious). Он выполняет вычисления корректно, но может пытаться извлечь значения признаков из доступных ему данных: зашифрованных векторов, открытого ключа, параметров модели.

Подробно — [docs/threat_model.md](docs/threat_model.md).

## Ограничения

- Прототип является исследовательским и не предназначен для использования в production-среде.
- Защищается только этап инференса; обучение модели выполняется в открытом виде.
- Не рассматриваются атаки по сторонним каналам, отказ в обслуживании, кража закрытого ключа.
- Защита канала передачи данных не входит в задачу (подразумевается TLS при реальной эксплуатации).
- Используется 1024-битный ключ Paillier, который не считается криптографически стойким для высокозащищённых систем.

## Лицензия

MIT License. См. [LICENSE](LICENSE).
