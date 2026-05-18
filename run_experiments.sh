#!/usr/bin/env bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
set -euo pipefail

# ------------------- вспомогательная функция -------------------
run_step() {
    local title="$1"
    local script="$2"
    echo "=== ${title} ==="
    python "$script"
}

# ------------------- офлайн-эксперименты -------------------
run_step "Experiment 0: Save environment info"           "experiments/00_environment_info.py"
run_step "Experiment 1: Learning the baseline model"     "experiments/01_train_baseline.py"
run_step "Experiment 2: Checking manual plaintext"       "experiments/02_validate_manual_inference.py"
run_step "Experiment 3: Encoded plaintext inference"     "experiments/03_run_encoded_inference.py"
run_step "Experiment 4: PHE inference"                   "experiments/04_run_phe_inference.py"
run_step "Experiment 5: Time measurements (latency)"     "experiments/05_benchmark_latency.py"
run_step "Experiment 6: Payload measurements"            "experiments/06_benchmark_payload.py"
run_step "Experiment 7: Feature scaling"                 "experiments/07_benchmark_feature_scaling.py"
run_step "Experiment 8: Benchmark by datasets"           "experiments/08_benchmark_datasets.py"
run_step "Experiment 9: Benchmark by key lengths"        "experiments/09_benchmark_key_lengths.py"
run_step "Experiment 10: Benchmark by scale"             "experiments/10_benchmark_scale.py"

# ------------------- Experiment 11: API roundtrip (требует сервер) -------------------
echo "=== Experiment 11: API roundtrip benchmark ==="

echo "Starting API server in background..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Ждём, пока сервер станет доступен
echo "Waiting for API server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/docs >/dev/null 2>&1; then
        echo "API server is ready."
        break
    fi
    sleep 1
done

# Проверка, что сервер поднялся
if ! curl -s http://localhost:8000/docs >/dev/null 2>&1; then
    echo "ERROR: API server did not start within 30 seconds."
    kill "$API_PID" 2>/dev/null || true
    exit 1
fi

# Запуск эксперимента 11 (set -e отключен для этого блока, чтобы самим обработать ошибку)
set +e
python experiments/11_benchmark_api_roundtrip.py
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "Error in Experiment 11, stopping server..."
    kill "$API_PID" 2>/dev/null || true
    exit "$EXIT_CODE"
fi

echo "Experiment 11 completed successfully. Stopping API server..."
kill "$API_PID" 2>/dev/null || true
# Дадим процессу время завершиться
sleep 1

# ------------------- регрессионные эксперименты -------------------
run_step "Experiment 12: Train regression baseline"     "experiments/12_train_regression_baseline.py"
run_step "Experiment 13: PHE regression"                "experiments/13_run_phe_regression.py"

echo ""
echo "All experiments have been successfully completed."
exit 0