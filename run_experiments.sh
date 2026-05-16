#!/usr/bin/env bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
set -euo pipefail

run_step() {
  local title="$1"
  local script="$2"

  echo "=== ${title} ==="
  python "$script"
}

run_step "Experiment 1: Learning the baseline model" "experiments/01_train_baseline.py"
run_step "Experiment 2: Checking manual plaintext" "experiments/02_validate_manual_inference.py"
run_step "Experiment 3: Encoded plaintext inference" "experiments/03_run_encoded_inference.py"
run_step "Experiment 4: PHE inference" "experiments/04_run_phe_inference.py"
run_step "Experiment 5: Time measurements (latency)" "experiments/05_benchmark_latency.py"
run_step "Experiment 6: Payload measurements" "experiments/06_benchmark_payload.py"
run_step "Experiment 7: Feature scaling" "experiments/07_benchmark_feature_scaling.py"
run_step "Experiment 8: Benchmark by datasets" "experiments/08_benchmark_datasets.py"
run_step "Experiment 9: Benchmark by key lengths" "experiments/09_benchmark_key_lengths.py"

# === Long-running / optional experiments ===
# To skip long steps, keep the lines below commented.
# To enable, uncomment the corresponding run_step lines.

run_step "Experiment 10: Benchmark by scale" "experiments/10_benchmark_scale.py"
# experiment 11 requires a running FastAPI server; run manually after starting uvicorn
# run_step "Experiment 11: API roundtrip benchmark" "experiments/11_benchmark_api_roundtrip.py"

echo "All experiments have been successfully completed."
echo "To run API roundtrip benchmark (experiment 11), first start the API:"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo "Then run:"
echo "  python experiments/11_benchmark_api_roundtrip.py"