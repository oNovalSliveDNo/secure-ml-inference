@echo off

echo === Experiment 1: Learning the baseline model ===
python experiments/01_train_baseline.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 2: Checking manual plaintext ===
python experiments/02_validate_manual_inference.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 3: Encoded plaintext inference ===
python experiments/03_run_encoded_inference.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 4: PHE inference ===
python experiments/04_run_phe_inference.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 5: Time measurements (latency) ===
python experiments/05_benchmark_latency.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 6: Payload measurements ===
python experiments/06_benchmark_payload.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 7: Feature scaling ===
python experiments/07_benchmark_feature_scaling.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 8: Benchmark by datasets ===
python experiments/08_benchmark_datasets.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 9: Benchmark by key lengths ===
python experiments/09_benchmark_key_lengths.py
if %errorlevel% neq 0 exit /b %errorlevel%

REM === Long-running / optional experiments ===
REM To skip long steps, leave this block commented.
REM To enable, remove leading REM from each line in the corresponding step.

REM echo === Experiment 10: Benchmark by scale (optional, long) ===
REM python experiments/10_benchmark_scale.py
REM if %errorlevel% neq 0 exit /b %errorlevel%

REM echo === Experiment 11: API roundtrip benchmark (optional, long) ===
REM python experiments/11_benchmark_api_roundtrip.py
REM if %errorlevel% neq 0 exit /b %errorlevel%

echo All experiments have been successfully completed.