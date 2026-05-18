@echo off
set PYTHONPATH=%CD%;%PYTHONPATH%

echo === Experiment 0: Save environment info ===
python experiments/00_environment_info.py
if %errorlevel% neq 0 exit /b %errorlevel%

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

echo === Experiment 10: Benchmark by scale ===
python experiments/10_benchmark_scale.py
if %errorlevel% neq 0 exit /b %errorlevel%

REM experiment 11 requires a running FastAPI server; run manually after starting uvicorn
REM echo === Experiment 11: API roundtrip benchmark ===
REM python experiments/11_benchmark_api_roundtrip.py
REM if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 12: Train regression baseline ===
python experiments/12_train_regression_baseline.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 13: PHE regression ===
python experiments/13_run_phe_regression.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo All experiments have been successfully completed.
echo To run API roundtrip benchmark (experiment 11), first start the API:
echo   uvicorn api.main:app --host 0.0.0.0 --port 8000
echo Then run:
echo   python experiments/11_benchmark_api_roundtrip.py
echo Regression experiments:
echo   12) python experiments/12_train_regression_baseline.py
echo   13) python experiments/13_run_phe_regression.py