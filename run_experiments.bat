@echo off
setlocal enabledelayedexpansion
set PYTHONPATH=%CD%;%PYTHONPATH%

REM ========== OFFLINE EXPERIMENTS ==========
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

echo === Experiment 10: Benchmark by scale ===
python experiments/10_benchmark_scale.py
if %errorlevel% neq 0 exit /b %errorlevel%

REM ========== API ROUNDTRIP (requires running server) ==========
echo === Experiment 11: API roundtrip benchmark ===
echo Starting API server in background...
start "SecureML-API" /MIN cmd /c "uvicorn api.main:app --host 0.0.0.0 --port 8000"
REM Wait for server to be ready (using PowerShell for compatibility)
:wait_api
powershell -Command "try { $response = Invoke-WebRequest -Uri http://localhost:8000/docs -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_api
)
echo API server is ready.

python experiments/11_benchmark_api_roundtrip.py
if %errorlevel% neq 0 (
    echo Error in Experiment 11, stopping server...
    taskkill /FI "WINDOWTITLE eq SecureML-API*" /F >nul 2>&1
    exit /b %errorlevel%
)

echo Experiment 11 completed successfully. Stopping API server...
taskkill /FI "WINDOWTITLE eq SecureML-API*" /F >nul 2>&1
if errorlevel 1 (
    REM Fallback if taskkill by title fails
    taskkill /F /IM uvicorn.exe >nul 2>&1
)
echo API server stopped.

REM ========== REGRESSION EXPERIMENTS ==========
echo === Experiment 12: Train regression baseline ===
python experiments/12_train_regression_baseline.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo === Experiment 13: PHE regression ===
python experiments/13_run_phe_regression.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo All experiments have been successfully completed.
exit /b 0