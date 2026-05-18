# Project Structure

This file provides a map of the repository to help quickly understand the codebase, locate relevant files, and follow architectural conventions.

## Quick Entry Points

- **Configuration**: `app/config.py`
- **ML Logic**: `app/model.py`, `app/inference.py`
- **Cryptography**: `app/crypto.py`
- **Server API**: `api/main.py`
- **Client UI**: `ui/streamlit_app.py`
- **Experiments**: `experiments/` (numbered scripts)

## Root Directory Layout

```
secure-ml-inference/
│
├── .gitignore                      # Git ignore rules (e.g., __pycache__, venv, .env, results/models/, etc.)
├── .env.example                    # Example environment variables (optional, for demonstration)
├── Dockerfile                      # Single-image build or dedicated API service (see docker-compose)
├── docker-compose.yml              # Service definitions: api (FastAPI) and ui (Streamlit) with network and volumes
├── requirements.txt                # Core Python dependencies for the whole project
├── README.md                       # Project overview, threats, architecture, run instructions, and limitations (written last)
│
├── app/                            # Core business logic (ML, cryptography, utilities)
│   ├── __init__.py                 # Package initialisation
│   ├── config.py                   # Configuration constants: RANDOM_STATE, TEST_SIZE, SCALE, KEY_LENGTH, THRESHOLD, FEATURE_COUNTS, etc.
│   ├── data.py                     # Dataset loading (load_breast_cancer), train/test split, feature name extraction
│   ├── model.py                    # Baseline model training (Pipeline: StandardScaler + LogisticRegression), model/scaler persistence, weight extraction (coef_, intercept_)
│   ├── linear_scorer.py            # Universal linear scorer for classification/regression using extracted weights and bias
│   ├── encoding.py                 # Fixed‑point encoding/decoding: encode_vector, encode_weights, encode_bias, decode_score
│   ├── crypto.py                   # Paillier wrapper: key generation, vector encryption, score decryption, ciphertext serialisation/deserialisation, size estimation
│   ├── client.py                   # Client class encapsulating client‑side logic: scaling, encoding, key generation, encryption, decryption, sigmoid/threshold
│   ├── server.py                   # Server class encapsulating server‑side logic: storing encoded weights, receiving Enc(x), computing Enc(score) = Σ Enc(x_i)*w_i + Enc(b)
│   ├── inference.py                # Functions for four inference modes: plaintext baseline, manual plaintext (via weights), encoded plaintext, PHE inference for single/batch objects
│   ├── metrics.py                  # Utilities for quality metrics (accuracy, precision, recall, F1, ROC‑AUC, match rate, score error) and time/size measurements
│   └── schemas.py                  # Pydantic models for API requests/responses (e.g., EncryptedInferRequest, EncryptedInferResponse), shared with FastAPI
│
├── api/                            # Server side (FastAPI), exposing inference API
│   ├── __init__.py
│   └── main.py                     # FastAPI entry point: creates app, loads model and weights on startup, handles GET /health, GET /model/info, POST /infer/encrypted (and optionally /infer/plaintext-demo)
│
├── ui/                             # Client demo application (Streamlit)
│   └── streamlit_app.py            # Multi‑page Streamlit app with tabs:
│                                   #   - Demo inference (select sample, run secure inference, compare)
│                                   #   - Protocol view (visualisation of client and server knowledge)
│                                   #   - Metrics dashboard (read CSV and plots from results/)
│                                   #   - Architecture (interaction diagram)
│
├── experiments/                    # Reproducible experiment scripts
│   ├── 00_environment_info.py      # Capture Python, package, platform, CPU, and runtime environment details to environment.json
│   ├── 01_train_baseline.py        # Train classification baseline, save model.pkl, scaler.pkl, weights.json, metadata.json to results/models/
│   ├── 02_validate_manual_inference.py  # Verify manual classification inference matches predict_proba, output errors and statistics
│   ├── 03_run_encoded_inference.py # Evaluate classification quality with fixed‑point encoding (no encryption), append row to quality_metrics.csv
│   ├── 04_run_phe_inference.py     # Run PHE classification inference on test set (or subset), append results to quality_metrics.csv│   ├── 05_benchmark_latency.py     # Measure timing: preprocessing, encoding, key gen, encryption, server compute, decryption, total, write to latency_metrics.csv
│   ├── 06_benchmark_payload.py     # Estimate plaintext/encrypted request/response sizes, expansion factors, write to payload_metrics.csv
│   ├── 07_benchmark_feature_scaling.py  # Measure time & size vs feature count (5,10,20,30), write to feature_scaling_metrics.csv and generate plots
│   ├── 08_benchmark_datasets.py    # Benchmark classification inference across datasets, write to dataset_benchmark_metrics.csv
│   ├── 09_benchmark_key_lengths.py # Benchmark PHE key length impact, write to key_length_metrics.csv
│   ├── 10_benchmark_scale.py       # Benchmark fixed‑point scale impact, write to scale_metrics.csv
│   ├── 11_benchmark_api_roundtrip.py  # Benchmark end-to-end encrypted API round trips, write to api_roundtrip_metrics.csv
│   ├── 12_train_regression_baseline.py  # Train regression baseline and save regression artifacts to results/models/
│   └── 13_run_phe_regression.py    # Run encoded/PHE regression inference and write regression_quality_metrics.csv
│
├── results/                        # Artifact storage (models, numeric results, plots)
│   ├── models/                     # Serialised model, scaler, JSON weights, etc.
│   ├── tables/                     # CSV/JSON metric files
│   │   ├── environment.json
│   │   ├── quality_metrics.csv
│   │   ├── regression_quality_metrics.csv
│   │   ├── latency_metrics.csv
│   │   ├── payload_metrics.csv
│   │   ├── feature_scaling_metrics.csv
│   │   ├── dataset_benchmark_metrics.csv
│   │   ├── key_length_metrics.csv
│   │   ├── scale_metrics.csv
│   │   └── api_roundtrip_metrics.csv
│   └── plots/                      # PNG plots: feature_count_vs_total_time.png, feature_count_vs_request_size.png, feature_count_vs_server_time.png
│
├── docs/                           # Project documentation (for thesis/dissertation)
│   ├── architecture.md             # Detailed architecture: components, data flows, roles, protocol steps
│   ├── threat_model.md             # Threat model, trust assumptions, protection boundaries (honest‑but‑curious server)
│   ├── experiment_protocol.md      # Experiment methodology: parameters, metrics, success criteria, interpretation
│   └── schemes/                    # Generated protocol and architecture diagrams
│       ├── protocol_flow.png
│       ├── plaintext_vs_encoded_vs_phe.png
│       ├── math_flow.png
│       └── threat_model.png
│
└── tests/                          # Pytest suite (22+ tests expected) covering encoding, crypto, inference, client/server, and API behavior
```

### Key Component Explanations

- **Root build & configuration files** (`Dockerfile`, `docker-compose.yml`, `requirements.txt`) guarantee environment reproducibility. The Docker image uses `python:3.10-slim`, installs dependencies, and launches `uvicorn` for `api` and `streamlit` for `ui` as separate services on a shared network.

- **`app/` – pure business logic, independent of web frameworks.** This keeps the core testable and reusable. Modules have clearly separated responsibilities:
  - `model.py` only handles training and persistence of a LogisticRegression pipeline.
  - `crypto.py` only deals with Paillier encryption/decryption.
  - `client.py` / `server.py` implement the protocol at the cell level.
  - `inference.py` combines these components into four scenarios.

- **`api/main.py`** – a thin FastAPI layer that loads model weights at startup (via `app/model.py`) and uses the `Server` class from `app/server.py`. The `/infer/encrypted` endpoint accepts data in the `schemas.EncryptedInferRequest` format, performs computation, and returns an `EncryptedInferResponse`. No direct handling of raw features.

- **`ui/streamlit_app.py`** – a “fat” client scenario. It imports `app/client.py`, `app/config.py`, `app/encoding.py`, `app/crypto.py`, and sends HTTP requests to the API. Client logic is completely separated from the server. Users can observe every data transformation step, ciphertext sizes, and the final comparison – a key demonstration for the thesis defence.

- **`experiments/`** – a set of self‑contained executable scripts. They use functions from `app/` and save results to `results/`. They run sequentially but can also be executed independently after model training. Each script logs progress and writes strictly defined metrics to CSV files.

- **`results/`** – contains no source code; all files are generated by the experiments. This makes it easy to include the folder in `.gitignore` and regenerate artifacts when needed.

- **`docs/`** – documentation required for the thesis text. These files describe the architecture, threat model, and experiment protocol. They can be directly transferred into the corresponding sections of the explanatory note.

### Additional Best Practices

1. **Single source of configuration** – all parameters (paths, seed, scale, etc.) are set in `app/config.py` and read from environment variables with sensible defaults. This allows easy overrides without changing code.
2. **Type hints and docstrings** – mandatory for all functions/methods, aiding tools like GitHub Copilot and simplifying debugging.
3. **Logging** – use the standard `logging` module instead of `print` throughout core logic and scripts.
4. **Testing** – the repository includes a `tests/` directory with an expected 22+ pytest tests covering core encoding, cryptography, inference, client/server, and API behavior. Experiment scripts also include assertions (e.g., comparing manual and sklearn predictions) to ensure correctness.
5. **Dependency management** – `requirements.txt` pins library versions for reproducibility. Multi‑stage Docker builds reduce image size.
6. **Error handling** – `api/main.py` includes exception handling with meaningful HTTP error codes (422 for invalid data, 500 for internal errors).
