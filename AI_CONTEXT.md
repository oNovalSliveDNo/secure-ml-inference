# AI Assistant Development Guidelines for secure-ml-inference

This file provides conventions, constraints, and expectations for any AI coding assistant contributing to this project.
Always adhere to these rules when generating, modifying, or reviewing code.

## Project Overview
- **Purpose**: Prototype of confidential ML inference using partially homomorphic encryption (Paillier) for tabular medical data.
- **Dataset**: Breast Cancer Wisconsin Diagnostic (sklearn `load_breast_cancer`).
- **Model**: StandardScaler + LogisticRegression (binary classification).
- **Protocol**: Client encrypts preprocessed features; server computes encrypted linear score; client decrypts and applies sigmoid/threshold.
- **Key Libraries**: `phe` (Paillier), `scikit-learn`, `FastAPI`, `Streamlit`, `pydantic`.
- **Full details**: See `README.md`, `docs/architecture.md`, `docs/threat_model.md`.

## Environment & Language
- Python **3.10+** only. Use f-strings, type hints, modern syntax.
- All dependencies are listed in `requirements.txt`. Do not introduce additional dependencies without updating the file.
- Use `venv` for local development.

## Code Style & Conventions
- **Type hints**: Every function signature must include full type hints for parameters and return values.
- **Docstrings**: Every public function/class must have a Google-style docstring (Args, Returns, Raises). Use `"""..."""`.
- **Logging**: Use the `logging` module (e.g., `logger.info(...)`) instead of `print()` for any runtime information.
- **Imports**: Standard library imports first, then third-party, then local (`from app. ...`). Use absolute imports.
- **Naming**: 
  - Functions/methods: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
- **Error handling**: Raise specific exceptions (e.g., `ValueError`, `FileNotFoundError`). Do not use bare `except:`.
- **Configuration**: All configurable constants must be read from `app/config.py`. No hard-coded magic numbers in other modules.

## Architectural Constraints
- **Separation of concerns**:
  - `app/` contains pure business logic (ML, encoding, crypto, inference, metrics). It **must not** import from `api/` or `ui/`.
  - `api/main.py` imports from `app/` and uses `app/schemas.py`.
  - `ui/streamlit_app.py` imports from `app/` and communicates with the API via HTTP.
- **Security boundary**:
  - The server (FastAPI or `Server` class) **never** receives plaintext features or the private key.
  - The client (`Client` class) performs all scaling, encoding, encryption, and decryption.
  - The server only stores encoded integer weights (`w_int`, `b_int`) and the public key.
- **Data flow** (as in `docs/architecture.md`):
  1. Client: raw features → StandardScaler → float scale features → fixed-point encode (× SCALE) → encrypt → send `encrypted_features` + `public_key_n` + `scale`.
  2. Server: compute `Enc(score_int) = Σ Enc(x_i) * w_int_i + Enc(b_int)`.
  3. Client: decrypt score_int → `z = score_int / (SCALE * SCALE)` → sigmoid → threshold.

## Fixed-Point Encoding Rules
- `SCALE = 10000` (from config, overridable).
- Encoding: `x_int = round(x_scaled * SCALE)`, `w_int = round(w * SCALE)`, `b_int = round(b * SCALE * SCALE)`.
- Decoding: `z = score_int / (SCALE * SCALE)`.
- The `encoded_plaintext_inference` mode must exactly replicate these steps without encryption to measure quantization error.

## Cryptography (Paillier)
- Use the `phe` library (`import phe as paillier`).
- Key generation: `public_key, private_key = paillier.generate_paillier_keypair(n_length=KEY_LENGTH)`.
- Encryption: `enc_num = public_key.encrypt(value)`.
- Decryption: `value = private_key.decrypt(enc_num)`.
- Ciphertext serialization for API: convert to string using `str(enc_num.ciphertext())`, and deserialize with `paillier.EncryptedNumber(public_key, int(ciphertext_str))`.
- The encrypted bias must be precomputed during server initialization.

## API (FastAPI) Endpoints
- `GET /health` → `{"status": "ok"}`
- `GET /model/info` → `{"feature_count": 30, "classes": ["malignant", "benign"]}`
- `POST /infer/encrypted`:
  - Request body: `EncryptedInferRequest` (Pydantic model from `app/schemas.py`)
  - Response: `EncryptedInferResponse`
  - Must measure server computation time (`server_compute_ms`).
- The API must not store any client state; it should be stateless across requests.

## Experiments & Results
- All experiment scripts are in `experiments/` and numbered 01–07.
- Each script:
  - Is self-contained (can be run independently after baseline training).
  - Loads model/scaler from `results/models/` using functions from `app/model.py`.
  - Saves output tables to `results/tables/` and plots to `results/plots/`.
- Benchmarking:
  - Use `time.perf_counter()` for high-resolution timing.
  - For mean/std/median, use `numpy` or `statistics`.
  - Payload sizes: estimate as length of JSON string after serializing the plaintext list / encrypted list.

## Unit Tests (Optional but Recommended)
- If you add tests, place them in a `tests/` directory.
- Use `pytest`. Test all encoding/decoding, crypto operations, and inference modes on a single sample.

## Git & Commits
- Commit after completing a meaningful unit of work (e.g., “Implement data loading”, “Add Paillier encryption helpers”).
- Write clear, descriptive commit messages in English.

## Development Order (Must Follow)
This is the recommended sequence to implement the project incrementally and safely:
1. `app/config.py`
2. `app/data.py` + test
3. `app/model.py` (train, save/load, extract weights, manual score)
4. `app/encoding.py`
5. `app/crypto.py`
6. `app/client.py` and `app/server.py`
7. `app/inference.py` (all four modes)
8. `experiments/01_train_baseline.py` (run to produce artifacts)
9. `experiments/02_validate_manual_inference.py`
10. `experiments/03_run_encoded_inference.py`
11. `experiments/04_run_phe_inference.py`
12. `experiments/05_benchmark_latency.py`
13. `experiments/06_benchmark_payload.py`
14. `experiments/07_benchmark_feature_scaling.py`
15. `api/main.py`
16. `ui/streamlit_app.py`
17. `Dockerfile` and `docker-compose.yml` (if not already complete)
18. Finalize `README.md`, polish documentation.

## AI Interaction Mode
- The AI should always refer to this file (`AI_CONTEXT.md`) as the source of truth for project rules.
- If in an interactive chat, the user may provide the entire content of this file once at the beginning. The AI must then follow all guidelines without needing to be reminded.
- If working in an IDE with a coding agent, the file will be automatically read from the repository root.

## Do NOT
- Do **not** change the cryptographic protocol (e.g., do not perform sigmoid under encryption).
- Do **not** introduce new libraries without approval.
- Do **not** use global mutable state in API or UI.
- Do **not** expose plaintext features to the server.
- Do **not** print sensitive data in logs.
- Do **not** deviate from the fixed-point encoding scheme.
