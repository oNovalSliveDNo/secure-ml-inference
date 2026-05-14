# ui/streamlit_app.py
"""Streamlit demo UI for secure ML inference."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.client import Client
from app.config import KEY_LENGTH, RANDOM_STATE, SCALE, TEST_SIZE
from app.data import load_dataset, split_dataset
from app.encoding import encode_bias, encode_weights
from app.model import extract_linear_params, load_model
from app.server import Server

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


st.set_page_config(page_title="Secure ML Inference Demo", layout="wide")

MODEL_PATH = Path("results/models/model.pkl")
TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")

st.set_page_config(page_title="Secure ML Inference Demo", layout="wide")


@st.cache_resource
def load_resources() -> dict[str, Any]:
    """Load model artifacts and test split once per app process.

    Returns:
        Dict with model, scaler, test features/labels and encoded params.

    Raises:
        FileNotFoundError: If the trained model artifact is missing.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact is missing at {MODEL_PATH}. Run experiments/01_train_baseline.py first."
        )

    model = load_model(str(MODEL_PATH))
    features, target = load_dataset()
    _, x_test, _, y_test = split_dataset(
        features=features,
        target=target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = model.named_steps["scaler"]
    w, b = extract_linear_params(model)

    return {
        "model": model,
        "scaler": scaler,
        "x_test": x_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "w_int": encode_weights(w=w, scale=SCALE),
        "b_int": encode_bias(b=b, scale=SCALE),
    }


def show_demo_inference(resources: dict[str, Any]) -> None:
    """Render interactive secure inference demo tab.

    Args:
        resources: Cached app resources from ``load_resources``.
    """
    st.header("Demo Inference")

    x_test = resources["x_test"]
    y_test = resources["y_test"]
    model = resources["model"]

    sample_idx = st.slider("Test sample index", 0, len(x_test) - 1, 0)
    sample = x_test.iloc[sample_idx]

    st.subheader("Raw Features")
    st.dataframe(pd.DataFrame({"feature": sample.index, "value": sample.values}), width=True)

    if st.button("Run Secure Inference", type="primary"):
        client = Client(scaler=resources["scaler"], scale=SCALE, key_length=KEY_LENGTH)
        server = Server(
            w_int=resources["w_int"], b_int=resources["b_int"], public_key=client.public_key
        )

        t0 = time.perf_counter()
        x_scaled = client.preprocess(sample.to_numpy(dtype=float).reshape(1, -1))
        x_int = client.encode(x_scaled)
        t1 = time.perf_counter()

        enc_x = client.encrypt(x_int)
        t2 = time.perf_counter()

        enc_score = server.compute_encrypted_score(enc_x)
        t3 = time.perf_counter()

        pred_secure, prob_secure = client.decrypt_and_predict(enc_score)
        t4 = time.perf_counter()

        pred_baseline = int(model.predict(sample.to_numpy(dtype=float).reshape(1, -1))[0])
        prob_baseline = float(
            model.predict_proba(sample.to_numpy(dtype=float).reshape(1, -1))[:, 1][0]
        )

        st.subheader("Protocol Execution")
        c1, c2, c3 = st.columns(3)
        c1.metric("Encrypted features count", len(enc_x))
        c2.metric("Ciphertext chars (sum)", sum(len(str(value.ciphertext())) for value in enc_x))
        c3.metric("Total protocol time (ms)", f"{(t4 - t0) * 1000.0:.2f}")

        timing_df = pd.DataFrame(
            {
                "stage": ["preprocess+encode", "encrypt", "server_compute", "decrypt+predict"],
                "time_ms": [
                    (t1 - t0) * 1000.0,
                    (t2 - t1) * 1000.0,
                    (t3 - t2) * 1000.0,
                    (t4 - t3) * 1000.0,
                ],
            }
        )
        st.dataframe(timing_df, width=True)

        st.subheader("Prediction")
        st.write(f"**True label:** {int(y_test.iloc[sample_idx])}")
        st.write(f"**Secure prediction:** {pred_secure} (p={prob_secure:.6f})")
        st.write(f"**Baseline prediction:** {pred_baseline} (p={prob_baseline:.6f})")
        st.write(
            f"**Match with baseline:** {'✅ yes' if pred_secure == pred_baseline else '❌ no'}"
        )


def show_protocol_view(resources: dict[str, Any]) -> None:
    """Render protocol knowledge separation and encrypted samples.

    Args:
        resources: Cached app resources.
    """
    st.header("Protocol View")

    client_table = pd.DataFrame(
        {
            "Сторона клиента": [
                "Raw features x",
                "Scaler",
                "Private key",
                "Public key",
                "Encoded/encrypted features",
                "Decryption result",
            ]
        }
    )
    server_table = pd.DataFrame(
        {
            "Сторона сервера": [
                "Encoded model weights w_int",
                "Encoded bias b_int",
                "Public key",
                "Encrypted features only",
                "Encrypted score",
                "No plaintext features",
            ]
        }
    )
    left, right = st.columns(2)
    left.dataframe(client_table, width=True)
    right.dataframe(server_table, width=True)

    demo_sample = resources["x_test"].iloc[0].to_numpy(dtype=float)
    client = Client(scaler=resources["scaler"], scale=SCALE, key_length=KEY_LENGTH)
    encrypted = client.encrypt(client.encode(client.preprocess(demo_sample.reshape(1, -1))))

    st.subheader("Example ciphertext strings")
    st.code(
        "\n".join(str(value.ciphertext())[:120] + "..." for value in encrypted[:3]), language="text"
    )


def show_metrics_dashboard() -> None:
    """Render metrics tables and plots from experiment artifacts."""
    st.header("Metrics Dashboard")

    csv_files = sorted(TABLES_DIR.glob("*.csv"))

    if not csv_files:
        st.info("No CSV metrics found in results/tables/. Run experiments 04-07 first.")

    for csv_file in csv_files:
        st.subheader(csv_file.name)
        st.dataframe(pd.read_csv(csv_file), width=True)

    plot_files = sorted(PLOTS_DIR.glob("*.png"))

    if not plot_files:
        st.info("No plots found in results/plots/.")

    for plot_file in plot_files:
        st.subheader(plot_file.name)
        st.image(str(plot_file), width=True)


def show_architecture() -> None:
    """Render architecture description tab."""
    st.header("Architecture")
    st.markdown(
        """
### Secure Inference Flow
1. **Client**: raw sample → scaler → fixed-point encoding (`SCALE`) → Paillier encryption.
2. **Server**: receives only encrypted features and public key, computes encrypted linear score `Enc(z_int)`.
3. **Client**: decrypts `z_int`, decodes to float `z`, applies sigmoid and threshold.

### Trust Boundary
- Server never sees plaintext features or the private key.
- Client never sends model weights back to server; server keeps encoded model parameters.

### Message Exchange
- Request: `public_key_n`, `encrypted_features[]`, `scale`
- Response: `encrypted_score`
        """
    )


def main() -> None:
    """Run the Streamlit application."""
    st.title("Secure ML Inference — Streamlit Client")
    try:
        resources = load_resources()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Demo Inference", "Protocol View", "Metrics Dashboard", "Architecture"]
    )

    with tab1:
        show_demo_inference(resources)
    with tab2:
        show_protocol_view(resources)
    with tab3:
        show_metrics_dashboard()
    with tab4:
        show_architecture()


if __name__ == "__main__":
    main()
