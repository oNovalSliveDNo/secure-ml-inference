# ui/streamlit_app.py
"""Streamlit demo UI for secure ML inference."""

import streamlit as st

st.set_page_config(page_title="Secure ML Inference Demo", layout="wide")


# Load data and model once
@st.cache_resource
def load_resources() -> None:
    """Load model, scaler, test data."""
    # TODO: load model.pkl, data, etc.
    return None


def show_demo_inference() -> None:
    st.header("Demo Inference")
    # TODO: select sample, show features, run secure inference
    pass


def show_protocol_view() -> None:
    st.header("Protocol View")
    # TODO: display client/server knowledge
    pass


def show_metrics_dashboard() -> None:
    st.header("Metrics Dashboard")
    # TODO: load CSV and plots
    pass


def show_architecture() -> None:
    st.header("Architecture")
    # TODO: show diagram / description
    pass


tab1, tab2, tab3, tab4 = st.tabs(
    ["Demo Inference", "Protocol View", "Metrics Dashboard", "Architecture"]
)

with tab1:
    show_demo_inference()
with tab2:
    show_protocol_view()
with tab3:
    show_metrics_dashboard()
with tab4:
    show_architecture()
