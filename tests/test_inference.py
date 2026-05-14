"""Unit tests for inference modes consistency."""

import numpy as np

from app.data import load_dataset, split_dataset
from app.inference import (
    encoded_plaintext_inference,
    manual_plaintext_inference,
    plaintext_inference,
)
from app.model import extract_linear_params, train_baseline_model


def test_manual_inference_matches_sklearn_on_subset() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, _ = split_dataset(X, y, test_size=0.2, random_state=42)

    pipeline = train_baseline_model(X_train.values, y_train.values, random_state=42)
    scaler = pipeline.named_steps["scaler"]
    w, b = extract_linear_params(pipeline)

    X_eval = X_test.values[:20]
    X_eval_scaled = scaler.transform(X_eval)

    pred_sklearn, prob_sklearn = plaintext_inference(pipeline, X_eval)
    pred_manual, prob_manual = manual_plaintext_inference(X_eval_scaled, w, b)

    np.testing.assert_array_equal(pred_manual, pred_sklearn)
    np.testing.assert_allclose(prob_manual, prob_sklearn, atol=1e-10)


def test_encoded_plaintext_is_close_to_manual_mode() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, _ = split_dataset(X, y, test_size=0.2, random_state=42)

    pipeline = train_baseline_model(X_train.values, y_train.values, random_state=42)
    scaler = pipeline.named_steps["scaler"]
    w, b = extract_linear_params(pipeline)

    X_eval_scaled = scaler.transform(X_test.values[:20])

    pred_manual, prob_manual = manual_plaintext_inference(X_eval_scaled, w, b)
    pred_encoded, prob_encoded = encoded_plaintext_inference(
        x_scaled=X_eval_scaled,
        w=w,
        b=b,
        scale=10_000,
        threshold=0.5,
    )

    np.testing.assert_array_equal(pred_encoded, pred_manual)
    np.testing.assert_allclose(prob_encoded, prob_manual, atol=2e-3)
