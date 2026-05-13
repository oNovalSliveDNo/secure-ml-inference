# app/data.py
"""Dataset loading and splitting for Breast Cancer Wisconsin."""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load breast cancer data as DataFrame and return features, target.

    Returns:
        X: DataFrame with 30 features.
        y: Series with binary target (0=malignant, 1=benign).
    """
    # TODO: implement using load_breast_cancer(as_frame=True)
    raise NotImplementedError


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into train/test with stratification.

    Args:
        X: Features DataFrame.
        y: Target Series.
        test_size: Proportion of test set.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    # TODO: implement using train_test_split(stratify=y)
    raise NotImplementedError
