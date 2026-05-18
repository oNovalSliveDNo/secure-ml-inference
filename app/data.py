# app/data.py
"""Dataset loading and splitting for Breast Cancer Wisconsin."""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the breast cancer dataset as pandas objects.

    Returns:
        A tuple of feature matrix and target vector `(X, y)` where `X` is a
        DataFrame of shape `(569, 30)` and `y` is a Series of shape `(569,)`.

    Raises:
        ValueError: If loaded dataset shapes do not match expected dimensions.
    """
    dataset = load_breast_cancer(as_frame=True)
    X: pd.DataFrame = dataset.data
    y: pd.Series = dataset.target

    if X.shape != (569, 30):
        raise ValueError(f"Unexpected features shape: {X.shape}. Expected (569, 30).")
    if y.shape != (569,):
        raise ValueError(f"Unexpected target shape: {y.shape}. Expected (569,).")

    return X, y


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
    use_stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into train/test subsets with class stratification.

    Args:
        features: Features DataFrame.
        target: Target Series.
        test_size: Proportion of dataset to include in the test split.
        random_state: Seed for reproducible splits.

    Returns:
        A tuple `(X_train, X_test, y_train, y_test)`.
    """
    stratify_arg = target if use_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )
    return X_train, X_test, y_train, y_test
