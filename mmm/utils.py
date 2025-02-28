"""
Utility functions for Marketing Mix Models.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging

logger = logging.getLogger(__name__)


def calculate_vif(X):
    """
    Calculate Variance Inflation Factor (VIF) for features to detect multicollinearity.

    Args:
        X: Feature matrix

    Returns:
        DataFrame with VIF values for each feature
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X_values = X.values
    else:
        feature_names = [f'X{i}' for i in range(X.shape[1])]
        X_values = X

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X_values, i) for i in range(X.shape[1])]

    # Sort by VIF value
    vif_data = vif_data.sort_values("VIF", ascending=False)

    # Flag high VIF features
    if (vif_data["VIF"] > 10).any():
        logger.warning("High multicollinearity detected in features")

    return vif_data


def evaluate_model_performance(y_true, y_pred):
    """
    Calculate performance metrics for model evaluation.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with performance metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate MAPE if no zeros in y_true
    if not np.any(y_true == 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        # Alternative when zeros present
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    # Return metrics
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def create_train_test_split(data, train_frac=0.8, date_col=None):
    """
    Create time-based train/test split.

    Args:
        data: DataFrame to split
        train_frac: Fraction of data to use for training
        date_col: Name of date column for sorting

    Returns:
        Tuple of (train_data, test_data)
    """
    # Sort by date if provided
    if date_col and date_col in data.columns:
        data = data.sort_values(by=date_col).reset_index(drop=True)

    # Calculate split point
    split_idx = int(len(data) * train_frac)

    # Split data
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()

    logger.info(f"Created train/test split: {len(train_data)} training samples, {len(test_data)} test samples")

    return train_data, test_data