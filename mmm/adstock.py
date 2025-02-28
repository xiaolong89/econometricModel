"""
Adstock transformations for Marketing Mix Modeling.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def apply_adstock(series, decay_rate=0.7, lag_weight=0.3, max_lag=4):
    """
    Apply adstock transformation to a time series.

    Args:
        series: Input time series (pandas Series or numpy array)
        decay_rate: Rate of decay for carryover effect
        lag_weight: Weight given to lagged values
        max_lag: Maximum lag to consider

    Returns:
        Transformed time series with adstock effect
    """
    import numpy as np

    # Convert to numpy array if it's a pandas Series
    if hasattr(series, 'values'):
        x = series.values
    else:
        x = series  # Assume it's already a numpy array

    n = len(x)
    y = np.zeros(n)

    # Apply adstock transformation
    for t in range(n):
        y[t] = x[t]  # Immediate effect
        for lag in range(1, min(t + 1, max_lag + 1)):
            # Add decayed effect from previous periods
            y[t] += lag_weight * (decay_rate ** lag) * x[t - lag]

    return y


def geometric_adstock(series, decay_rate=0.7, max_lag=10):
    """
    Apply geometric adstock transformation to a media variable.
    This is the classic adstock transformation with exponential decay.

    Args:
        series: Series containing media variable
        decay_rate: Rate of decay (between 0 and 1)
        max_lag: Maximum number of lag periods to consider

    Returns:
        Series containing adstocked values
    """
    # Convert to numpy for efficiency
    x = series.values
    n = len(x)
    adstocked = np.zeros(n)

    # Create weights for each lag period
    weights = np.array([decay_rate ** i for i in range(max_lag + 1)])
    weights = weights / weights.sum()  # Normalize

    # Apply geometric adstock transformation
    for i in range(n):
        for j in range(min(i + 1, max_lag + 1)):
            adstocked[i] += weights[j] * x[i - j] if i - j >= 0 else 0

    return pd.Series(adstocked, index=series.index)


def weibull_adstock(series, shape=2.0, scale=2.0, max_lag=10):
    """
    Apply Weibull adstock transformation to a media variable.
    This allows for more flexible response shape including delayed peak effects.

    Args:
        series: Series containing media variable
        shape: Shape parameter of Weibull distribution (>1 gives a delayed peak)
        scale: Scale parameter of Weibull distribution
        max_lag: Maximum number of lag periods to consider

    Returns:
        Series containing adstocked values
    """
    # Convert to numpy for efficiency
    x = series.values
    n = len(x)
    adstocked = np.zeros(n)

    # Create weights using Weibull distribution
    lag_periods = np.arange(max_lag + 1)
    # PDF of Weibull for the weights
    weights = (shape / scale) * (lag_periods / scale) ** (shape - 1) * np.exp(-(lag_periods / scale) ** shape)
    weights = weights / weights.sum()  # Normalize

    # Apply Weibull adstock transformation
    for i in range(n):
        for j in range(min(i + 1, max_lag + 1)):
            adstocked[i] += weights[j] * x[i - j] if i - j >= 0 else 0

    return pd.Series(adstocked, index=series.index)


def delayed_adstock(series, peak_lag=2, decay_rate=0.7, max_lag=10):
    """
    Apply delayed adstock transformation where peak effect occurs after some lag.

    Args:
        series: Series containing media variable
        peak_lag: Lag period where the effect peaks
        decay_rate: Rate of decay after peak (between 0 and 1)
        max_lag: Maximum number of lag periods to consider

    Returns:
        Series containing adstocked values
    """
    # Convert to numpy for efficiency
    x = series.values
    n = len(x)
    adstocked = np.zeros(n)

    # Create weights with delayed peak
    weights = np.zeros(max_lag + 1)
    for i in range(max_lag + 1):
        if i <= peak_lag:
            weights[i] = i / peak_lag  # Linear ramp up to peak
        else:
            weights[i] = decay_rate ** (i - peak_lag)  # Exponential decay after peak

    weights = weights / weights.sum()  # Normalize

    # Apply delayed adstock transformation
    for i in range(n):
        for j in range(min(i + 1, max_lag + 1)):
            adstocked[i] += weights[j] * x[i - j] if i - j >= 0 else 0

    return pd.Series(adstocked, index=series.index)


def apply_adstock_to_dataframe(df, media_cols, decay_rates=None, lag_weights=None, max_lags=None):
    """
    Apply adstock transformations to multiple media columns in a dataframe.

    Args:
        df: DataFrame containing media variables
        media_cols: List of media column names
        decay_rates: Dictionary mapping column names to decay rates
        lag_weights: Dictionary mapping column names to lag weights
        max_lags: Dictionary mapping column names to maximum lags

    Returns:
        DataFrame with added adstocked columns
    """
    result_df = df.copy()
    adstocked_cols = []

    # Default parameters if not provided
    if decay_rates is None:
        decay_rates = {}
    if lag_weights is None:
        lag_weights = {}
    if max_lags is None:
        max_lags = {}

    # Default decay rates based on channel type
    default_params = {
        'tv': {'decay_rate': 0.7, 'lag_weight': 0.3, 'max_lag': 8},
        'radio': {'decay_rate': 0.6, 'lag_weight': 0.3, 'max_lag': 4},
        'print': {'decay_rate': 0.5, 'lag_weight': 0.3, 'max_lag': 4},
        'digital': {'decay_rate': 0.5, 'lag_weight': 0.3, 'max_lag': 4},
        'display': {'decay_rate': 0.5, 'lag_weight': 0.3, 'max_lag': 4},
        'search': {'decay_rate': 0.3, 'lag_weight': 0.3, 'max_lag': 2},
        'social': {'decay_rate': 0.5, 'lag_weight': 0.3, 'max_lag': 5},
        'video': {'decay_rate': 0.6, 'lag_weight': 0.3, 'max_lag': 6},
        'email': {'decay_rate': 0.4, 'lag_weight': 0.3, 'max_lag': 3},
        'default': {'decay_rate': 0.5, 'lag_weight': 0.3, 'max_lag': 4}
    }

    for col in media_cols:
        # Determine channel type
        channel_type = 'default'
        for key in default_params:
            if key in col.lower():
                channel_type = key
                break

        # Get parameters for this column
        decay_rate = decay_rates.get(col, default_params[channel_type]['decay_rate'])
        lag_weight = lag_weights.get(col, default_params[channel_type]['lag_weight'])
        max_lag = max_lags.get(col, default_params[channel_type]['max_lag'])

        # Apply adstock transformation
        col_name = f"{col}_adstocked"
        logger.info(
            f"Applying adstock to {col} with decay_rate={decay_rate}, lag_weight={lag_weight}, max_lag={max_lag}")
        result_df[col_name] = apply_adstock(
            df[col],
            decay_rate=decay_rate,
            lag_weight=lag_weight,
            max_lag=max_lag
        )
        adstocked_cols.append(col_name)

    return result_df, adstocked_cols
