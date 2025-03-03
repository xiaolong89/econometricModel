"""
Adapter module to handle differences between expected and actual function signatures.

This module provides wrappers and adapters for mmm module functions to make
them compatible with our test suite, handling any signature differences.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any

# Import original functions to adapt
from mmm.adstock import apply_adstock_to_all_media as original_apply_adstock_to_all_media


def apply_adstock_to_dataframe(
        df: pd.DataFrame,
        media_cols: List[str],
        decay_rates: Optional[Dict[str, float]] = None,
        lag_weights: Optional[Dict[str, float]] = None,
        max_lags: Optional[Dict[str, int]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adapter for apply_adstock_to_all_media to maintain backward compatibility.

    This function has the same signature as our test expects, but calls
    the actual implementation function.

    Args:
        df: DataFrame containing media variables
        media_cols: List of media column names
        decay_rates: Dictionary mapping column names to decay rates
        lag_weights: Dictionary mapping column names to lag weights
        max_lags: Dictionary mapping column names to maximum lags

    Returns:
        Tuple of (DataFrame with added adstocked columns, list of added column names)
    """
    # Call the actual implementation
    return original_apply_adstock_to_all_media(
        df, media_cols, decay_rates, lag_weights, max_lags
    )
