"""
Basic unit tests for the adstock module focusing on core functionality.
"""

import pytest
import pandas as pd
import numpy as np

from mmm.adstock import apply_adstock


@pytest.mark.adstock
class TestBasicAdstock:
    """Basic tests for adstock transformation functions."""

    def test_apply_adstock_shape(self):
        """Test that adstock preserves the shape of the input."""
        # Create test input
        x = np.array([100, 200, 150, 300, 250])

        # Apply adstock
        y = apply_adstock(x, decay_rate=0.7, lag_weight=0.3, max_lag=3)

        # Check output shape
        assert len(y) == len(x), "Output should have same length as input"

    def test_apply_adstock_first_value(self):
        """Test that the first value is unchanged."""
        # Create test input
        x = np.array([100, 200, 150, 300, 250])

        # Apply adstock
        y = apply_adstock(x, decay_rate=0.7, lag_weight=0.3, max_lag=3)

        # Check first value
        assert y[0] == x[0], "First value should be unchanged"

    def test_apply_adstock_decay(self):
        """Test that adstock effect decays over time."""
        # Create impulse input
        x = np.zeros(10)
        x[0] = 100

        # Apply adstock
        y = apply_adstock(x, decay_rate=0.7, lag_weight=0.3, max_lag=9)

        # Check decay pattern
        for i in range(1, len(y) - 1):
            assert y[i] > y[i + 1], f"Value at position {i} should be greater than at position {i + 1}"

    def test_apply_adstock_with_real_data(self, basic_test_data):
        """Test adstock using real data."""
        # Get TV spend data
        tv_spend = basic_test_data['TV_Spend'].values

        # Apply adstock
        tv_adstocked = apply_adstock(tv_spend, decay_rate=0.7, lag_weight=0.3, max_lag=5)

        # Check that adstocked values are greater than or equal to original
        for i in range(len(tv_spend)):
            assert tv_adstocked[i] >= tv_spend[i], f"Adstocked value at {i} should be >= original"


@pytest.mark.adstock
class TestAdstockParameters:
    """Tests for different adstock parameter settings."""

    def test_decay_rate_effect(self):
        """Test the effect of different decay rates."""
        # Create impulse input
        x = np.zeros(10)
        x[0] = 100

        # Apply adstock with different decay rates
        high_decay = apply_adstock(x, decay_rate=0.9, lag_weight=0.3, max_lag=9)
        low_decay = apply_adstock(x, decay_rate=0.4, lag_weight=0.3, max_lag=9)

        # Check that higher decay rate leads to slower decay
        for i in range(1, len(x)):
            assert high_decay[i] > low_decay[i], f"High decay should be greater at position {i}"

    def test_lag_weight_effect(self):
        """Test the effect of different lag weights."""
        # Create impulse input
        x = np.zeros(10)
        x[0] = 100

        # Apply adstock with different lag weights
        high_weight = apply_adstock(x, decay_rate=0.7, lag_weight=0.8, max_lag=9)
        low_weight = apply_adstock(x, decay_rate=0.7, lag_weight=0.2, max_lag=9)

        # Check that higher lag weight leads to higher carryover values
        for i in range(1, len(x)):
            assert high_weight[i] > low_weight[i], f"High weight should be greater at position {i}"

    def test_max_lag_effect(self):
        """Test the effect of different max lag values."""
        # Create impulse input
        x = np.zeros(10)
        x[0] = 100

        # Apply adstock with different max lags
        long_lag = apply_adstock(x, decay_rate=0.7, lag_weight=0.3, max_lag=9)
        short_lag = apply_adstock(x, decay_rate=0.7, lag_weight=0.3, max_lag=3)

        # Check that long lag has effect beyond short lag cutoff
        for i in range(4, len(x)):
            # After the max_lag of the short version, values should differ
            assert long_lag[i] != short_lag[i], f"Long and short lag should differ at position {i}"


@pytest.mark.adstock
class TestDataFrameAdstock:
    """Tests for applying adstock to a DataFrame."""

    def test_apply_adstock_to_dataframe(self, basic_test_data):
        """Test applying adstock to DataFrame columns."""
        from mmm.adstock import apply_adstock_to_all_media

        # Get media columns
        media_cols = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']

        # Apply adstock
        result_df, adstock_cols = apply_adstock_to_all_media(basic_test_data, media_cols)

        # Check that result contains original data
        assert all(
            col in result_df.columns for col in basic_test_data.columns), "Result should contain all original columns"

        # Check adstocked columns
        for col in media_cols:
            adstock_col = f"{col}_adstocked"
            assert adstock_col in result_df.columns, f"Expected {adstock_col} in result"
            assert adstock_col in adstock_cols, f"Expected {adstock_col} in adstock_cols"
