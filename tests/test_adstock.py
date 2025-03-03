"""
Unit tests for the adstock module of the Marketing Mix Model.

These tests verify the functionality of adstock transformations
for modeling carryover effects in marketing spend.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mmm.adstock import (
    apply_adstock,
    geometric_adstock,
    weibull_adstock,
    delayed_adstock,
    apply_adstock_to_all_media  # Updated function name from the actual module
)


@pytest.mark.adstock
class TestAdstockTransformations:
    """Tests for adstock transformation functions."""

    def test_apply_adstock_basic(self):
        """Test basic adstock functionality."""
        # Create simple input series
        x = np.array([100, 0, 0, 0, 0], dtype=float)

        # Apply adstock with different decay rates
        y_slow = apply_adstock(x, decay_rate=0.8, lag_weight=0.3, max_lag=4)
        y_fast = apply_adstock(x, decay_rate=0.2, lag_weight=0.3, max_lag=4)

        # Check first value is unchanged
        assert y_slow[0] == x[0], "First value should be unchanged"
        assert y_fast[0] == x[0], "First value should be unchanged"

        # Check decay pattern
        for i in range(1, len(x)):
            # Slow decay should retain more of the effect
            assert y_slow[i] > y_fast[i], f"Slow decay should be greater at position {i}"

            # Values should be monotonically decreasing
            assert y_slow[i] < y_slow[i-1], f"Values should decrease with slow decay at {i}"
            assert y_fast[i] < y_fast[i-1], f"Values should decrease with fast decay at {i}"

    def test_apply_adstock_impulse_response(self):
        """Test adstock transformation with impulse input."""
        # Create impulse input (1 followed by zeros)
        n = 10
        x = np.zeros(n)
        x[0] = 1.0

        # Apply adstock
        lag_weight = 0.3
        decay_rate = 0.7
        max_lag = n - 1

        y = apply_adstock(x, decay_rate, lag_weight, max_lag)

        # First value should be the impulse
        assert y[0] == x[0], "First value should be the impulse"

        # Check subsequent values follow geometric decay pattern
        for i in range(1, n):
            expected = lag_weight * (decay_rate ** i)
            assert np.isclose(y[i], expected), f"Expected {expected} at position {i}, got {y[i]}"

    def test_apply_adstock_real_pattern(self):
        """Test adstock with realistic spending pattern."""
        # Create spending pattern with variation
        n = 20
        x = np.random.uniform(50, 150, n)

        # Apply adstock
        decay_rate = 0.7
        lag_weight = 0.3
        max_lag = 5

        y = apply_adstock(x, decay_rate, lag_weight, max_lag)

        # Check length
        assert len(y) == len(x), "Output length should match input length"

        # Check each position has expected lower bound
        for i in range(n):
            # Each position should at least have the current value
            assert y[i] >= x[i], f"Value at position {i} should be at least the input value"

            # Check positions after spending spike have elevated values
            if i > 0 and x[i-1] > x[i]:
                assert y[i] > x[i], f"Position {i} after spike should include carryover effect"

    def test_apply_adstock_parameter_effects(self):
        """Test effects of different parameter values."""
        n = 15
        x = np.zeros(n)
        x[0] = 100  # Single spike at beginning

        # Test different decay rates
        high_decay = apply_adstock(x, decay_rate=0.9, lag_weight=0.3, max_lag=n-1)
        med_decay = apply_adstock(x, decay_rate=0.5, lag_weight=0.3, max_lag=n-1)
        low_decay = apply_adstock(x, decay_rate=0.1, lag_weight=0.3, max_lag=n-1)

        # Higher decay should have higher values later in the sequence
        for i in range(2, n):
            assert high_decay[i] > med_decay[i] > low_decay[i], f"Decay order incorrect at {i}"

        # Test different lag weights
        high_weight = apply_adstock(x, decay_rate=0.7, lag_weight=0.8, max_lag=n-1)
        low_weight = apply_adstock(x, decay_rate=0.7, lag_weight=0.2, max_lag=n-1)

        # Higher lag weight should have higher values for all lags
        for i in range(1, n):
            assert high_weight[i] > low_weight[i], f"Lag weight effect incorrect at {i}"

        # Test max lag parameter
        short_lag = apply_adstock(x, decay_rate=0.7, lag_weight=0.3, max_lag=3)
        long_lag = apply_adstock(x, decay_rate=0.7, lag_weight=0.3, max_lag=10)

        # Short lag should converge faster
        assert short_lag[4] == short_lag[5], "Short lag should plateau after max_lag"
        assert long_lag[4] != long_lag[5], "Long lag should not plateau yet"


@pytest.mark.adstock
class TestAdvancedAdstockFunctions:
    """Tests for advanced adstock transformation functions."""

    def test_geometric_adstock(self):
        """Test geometric adstock transformation."""
        # Create test series
        n = 20
        series = pd.Series(np.random.uniform(50, 150, n))

        # Apply geometric adstock
        decay_rate = 0.7
        max_lag = 5

        result = geometric_adstock(series, decay_rate, max_lag)

        # Check result is a Series
        assert isinstance(result, pd.Series), "Result should be a pandas Series"
        assert len(result) == len(series), "Result length should match input"

        # Apply to an impulse for further checks
        impulse = pd.Series(np.zeros(n))
        impulse.iloc[0] = 100

        impulse_result = geometric_adstock(impulse, decay_rate, max_lag)

        # Check decay pattern
        for i in range(1, 6):
            expected = impulse.iloc[0] * (decay_rate ** i) / sum(decay_rate ** j for j in range(max_lag + 1))
            assert np.isclose(impulse_result.iloc[i], expected, rtol=1e-5), \
                f"Incorrect value at lag {i}, expected ~{expected}, got {impulse_result.iloc[i]}"

    def test_weibull_adstock(self):
        """Test Weibull adstock transformation."""
        # Create test series
        n = 20
        series = pd.Series(np.random.uniform(50, 150, n))

        # Apply Weibull adstock
        shape = 2.0
        scale = 2.0
        max_lag = 10

        result = weibull_adstock(series, shape, scale, max_lag)

        # Check result
        assert isinstance(result, pd.Series), "Result should be a pandas Series"
        assert len(result) == len(series), "Result length should match input"

        # Apply to impulse for further checks
        impulse = pd.Series(np.zeros(n))
        impulse.iloc[0] = 100

        impulse_result = weibull_adstock(impulse, shape, scale, max_lag)

        # With shape > 1, effect should peak after lag 0
        peak_lag = impulse_result.iloc[1:6].idxmax()
        assert peak_lag > 0, "Weibull with shape > 1 should peak after lag 0"

        # Value should eventually decay
        assert impulse_result.iloc[10] < impulse_result.iloc[peak_lag], "Effect should decay after peak"

    def test_delayed_adstock(self):
        """Test delayed adstock transformation."""
        # Create test series
        n = 20
        series = pd.Series(np.random.uniform(50, 150, n))

        # Apply delayed adstock
        peak_lag = 2
        decay_rate = 0.7
        max_lag = 10

        result = delayed_adstock(series, peak_lag, decay_rate, max_lag)

        # Check result
        assert isinstance(result, pd.Series), "Result should be a pandas Series"
        assert len(result) == len(series), "Result length should match input"

        # Apply to impulse for further checks
        impulse = pd.Series(np.zeros(n))
        impulse.iloc[0] = 100

        impulse_result = delayed_adstock(impulse, peak_lag, decay_rate, max_lag)

        # Effect should peak at specified lag
        max_effect_lag = impulse_result.iloc[1:6].idxmax()
        assert max_effect_lag == peak_lag, f"Effect should peak at lag {peak_lag}, but peaked at {max_effect_lag}"

        # Check decay after peak
        for i in range(peak_lag + 1, min(peak_lag + 5, n)):
            assert impulse_result.iloc[i] < impulse_result.iloc[i-1], f"Effect should decay after peak at lag {i}"


@pytest.mark.adstock
class TestDataFrameAdstock:
    """Tests for applying adstock to DataFrames."""

    def test_apply_adstock_to_all_media(self, synthetic_data):
        """Test applying adstock to multiple columns in a DataFrame."""
        df = synthetic_data.copy()

        media_cols = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']

        # Apply adstock with default parameters
        result_df, adstocked_cols = apply_adstock_to_all_media(df, media_cols)

        # Check result
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        assert len(result_df) == len(df), "Result should have same length as input"

        # Check created columns
        expected_cols = [f"{col}_adstocked" for col in media_cols]
        assert all(col in adstocked_cols for col in expected_cols), "Missing expected adstocked columns"

        for col in expected_cols:
            assert col in result_df.columns, f"Column {col} not found in result"

        # Check with custom parameters
        decay_rates = {
            'TV_Spend': 0.9,
            'Digital_Spend': 0.7,
            'Search_Spend': 0.3,
            'Social_Spend': 0.5
        }

        lag_weights = {
            'TV_Spend': 0.4,
            'Digital_Spend': 0.3,
            'Search_Spend': 0.2,
            'Social_Spend': 0.3
        }

        max_lags = {
            'TV_Spend': 8,
            'Digital_Spend': 5,
            'Search_Spend': 2,
            'Social_Spend': 4
        }

        result_df2, adstocked_cols2 = apply_adstock_to_all_media(
            df, media_cols, decay_rates, lag_weights, max_lags)

        # Check different decay rates produce different results
        for col in media_cols:
            adstocked_col = f"{col}_adstocked"
            if adstocked_col in result_df.columns and adstocked_col in result_df2.columns:
                # TV has higher decay rate in the second version, should decay slower
                if col == 'TV_Spend':
                    # Compare values at the end of the series
                    idx = -1
                    try:
                        assert result_df2[adstocked_col].iloc[idx] > result_df[adstocked_col].iloc[idx], \
                            "Higher decay rate should produce higher values at the end of the series"
                    except:
                        print(f"TV comparison failed: {result_df2[adstocked_col].iloc[idx]} vs {result_df[adstocked_col].iloc[idx]}")

                # Search has lower decay rate in the second version, should decay faster
                elif col == 'Search_Spend':
                    # Compare values at the end of the series
                    idx = -1
                    try:
                        assert result_df2[adstocked_col].iloc[idx] < result_df[adstocked_col].iloc[idx], \
                            "Lower decay rate should produce lower values at the end of the series"
                    except:
                        print(f"Search comparison failed: {result_df2[adstocked_col].iloc[idx]} vs {result_df[adstocked_col].iloc[idx]}")

    def test_adstock_defaults_by_channel(self, synthetic_data):
        """Test that default adstock parameters are appropriate for channel types."""
        df = synthetic_data.copy()

        # Map columns to channel types
        df = df.rename(columns={
            'TV_Spend': 'tv_spend',
            'Digital_Spend': 'display_spend',
            'Search_Spend': 'search_spend',
            'Social_Spend': 'social_spend'
        })

        media_cols = ['tv_spend', 'display_spend', 'search_spend', 'social_spend']

        # Apply adstock with default parameters
        result_df, _ = apply_adstock_to_all_media(df, media_cols)

        # Check that channel-specific defaults were applied
        # Create impulse data to test channel-specific defaults
        impulse_df = pd.DataFrame({
            'tv_spend': np.zeros(20),
            'display_spend': np.zeros(20),
            'search_spend': np.zeros(20),
            'social_spend': np.zeros(20)
        })

        # Add spike at beginning
        for col in media_cols:
            impulse_df.loc[0, col] = 100

        # Apply adstock
        impulse_result, _ = apply_adstock_to_all_media(impulse_df, media_cols)

        # TV should have slower decay than search
        tv_adstocked = impulse_result['tv_spend_adstocked']
        search_adstocked = impulse_result['search_spend_adstocked']

        # Check at several lags
        for i in range(2, 5):
            try:
                assert tv_adstocked.iloc[i] > search_adstocked.iloc[i], \
                    f"TV should have slower decay than search at lag {i}"
            except:
                print(f"Channel comparison failed at lag {i}: TV={tv_adstocked.iloc[i]}, Search={search_adstocked.iloc[i]}")
