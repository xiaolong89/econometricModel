"""
Unit tests for the preprocessing module of the Marketing Mix Model.

This file contains tests for data loading, feature detection, stationarity checks,
and transformation functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from mmm.preprocessing import (
    detect_media_columns,
    detect_control_columns,
    check_stationarity,
    make_stationary,
    add_seasonality_features,
    orthogonalize_features,
    apply_diminishing_returns_transformations,
    preprocess_for_modeling
)


@pytest.mark.preprocessing
class TestFeatureDetection:
    """Tests for automatic feature detection functions."""

    def test_detect_media_columns(self, synthetic_data):
        """Test detection of media columns."""
        # Test with standard naming
        detected = detect_media_columns(synthetic_data)
        expected = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']
        assert set(detected) == set(expected), "Failed to detect standard media columns"

        # Test with custom naming
        df = synthetic_data.copy()
        df = df.rename(columns={
            'TV_Spend': 'television_advertising',
            'Digital_Spend': 'programmatic_spend',
            'Search_Spend': 'google_ads_cost',
            'Social_Spend': 'facebook_marketing'
        })

        detected = detect_media_columns(df)
        assert len(detected) == 4, "Failed to detect media columns with custom naming"
        assert 'television_advertising' in detected
        assert 'programmatic_spend' in detected
        assert 'google_ads_cost' in detected
        assert 'facebook_marketing' in detected

    def test_detect_control_columns(self, synthetic_data):
        """Test detection of control columns."""
        target_col = 'Sales'
        date_col = 'date'
        media_cols = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']

        detected = detect_control_columns(synthetic_data, target_col, date_col, media_cols)
        expected = ['price_index', 'competitor_price_index', 'gdp_index', 'consumer_confidence']

        assert set(detected) == set(expected), "Failed to detect control columns"

        # Test with custom control variable names
        df = synthetic_data.copy()
        df = df.rename(columns={
            'price_index': 'product_pricing',
            'competitor_price_index': 'competitor_pricing',
            'gdp_index': 'economic_indicator',
            'consumer_confidence': 'consumer_sentiment'
        })

        detected = detect_control_columns(df, target_col, date_col, media_cols)
        assert len(detected) == 4, "Failed to detect control columns with custom naming"
        assert 'product_pricing' in detected
        assert 'competitor_pricing' in detected
        assert 'economic_indicator' in detected
        assert 'consumer_sentiment' in detected


@pytest.mark.preprocessing
class TestStationarityChecks:
    """Tests for stationarity check functions."""

    def test_check_stationarity(self, synthetic_data):
        """Test stationarity check function."""
        # Create non-stationary series
        trend = np.linspace(0, 100, 100)
        non_stationary = pd.Series(trend + np.random.normal(0, 5, 100))

        # Create stationary series
        stationary = pd.Series(np.random.normal(50, 10, 100))

        # Test non-stationary series
        is_stationary, _, p_value = check_stationarity(non_stationary)
        assert not is_stationary, "Non-stationary series incorrectly identified as stationary"
        assert p_value > 0.05, "Expected p-value > 0.05 for non-stationary series"

        # Test stationary series
        is_stationary, _, p_value = check_stationarity(stationary)
        assert is_stationary, "Stationary series incorrectly identified as non-stationary"
        assert p_value <= 0.05, "Expected p-value <= 0.05 for stationary series"

    def test_make_stationary(self, synthetic_data):
        """Test make_stationary function with different transformations."""
        # Create non-stationary series with strong trend
        time = np.arange(100)
        trend_data = pd.DataFrame({
            'target': 1000 + 50 * time + np.random.normal(0, 200, 100)
        })

        # Test log transformation
        transformed_df, new_target, _ = make_stationary(
            trend_data, 'target', transformation_type='log')

        assert new_target == 'target_log', "Expected new target column name to be 'target_log'"
        assert 'target_log' in transformed_df.columns, "Log-transformed column not found"

        # Verify log transformation made the series more stationary
        is_stationary_orig, _, p_orig = check_stationarity(trend_data['target'])
        is_stationary_log, _, p_log = check_stationarity(transformed_df['target_log'])

        assert p_log < p_orig, "Log transformation did not improve stationarity"

        # Test differencing
        transformed_df, new_target, _ = make_stationary(
            trend_data, 'target', transformation_type='diff')

        assert new_target == 'target_diff', "Expected new target column name to be 'target_diff'"
        assert 'target_diff' in transformed_df.columns, "Differenced column not found"

        # Verify differencing made the series stationary
        is_stationary_diff, _, _ = check_stationarity(transformed_df['target_diff'][1:])  # Skip first (NA) value
        assert is_stationary_diff, "Differencing did not make the series stationary"


@pytest.mark.preprocessing
class TestSeasonalityFeatures:
    """Tests for seasonality feature functions."""

    def test_add_seasonality_features(self, synthetic_data):
        """Test adding seasonality features."""
        # Test with date column
        df = synthetic_data.copy()
        result_df = add_seasonality_features(df, 'date')

        # Check expected columns
        expected_columns = [
            'month', 'quarter', 'year', 'week_of_year',
            'time_trend', 'quarter_2', 'quarter_3', 'quarter_4'
        ]

        for col in expected_columns:
            assert col in result_df.columns, f"Expected column {col} not found"

        # Check values
        assert result_df['month'].min() >= 1, "Month values should be >= 1"
        assert result_df['month'].max() <= 12, "Month values should be <= 12"

        assert result_df['quarter'].min() >= 1, "Quarter values should be >= 1"
        assert result_df['quarter'].max() <= 4, "Quarter values should be <= 4"

        # Check time trend
        assert result_df['time_trend'].equals(pd.Series(range(len(df)))), "Time trend should be sequential"

        # Check quarter dummies
        assert set(result_df['quarter_2'].unique()).issubset({0, 1}), "Quarter_2 should be binary"
        assert set(result_df['quarter_3'].unique()).issubset({0, 1}), "Quarter_3 should be binary"
        assert set(result_df['quarter_4'].unique()).issubset({0, 1}), "Quarter_4 should be binary"


@pytest.mark.preprocessing
class TestFeatureTransformations:
    """Tests for feature transformation functions."""

    def test_orthogonalize_features(self):
        """Test feature orthogonalization."""
        # Create dataframe with correlated features
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, n)  # Correlated with x1
        x3 = np.random.normal(0, 1, n)  # Independent

        df = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'target': 3 * x1 + 2 * x2 + x3 + np.random.normal(0, 0.5, n)
        })

        # Original correlation
        orig_corr = df[['x1', 'x2']].corr().iloc[0, 1]
        assert abs(orig_corr) > 0.7, "Test setup error: features should be correlated"

        # Test QR orthogonalization
        feature_cols = ['x1', 'x2', 'x3']
        result_df = orthogonalize_features(df, feature_cols, method='qr')

        # Check orthogonalized columns exist
        assert 'x1_ortho' in result_df.columns, "Orthogonalized x1 column not found"
        assert 'x2_ortho' in result_df.columns, "Orthogonalized x2 column not found"
        assert 'x3_ortho' in result_df.columns, "Orthogonalized x3 column not found"

        # Check orthogonality
        ortho_cols = ['x1_ortho', 'x2_ortho', 'x3_ortho']
        ortho_corr = result_df[ortho_cols].corr()

        # Off-diagonal elements should be close to zero
        for i in range(len(ortho_cols)):
            for j in range(i + 1, len(ortho_cols)):
                assert abs(ortho_corr.iloc[i, j]) < 0.01, "Features not properly orthogonalized"

    def test_apply_diminishing_returns_transformations(self, synthetic_data):
        """Test applying diminishing returns transformations."""
        df = synthetic_data.copy()
        media_cols = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']

        # Test log transformation
        result_df, transformed_cols = apply_diminishing_returns_transformations(
            df, media_cols, method='log')

        expected_cols = ['TV_Spend_log', 'Digital_Spend_log', 'Search_Spend_log', 'Social_Spend_log']
        assert set(transformed_cols) == set(expected_cols), "Unexpected transformed column names"

        # Check that all columns exist
        for col in expected_cols:
            assert col in result_df.columns, f"Expected column {col} not found"

        # Check log transformation was applied correctly
        for original, transformed in zip(media_cols, expected_cols):
            # Log1p should be monotonically increasing with original
            assert (result_df[transformed].diff()[1:] * df[original].diff()[1:]).min() >= 0, \
                f"Log transformation of {original} not monotonic"

            # Check specific values
            idx = df[original].idxmax()
            assert np.isclose(
                result_df.loc[idx, transformed],
                np.log1p(df.loc[idx, original] + result_df[transformed].min())
            ), f"Log transformation incorrect for {original}"

        # Test hill transformation
        result_df, transformed_cols = apply_diminishing_returns_transformations(
            df, media_cols, method='hill')

        expected_cols = ['TV_Spend_hill', 'Digital_Spend_hill', 'Search_Spend_hill', 'Social_Spend_hill']
        assert set(transformed_cols) == set(
            expected_cols), "Unexpected transformed column names for hill transformation"

        for col in expected_cols:
            assert col in result_df.columns, f"Expected column {col} not found"

        # Hill function should be bounded between 0 and 1
        for col in expected_cols:
            assert result_df[col].min() >= 0, f"Hill transformation should be non-negative for {col}"
            assert result_df[col].max() <= 1, f"Hill transformation should be <= 1 for {col}"


@pytest.mark.preprocessing
class TestEndToEndPreprocessing:
    """Tests for end-to-end preprocessing workflow."""

    def test_preprocess_for_modeling(self, synthetic_data):
        """Test end-to-end preprocessing workflow."""
        df = synthetic_data.copy()

        # Define columns for preprocessing
        target = 'Sales'
        date_col = 'date'
        media_cols = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']
        control_cols = ['price_index', 'competitor_price_index', 'gdp_index', 'consumer_confidence']

        # Run preprocessing
        processed_df, X, y, feature_names = preprocess_for_modeling(
            df,
            target=target,
            date_col=date_col,
            media_cols=media_cols,
            control_cols=control_cols,
            make_stationary_flag=True,
            orthogonalize=True,
            add_seasonality=True
        )

        # Check output types
        assert isinstance(processed_df, pd.DataFrame), "processed_df should be a DataFrame"
        assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
        assert isinstance(y, pd.Series), "y should be a Series"
        assert isinstance(feature_names, list), "feature_names should be a list"

        # Check dimensions
        assert len(processed_df) == len(df), "processed_df should have same length as input df"
        assert len(X) == len(df), "X should have same length as input df"
        assert len(y) == len(df), "y should have same length as input df"
        assert len(feature_names) > 0, "feature_names should not be empty"
        assert len(feature_names) == X.shape[1], "feature_names length should match X columns"

        # Check transformed target
        assert 'Sales_log' in processed_df.columns, "Expected log-transformed target column"

        # Check X contains expected features
        media_transformations = [col for col in X.columns if any(media in col for media in media_cols)]
        assert len(media_transformations) > 0, "No media transformations found in X"

        control_features = [col for col in X.columns if any(ctrl in col for ctrl in control_cols)]
        assert len(control_features) > 0, "No control features found in X"

        seasonality_features = [
            col for col in X.columns if 'quarter_' in col or 'time_trend' in col or 'holiday_' in col
        ]
        assert len(seasonality_features) > 0, "No seasonality features found in X"
