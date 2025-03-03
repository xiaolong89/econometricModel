"""
Simplified test fixtures for the basic Marketing Mix Model test suite.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest
import json
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def basic_test_data():
    """
    Load the basic test dataset.

    Returns:
        DataFrame with synthetic data for basic testing
    """
    data_path = Path(__file__).parent / "data" / "basic_test_data.csv"

    # If file doesn't exist, generate it
    if not data_path.exists():
        from generate_basic_test_data import create_basic_test_data
        df = create_basic_test_data(n_periods=100)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        return df

    return pd.read_csv(data_path)


@pytest.fixture(scope="session")
def edge_case_data(test_data_dir):
    """
    Load edge case datasets.

    Returns:
        Dictionary of DataFrames with edge cases
    """
    config_path = test_data_dir / "basic_test_config.json"

    if not config_path.exists():
        pytest.skip("Edge case data not found. Run generate_basic_test_data.py first.")

    with open(config_path, 'r') as f:
        config = json.load(f)

    edge_cases = {}
    for case, info in config.get('edge_cases', {}).items():
        file_path = test_data_dir / info['file']
        if file_path.exists():
            edge_cases[case] = pd.read_csv(file_path)

    return edge_cases


@pytest.fixture
def log_transformed_data(basic_test_data):
    """
    Create log-transformed data for testing.

    Returns:
        DataFrame with log-transformed media variables
    """
    df = basic_test_data.copy()

    # Apply log transformation to media variables
    media_cols = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']
    for col in media_cols:
        df[f"{col}_log"] = np.log1p(df[col])

    return df


@pytest.fixture
def adstocked_data(basic_test_data):
    """
    Create adstocked data for testing.

    Returns:
        DataFrame with adstocked media variables
    """
    df = basic_test_data.copy()

    # Apply simple adstock transformation to media variables
    media_cols = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']

    for col in media_cols:
        values = df[col].values
        adstocked = np.zeros_like(values, dtype=float)

        # Apply geometric adstock with decay rate 0.7
        decay_rate = 0.7
        for i in range(len(values)):
            if i == 0:
                adstocked[i] = values[i]
            else:
                adstocked[i] = decay_rate * adstocked[i - 1] + values[i]

        df[f"{col}_adstocked"] = adstocked

    return df


@pytest.fixture
def mmm_instance():
    """
    Create a MarketingMixModel instance for testing.

    Returns:
        Initialized MarketingMixModel instance
    """
    try:
        from mmm.core import MarketingMixModel
        return MarketingMixModel()
    except (ImportError, AttributeError):
        # Mock MMM if not available
        class MockMMM:
            def __init__(self):
                self.data = None
                self.preprocessed_data = None
                self.feature_names = []

            def load_data_from_dataframe(self, df):
                self.data = df.copy()
                return self.data

        return MockMMM()


@pytest.fixture
def fitted_model(basic_test_data):
    """
    Create a simple fitted model for testing.

    Returns:
        Fitted statsmodels OLS model
    """
    import statsmodels.api as sm

    # Prepare data
    X = basic_test_data[['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']]
    y = basic_test_data['Sales']

    # Add constant
    X_const = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X_const).fit()

    return model
