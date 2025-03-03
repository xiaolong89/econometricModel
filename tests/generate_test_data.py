"""
Generate synthetic test datasets for Marketing Mix Model testing.

This script creates various test datasets with different characteristics
to thoroughly test the MMM functionality.
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import datetime


def create_synthetic_data(
        n_periods=100,
        include_trend=True,
        include_seasonality=True,
        include_noise=True,
        seed=42,
        output_path=None
):
    """
    Generate synthetic marketing and sales data.

    Args:
        n_periods: Number of time periods
        include_trend: Whether to include an upward trend
        include_seasonality: Whether to include seasonal patterns
        include_noise: Whether to add random noise
        seed: Random seed for reproducibility
        output_path: Path to save the CSV file (optional)

    Returns:
        pandas DataFrame with synthetic data
    """
    np.random.seed(seed)

    # Create date range
    start_date = datetime.datetime(2022, 1, 1)
    dates = [start_date + datetime.timedelta(days=i * 7) for i in range(n_periods)]

    # Base values for channels
    tv_spend_base = 50000
    digital_spend_base = 30000
    search_spend_base = 20000
    social_spend_base = 10000

    # Create spending patterns with variation over time
    tv_spend = np.random.normal(tv_spend_base, tv_spend_base * 0.2, n_periods)
    digital_spend = np.random.normal(digital_spend_base, digital_spend_base * 0.15, n_periods)
    search_spend = np.random.normal(search_spend_base, search_spend_base * 0.1, n_periods)
    social_spend = np.random.normal(social_spend_base, social_spend_base * 0.25, n_periods)

    # Ensure no negative values
    tv_spend = np.maximum(tv_spend, 0)
    digital_spend = np.maximum(digital_spend, 0)
    search_spend = np.maximum(search_spend, 0)
    social_spend = np.maximum(social_spend, 0)

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'TV_Spend': tv_spend,
        'Digital_Spend': digital_spend,
        'Search_Spend': search_spend,
        'Social_Spend': social_spend
    })

    # Add control variables
    df['price_index'] = 100 + np.random.normal(0, 5, n_periods)
    df['competitor_price_index'] = 100 + np.random.normal(0, 8, n_periods)
    df['gdp_index'] = 100 + np.cumsum(np.random.normal(0.2, 0.3, n_periods))
    df['consumer_confidence'] = 70 + np.random.normal(0, 3, n_periods)

    # Calculate sales based on marketing inputs with realistic effects
    # Base effect: Intercept
    sales = np.ones(n_periods) * 500000

    # Marketing effects with realistic elasticities and diminishing returns
    # Using log transformation to model diminishing returns
    tv_elasticity = 0.2
    digital_elasticity = 0.15
    search_elasticity = 0.25
    social_elasticity = 0.1

    # Apply effects with diminishing returns (log transformation)
    sales += tv_elasticity * 100000 * np.log1p(tv_spend / 10000)
    sales += digital_elasticity * 100000 * np.log1p(digital_spend / 10000)
    sales += search_elasticity * 100000 * np.log1p(search_spend / 10000)
    sales += social_elasticity * 100000 * np.log1p(social_spend / 10000)

    # Add carryover effects (adstock)
    # Simple implementation for synthetic data
    tv_adstock = np.zeros_like(tv_spend)
    for i in range(n_periods):
        if i == 0:
            tv_adstock[i] = tv_spend[i]
        else:
            tv_adstock[i] = 0.7 * tv_adstock[i - 1] + tv_spend[i]

    # Add some adstock effect
    sales += 0.05 * tv_adstock

    # Add trend if requested
    if include_trend:
        trend = np.linspace(0, 100000, n_periods)
        sales += trend

    # Add seasonality if requested
    if include_seasonality:
        seasonality = 50000 * np.sin(np.linspace(0, 4 * np.pi, n_periods))
        sales += seasonality

    # Add noise if requested
    if include_noise:
        noise = np.random.normal(0, 20000, n_periods)
        sales += noise

    # Add to DataFrame
    df['Sales'] = sales

    # Save to file if path provided
    if output_path:
        df.to_csv(output_path, index=False)

    return df


def create_edge_case_data():
    """
    Create datasets with edge cases for testing robustness.

    Returns:
        Dictionary of DataFrames with different edge cases
    """
    edge_cases = {}

    # 1. Dataset with zero spend for one channel
    df_zero_spend = create_synthetic_data(n_periods=50, seed=123)
    df_zero_spend.loc[10:20, 'TV_Spend'] = 0
    edge_cases['zero_spend'] = df_zero_spend

    # 2. Dataset with missing values
    df_missing = create_synthetic_data(n_periods=50, seed=234)
    df_missing.loc[5:8, 'Digital_Spend'] = np.nan
    df_missing.loc[15:18, 'Sales'] = np.nan
    edge_cases['missing_values'] = df_missing

    # 3. Dataset with extreme outliers
    df_outliers = create_synthetic_data(n_periods=50, seed=345)
    df_outliers.loc[25, 'Search_Spend'] = df_outliers['Search_Spend'].mean() * 10
    df_outliers.loc[35, 'Sales'] = df_outliers['Sales'].mean() * 5
    edge_cases['outliers'] = df_outliers

    # 4. Very small dataset
    df_small = create_synthetic_data(n_periods=10, seed=456)
    edge_cases['small_dataset'] = df_small

    # 5. Large dataset
    df_large = create_synthetic_data(n_periods=500, seed=567)
    edge_cases['large_dataset'] = df_large

    # 6. Dataset with high multicollinearity
    df_multicollinearity = create_synthetic_data(n_periods=50, seed=678)
    # Make digital spend highly correlated with search spend
    df_multicollinearity['Digital_Spend'] = df_multicollinearity['Search_Spend'] * 1.5 + np.random.normal(0, 500, 50)
    edge_cases['multicollinearity'] = df_multicollinearity

    # 7. Dataset with extreme seasonality
    df_seasonal = create_synthetic_data(n_periods=100, include_seasonality=True, seed=789)
    df_seasonal['Sales'] = df_seasonal['Sales'] + 200000 * np.sin(np.linspace(0, 8 * np.pi, 100))
    edge_cases['high_seasonality'] = df_seasonal

    return edge_cases


if __name__ == "__main__":
    # Create test data directory if it doesn't exist
    data_dir = Path("tests/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate standard synthetic dataset
    print("Generating standard synthetic dataset...")
    standard_df = create_synthetic_data(output_path=data_dir / "synthetic_data.csv")

    # Generate edge case datasets
    print("Generating edge case datasets...")
    edge_cases = create_edge_case_data()

    # Save edge case datasets
    for name, df in edge_cases.items():
        df.to_csv(data_dir / f"{name}_data.csv", index=False)

    print(f"Generated {1 + len(edge_cases)} test datasets in {data_dir}")

    # Create test configuration file with metadata
    config = {
        "standard_dataset": {
            "file": "synthetic_data.csv",
            "n_periods": 100,
            "expected_elasticities": {
                "TV_Spend": 0.2,
                "Digital_Spend": 0.15,
                "Search_Spend": 0.25,
                "Social_Spend": 0.1
            }
        },
        "edge_cases": {name: {"file": f"{name}_data.csv"} for name in edge_cases}
    }

    with open(data_dir / "test_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Test data generation complete!")
