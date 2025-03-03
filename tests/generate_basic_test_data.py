"""
Generate basic test datasets matching the core MMM implementation.

This script creates test data with the simplified structure:
TV_Spend, Digital_Spend, Search_Spend, Social_Spend, Sales
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_basic_test_data(n_periods=100, noise_level=0.1, seed=42, output_path=None):
    """
    Generate synthetic marketing and sales data with basic structure.

    Args:
        n_periods: Number of time periods
        noise_level: Level of random noise (0-1)
        seed: Random seed for reproducibility
        output_path: Path to save the CSV file (optional)

    Returns:
        pandas DataFrame with synthetic data
    """
    np.random.seed(seed)

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

    # Create DataFrame with only the essential columns
    df = pd.DataFrame({
        'TV_Spend': tv_spend,
        'Digital_Spend': digital_spend,
        'Search_Spend': search_spend,
        'Social_Spend': social_spend
    })

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
            tv_adstock[i] = 0.7 * tv_adstock[i-1] + tv_spend[i]

    # Add some adstock effect
    sales += 0.05 * tv_adstock

    # Add random noise
    noise = np.random.normal(0, sales.mean() * noise_level, n_periods)
    sales += noise

    # Add to DataFrame
    df['Sales'] = sales

    # Save to file if path provided
    if output_path:
        df.to_csv(output_path, index=False)

    return df

def create_edge_case_data():
    """
    Create simplified datasets with edge cases for testing robustness.

    Returns:
        Dictionary of DataFrames with different edge cases
    """
    edge_cases = {}

    # 1. Dataset with zero spend for one channel
    df_zero_spend = create_basic_test_data(n_periods=20, seed=123)
    df_zero_spend.loc[5:10, 'TV_Spend'] = 0
    edge_cases['zero_spend'] = df_zero_spend

    # 2. Dataset with missing values
    df_missing = create_basic_test_data(n_periods=20, seed=234)
    df_missing.loc[5:7, 'Digital_Spend'] = np.nan
    edge_cases['missing_values'] = df_missing

    # 3. Dataset with extreme outliers
    df_outliers = create_basic_test_data(n_periods=20, seed=345)
    df_outliers.loc[10, 'Search_Spend'] = df_outliers['Search_Spend'].mean() * 10
    edge_cases['outliers'] = df_outliers

    # 4. Very small dataset
    df_small = create_basic_test_data(n_periods=5, seed=456)
    edge_cases['small_dataset'] = df_small

    return edge_cases

if __name__ == "__main__":
    # Create test data directory if it doesn't exist
    data_dir = Path("tests/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate standard basic dataset
    print("Generating basic test dataset...")
    standard_df = create_basic_test_data(output_path=data_dir / "basic_test_data.csv")

    # Generate edge case datasets
    print("Generating edge case datasets...")
    edge_cases = create_edge_case_data()

    # Save edge case datasets
    for name, df in edge_cases.items():
        df.to_csv(data_dir / f"basic_{name}_data.csv", index=False)

    print(f"Generated {1 + len(edge_cases)} basic test datasets in {data_dir}")

    # Create test configuration file with metadata
    import json
    config = {
        "basic_dataset": {
            "file": "basic_test_data.csv",
            "n_periods": 100,
            "expected_elasticities": {
                "TV_Spend": 0.2,
                "Digital_Spend": 0.15,
                "Search_Spend": 0.25,
                "Social_Spend": 0.1
            }
        },
        "edge_cases": {name: {"file": f"basic_{name}_data.csv"} for name in edge_cases}
    }

    with open(data_dir / "basic_test_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Basic test data generation complete!")
