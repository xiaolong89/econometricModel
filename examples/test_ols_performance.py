"""
Test script for evaluating OLS performance with improved diminishing returns modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import statsmodels.api as sm
import seaborn as sns

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from mmm.core import MarketingMixModel
from mmm.utils import create_train_test_split, evaluate_model_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_media_correlation(mmm):
    """Analyze correlation between media channels to detect multicollinearity"""

    # Get the media columns after transformation
    media_cols = [col for col in mmm.feature_names if any(
        pattern in col.lower() for pattern in ['spend', 'tv_', 'digital', 'search', 'social', 'video', 'email']
    )]

    # Extract just these columns for correlation analysis
    media_data = mmm.X[media_cols]

    # Calculate correlation matrix
    correlation = media_data.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Media Channel Correlation Matrix')
    plt.tight_layout()
    plt.savefig('media_correlation.png')

    # Identify highly correlated pairs
    high_correlation_threshold = 0.7
    high_corr_pairs = []

    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):
            if abs(correlation.iloc[i, j]) >= high_correlation_threshold:
                high_corr_pairs.append((
                    correlation.columns[i],
                    correlation.columns[j],
                    correlation.iloc[i, j]
                ))

    # Print insights
    if high_corr_pairs:
        print("\nHigh Correlation Pairs (|r| >= 0.7):")
        for col1, col2, corr in high_corr_pairs:
            print(f"  {col1} & {col2}: r = {corr:.3f}")

        print("\nPossible Actions:")
        print("1. Remove one of the highly correlated variables")
        print("2. Create a composite variable")
        print("3. Use regularization to handle multicollinearity")
    else:
        print("\nNo highly correlated media channel pairs detected.")

    return correlation, high_corr_pairs


def test_ols_with_diminishing_returns():
    """Test OLS with improved diminishing returns handling"""

    # Load mock data
    data_path = Path(__file__).parent / 'mock_marketing_data.csv'
    mock_data = pd.read_csv(data_path)
    mock_data['date'] = pd.to_datetime(mock_data['date'])

    # Initialize MMM
    mmm = MarketingMixModel()
    mmm.load_data_from_dataframe(mock_data)

    # Define column groups
    media_cols = [
        'tv_spend',
        'digital_display_spend',
        'search_spend',
        'social_media_spend',
        'video_spend',
        'email_spend'
    ]

    control_cols = [
        'price_index',
        'competitor_price_index',
        'gdp_index',
        'consumer_confidence'
    ]

    # Apply log transformations for diminishing returns
    for col in media_cols:
        mock_data[f"{col}_log"] = np.log1p(mock_data[col])

    # Use the transformed data
    mmm.load_data_from_dataframe(mock_data)

    # Create list of transformed media columns
    transformed_cols = [f"{col}_log" for col in media_cols]

    # Preprocess data - using ONLY log-transformed media columns
    logger.info("Preprocessing data with log-transformed media variables...")
    mmm.preprocess_data(
        target='revenue',
        date_col='date',
        media_cols=transformed_cols,  # Only use log-transformed columns
        control_cols=control_cols
    )

    # Apply adstock to the log-transformed variables
    logger.info("Applying adstock transformations to log-transformed variables...")

    # Custom adstock parameters for each channel
    decay_rates = {
        'tv_spend_log': 0.85,  # TV has longer effect
        'digital_display_spend_log': 0.7,
        'search_spend_log': 0.3,  # Search has more immediate effect
        'social_media_spend_log': 0.6,
        'video_spend_log': 0.75,
        'email_spend_log': 0.4
    }

    max_lags = {
        'tv_spend_log': 8,  # TV has longer lag
        'digital_display_spend_log': 4,
        'search_spend_log': 2,  # Search has shorter lag
        'social_media_spend_log': 5,
        'video_spend_log': 6,
        'email_spend_log': 3
    }

    mmm.apply_adstock_to_all_media(
        media_cols=transformed_cols,
        decay_rates=decay_rates,
        max_lags=max_lags
    )

    # Fit the model
    logger.info("Fitting the OLS model...")
    results = mmm.fit_model()
    print("\nOLS Model with Log-Transformed Media Variables:")
    print(results.summary())

    # Analyze media channel correlation
    print("\nAnalyzing media channel correlation...")
    correlation, high_corr_pairs = analyze_media_correlation(mmm)

    # Calculate elasticities with fixed calculation
    logger.info("Calculating elasticities...")
    elasticities = mmm.calculate_elasticities()

    # Print elasticities compared to true values
    true_elasticities = {
        'tv_spend': 0.20,
        'digital_display_spend': 0.15,
        'search_spend': 0.25,
        'social_media_spend': 0.18,
        'video_spend': 0.12,
        'email_spend': 0.05
    }

    print("\nElasticity Comparison:")
    print("Channel               | True    | Estimated")
    print("---------------------|---------|----------")
    for channel, true_value in true_elasticities.items():
        est_value = elasticities.get(channel, 0)
        print(f"{channel:20} | {true_value:.3f}    | {est_value:.3f}")

    # Create train/test split for validation - ensuring chronological ordering
    logger.info("Creating time-based train/test split...")

    # Sort by date to ensure chronological order
    sorted_data = mmm.preprocessed_data.sort_values('date').copy()

    # Use the last 20% as test set
    train_size = int(len(sorted_data) * 0.8)
    train_data = sorted_data.iloc[:train_size]
    test_data = sorted_data.iloc[train_size:]

    logger.info(f"Train set: {train_data['date'].min()} to {train_data['date'].max()}")
    logger.info(f"Test set: {test_data['date'].min()} to {test_data['date'].max()}")

    # Train model on training data
    train_mmm = MarketingMixModel()
    train_mmm.load_data_from_dataframe(train_data)
    train_mmm.feature_names = mmm.feature_names
    train_mmm.X = train_data[mmm.feature_names]
    train_mmm.y = train_data[mmm.target]
    train_mmm.target = mmm.target
    train_results = train_mmm.fit_model()

    # Predict on test set
    X_test = sm.add_constant(test_data[mmm.feature_names])
    y_test = test_data[mmm.target]
    predictions = train_results.predict(X_test)

    # Evaluate performance
    metrics = evaluate_model_performance(y_test, predictions)
    print(f"\nTest Set Performance:")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(test_data['date'], y_test, 'b-', label='Actual')
    plt.plot(test_data['date'], predictions, 'r-', label='Predicted')
    plt.title('Actual vs. Predicted Revenue (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ols_actual_vs_predicted.png')

    # Plot revenue over time with forecast
    plt.figure(figsize=(14, 7))

    # Plot training data
    plt.plot(train_data['date'], train_data[mmm.target], 'b-', label='Training Data')

    # Plot test data and forecast
    plt.plot(test_data['date'], y_test, 'g-', label='Actual (Test Period)')
    plt.plot(test_data['date'], predictions, 'r--', label='Forecast')

    plt.title('Revenue Forecast Performance')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('revenue_forecast.png')

    # Plot coefficients for transformed variables
    plt.figure(figsize=(12, 6))

    # Get coefficients for transformed media variables only
    media_coefs = {}
    for var in [col for col in transformed_cols if '_adstocked' in col]:
        if var in train_results.params:
            base_channel = var.replace('_log_adstocked', '')
            media_coefs[base_channel] = train_results.params[var]

    # Sort by absolute magnitude
    sorted_channels = sorted(media_coefs.keys(), key=lambda x: abs(media_coefs[x]), reverse=True)
    values = [media_coefs[ch] for ch in sorted_channels]

    # Create bar chart
    colors = ['green' if v > 0 else 'red' for v in values]
    plt.bar(sorted_channels, values, color=colors)

    plt.title('Media Channel Coefficients (Log-Transformed & Adstocked)')
    plt.xlabel('Channel')
    plt.ylabel('Coefficient Value')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('media_coefficients.png')

    logger.info("Test completed successfully! Output plots saved.")


if __name__ == "__main__":
    test_ols_with_diminishing_returns()
