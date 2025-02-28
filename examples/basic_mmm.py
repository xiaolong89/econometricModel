"""
Marketing Mix Model with Diminishing Returns
This script implements a simple MMM with log transformations to model
the diminishing returns effect of marketing spend.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_log_transform(series, constant=1.0):
    """
    Apply log transformation to model diminishing returns.

    Args:
        series: Input time series (pandas Series or numpy array)
        constant: Small constant to add to handle zeros (default: 1.0)

    Returns:
        Log-transformed series
    """
    if hasattr(series, 'name'):
        # For pandas Series, retain index and name
        return pd.Series(
            np.log1p(series + constant),
            index=series.index,
            name=f"{series.name}_log"
        )
    else:
        # For numpy arrays
        return np.log1p(series + constant)


def compare_models(baseline_model, diminishing_model, X_test, y_test, X_dim_test):
    """
    Compare performance of models with and without diminishing returns.

    Args:
        baseline_model: Model fitted without log transformations
        diminishing_model: Model fitted with log transformations
        X_test: Original test features
        y_test: Test target variable
        X_dim_test: Test features with log transformations

    Returns:
        Dictionary with comparison metrics
    """
    # Add constant for prediction
    X_test_const = sm.add_constant(X_test)
    X_dim_test_const = sm.add_constant(X_dim_test)

    # Make predictions
    baseline_pred = baseline_model.predict(X_test_const)
    diminishing_pred = diminishing_model.predict(X_dim_test_const)

    # Calculate metrics
    metrics = {
        'baseline': {
            'r2': r2_score(y_test, baseline_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, baseline_pred)),
            'mape': mean_absolute_percentage_error(y_test, baseline_pred) * 100
        },
        'diminishing': {
            'r2': r2_score(y_test, diminishing_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, diminishing_pred)),
            'mape': mean_absolute_percentage_error(y_test, diminishing_pred) * 100
        }
    }

    return metrics


def calculate_elasticities(model, X, y, transformation_type='linear'):
    """
    Calculate elasticities for media variables.

    Args:
        model: Fitted statsmodels OLS model
        X: Feature matrix
        y: Target variable
        transformation_type: 'linear' or 'log' depending on the transformation applied

    Returns:
        Dictionary mapping channels to elasticities
    """
    elasticities = {}

    # Calculate mean of target variable
    y_mean = y.mean()

    # For each media variable
    for feature in X.columns:
        if feature == 'const':
            continue

        # Get coefficient
        coef = model.params[feature]

        # Get mean of feature
        x_mean = X[feature].mean()

        # Calculate elasticity based on transformation type
        if transformation_type == 'linear':
            # For linear models: elasticity = coefficient * (x_mean / y_mean)
            elasticity = coef * (x_mean / y_mean)
        elif transformation_type == 'log':
            # For log-transformed models: elasticity is directly the coefficient
            elasticity = coef
        else:
            raise ValueError(f"Unknown transformation type: {transformation_type}")

        # Store elasticity
        # Remove '_log' suffix if present for consistent comparison
        original_feature = feature.replace('_log', '')
        elasticities[original_feature] = elasticity

    return elasticities


def plot_transformation_effect(original_series, transformed_series, channel_name):
    """
    Plot the effect of log transformation on marketing variables.

    Args:
        original_series: Original marketing series
        transformed_series: Log-transformed series
        channel_name: Name of the marketing channel
    """
    plt.figure(figsize=(10, 6))

    # Sort values for clearer visualization
    sorted_indices = np.argsort(original_series)
    x_sorted = original_series.iloc[sorted_indices]
    y_sorted = transformed_series.iloc[sorted_indices]

    plt.scatter(x_sorted, y_sorted, alpha=0.7)
    plt.plot(x_sorted, y_sorted, 'r-', alpha=0.5)

    plt.title(f'Diminishing Returns Effect: {channel_name}')
    plt.xlabel(f'Original {channel_name}')
    plt.ylabel(f'Log-Transformed {channel_name}')
    plt.grid(True, alpha=0.3)

    # Add S-curve formula annotation
    plt.annotate('Transformation: log(x + 1)',
                 xy=(0.05, 0.95),
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    return plt.gcf()


def plot_response_curve(model, feature, feature_values, channel_name, transformation_type='linear'):
    """
    Plot the response curve for a specific marketing channel.

    Args:
        model: Fitted model
        feature: Feature name
        feature_values: Range of values to plot
        channel_name: Marketing channel name
        transformation_type: 'linear' or 'log'
    """
    plt.figure(figsize=(10, 6))

    # Create a range of values for the feature
    x_range = np.linspace(0, feature_values.max() * 1.2, 100)

    # Calculate predicted values
    y_vals = []

    # Get coefficient and intercept
    intercept = model.params['const']
    coef = model.params[feature]

    for x in x_range:
        if transformation_type == 'linear':
            y = intercept + coef * x
        else:  # log
            y = intercept + coef * np.log1p(x)
        y_vals.append(y)

    # Plot the response curve
    plt.plot(x_range, y_vals, 'b-', linewidth=2)

    # Highlight the actual data range
    actual_min = feature_values.min()
    actual_max = feature_values.max()
    actual_range = plt.axvspan(actual_min, actual_max, alpha=0.2, color='gray')

    plt.title(f'Response Curve: {channel_name}')
    plt.xlabel(f'{channel_name} Spend')
    plt.ylabel('Predicted Response')
    plt.grid(True, alpha=0.3)

    # Add annotations
    if transformation_type == 'linear':
        plt.annotate('Linear Response',
                     xy=(0.05, 0.95),
                     xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    else:
        plt.annotate('Diminishing Returns (Log) Response',
                     xy=(0.05, 0.95),
                     xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    return plt.gcf()


def main():
    # Step 1: Load the Data
    logger.info("Loading mmm_data.csv...")
    data_path = Path("C:\_econometricModel\data\mmm_data.csv")
    df = pd.read_csv(data_path)

    # Define variables
    target = 'Sales'
    media_cols = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']

    # Split data into train/test (80/20)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    logger.info(f"Training set: {len(train_df)} observations")
    logger.info(f"Test set: {len(test_df)} observations")

    # Step 2: Fit baseline model (no transformations)
    logger.info("Fitting baseline model...")
    X_train = train_df[media_cols]
    y_train = train_df[target]
    X_train_const = sm.add_constant(X_train)
    baseline_model = sm.OLS(y_train, X_train_const).fit()

    logger.info(f"Baseline model R²: {baseline_model.rsquared:.4f}")
    logger.info(f"Baseline model Adjusted R²: {baseline_model.rsquared_adj:.4f}")

    # Step 3: Apply log transformations to model diminishing returns
    logger.info("Applying log transformations for diminishing returns...")
    log_cols = []

    for col in media_cols:
        # Apply to full dataset
        df[f'{col}_log'] = apply_log_transform(df[col])
        log_cols.append(f'{col}_log')

        # Visualize transformation effect
        fig = plot_transformation_effect(df[col], df[f'{col}_log'], col)
        fig.savefig(f"{col}_diminishing_returns.png")

    # Re-split for consistency with log-transformed features
    train_df_log = df.iloc[:train_size].copy()
    test_df_log = df.iloc[train_size:].copy()

    # Step 4: Fit model with log-transformed features
    logger.info("Fitting model with log-transformed features...")
    X_train_log = train_df_log[log_cols]
    X_train_log_const = sm.add_constant(X_train_log)
    diminishing_model = sm.OLS(y_train, X_train_log_const).fit()

    logger.info(f"Diminishing returns model R²: {diminishing_model.rsquared:.4f}")
    logger.info(f"Diminishing returns model Adjusted R²: {diminishing_model.rsquared_adj:.4f}")

    # Step 5: Compare models on test set
    logger.info("Comparing models on test set...")
    X_test = test_df[media_cols]
    y_test = test_df[target]
    X_test_log = test_df_log[log_cols]

    comparison = compare_models(baseline_model, diminishing_model, X_test, y_test, X_test_log)

    print("\nTest Set Performance Comparison:")
    print(
        f"Baseline Model      - R²: {comparison['baseline']['r2']:.4f}, RMSE: {comparison['baseline']['rmse']:.2f}, MAPE: {comparison['baseline']['mape']:.2f}%")
    print(
        f"Diminishing Returns - R²: {comparison['diminishing']['r2']:.4f}, RMSE: {comparison['diminishing']['rmse']:.2f}, MAPE: {comparison['diminishing']['mape']:.2f}%")

    # Calculate improvement
    r2_improvement = comparison['diminishing']['r2'] - comparison['baseline']['r2']
    rmse_improvement = comparison['baseline']['rmse'] - comparison['diminishing']['rmse']
    mape_improvement = comparison['baseline']['mape'] - comparison['diminishing']['mape']

    print(f"\nImprovements with Diminishing Returns:")
    print(f"R² Improvement: {r2_improvement:.4f} ({r2_improvement / comparison['baseline']['r2'] * 100:.2f}%)")
    print(f"RMSE Improvement: {rmse_improvement:.2f} ({rmse_improvement / comparison['baseline']['rmse'] * 100:.2f}%)")
    print(f"MAPE Improvement: {mape_improvement:.2f} percentage points")

    # Step 6: Calculate and compare elasticities
    logger.info("Calculating elasticities...")

    # Baseline elasticities
    baseline_elasticities = calculate_elasticities(
        baseline_model,
        sm.add_constant(train_df[media_cols]),
        train_df[target],
        transformation_type='linear'
    )

    # Diminishing returns elasticities
    diminishing_elasticities = calculate_elasticities(
        diminishing_model,
        sm.add_constant(train_df_log[log_cols]),
        train_df[target],
        transformation_type='log'
    )

    # Print elasticity comparison
    print("\nElasticity Comparison:")
    print("Channel               | Baseline | Diminishing | % Difference")
    print("---------------------|----------|-------------|-------------")
    for channel in media_cols:
        baseline_value = baseline_elasticities[channel]
        diminishing_value = diminishing_elasticities[channel]
        pct_diff = (diminishing_value - baseline_value) / baseline_value * 100
        print(f"{channel:20} | {baseline_value:.4f}   | {diminishing_value:.4f}     | {pct_diff:+.2f}%")

    # Step 7: Plot response curves
    logger.info("Plotting response curves...")

    # Generate response curves for each channel
    for i, channel in enumerate(media_cols):
        # Skip other variable effects for cleaner visualization

        # Plot linear response
        fig = plot_response_curve(
            baseline_model,
            channel,
            df[channel],
            channel,
            transformation_type='linear'
        )
        fig.savefig(f"{channel}_linear_response.png")

        # Plot log response
        fig = plot_response_curve(
            diminishing_model,
            f"{channel}_log",
            df[channel],
            channel,
            transformation_type='log'
        )
        fig.savefig(f"{channel}_log_response.png")

    # Step 8: Visualize model fit
    logger.info("Visualizing model fit...")

    # Create a figure for model comparison
    plt.figure(figsize=(14, 7))

    # Plot actual values
    plt.plot(df.index, df[target], label='Actual', color='black', linewidth=2)

    # Plot baseline model predictions (full dataset)
    X_full = sm.add_constant(df[media_cols])
    baseline_pred_full = baseline_model.predict(X_full)
    plt.plot(df.index, baseline_pred_full, label='Baseline Model', color='blue', linestyle='--')

    # Plot diminishing returns model predictions (full dataset)
    X_full_log = sm.add_constant(df[log_cols])
    diminishing_pred_full = diminishing_model.predict(X_full_log)
    plt.plot(df.index, diminishing_pred_full, label='Diminishing Returns Model', color='red')

    # Add vertical line to indicate train/test split
    plt.axvline(x=train_size, color='gray', linestyle='-', alpha=0.5, label='Train/Test Split')

    plt.title('Model Fit Comparison: Baseline vs Diminishing Returns')
    plt.xlabel('Time Period')
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png')

    # Step 9: Print model summary
    print("\nFinal Diminishing Returns Model Summary:")
    print(diminishing_model.summary())

    logger.info("Analysis complete. Output visualizations saved to current directory.")


if __name__ == "__main__":
    main()
