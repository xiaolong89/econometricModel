"""
Log-log model implementation for Marketing Mix Model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


def apply_log_transformations(df, target_col, feature_cols):
    """
    Apply log1p transformations to both target and feature columns.

    Args:
        df: DataFrame with the data
        target_col: Name of the target column (e.g., 'Sales')
        feature_cols: List of feature column names (e.g., ['TV_Spend', 'Digital_Spend'])

    Returns:
        Tuple of (transformed_df, log_target_col, log_feature_cols)
    """
    transformed_df = df.copy()

    # Transform target (for log-log model)
    log_target_col = f"{target_col}_log"
    transformed_df[log_target_col] = np.log1p(df[target_col])

    # Transform features
    log_feature_cols = []
    for col in feature_cols:
        log_col = f"{col}_log"
        transformed_df[log_col] = np.log1p(df[col])
        log_feature_cols.append(log_col)

    return transformed_df, log_target_col, log_feature_cols


def calculate_elasticities(model, X, y, model_type='linear-linear'):
    """
    Calculate elasticities for media variables with proper handling of different transformation types.

    Args:
        model: Fitted statsmodels OLS model
        X: Feature matrix
        y: Target variable
        model_type: Transformation type applied to the model:
            - 'linear-linear': Neither X nor y transformed (standard OLS)
            - 'log-log': Both X and y are log-transformed
            - 'semi-log': Only X variables are log-transformed, y is in original scale
            - 'log-linear': Only y is log-transformed, X variables are in original scale

    Returns:
        Dictionary mapping channels to elasticities
    """
    elasticities = {}

    # Calculate mean of target variable
    y_mean = y.mean()

    # For each feature
    for feature in X.columns:
        if feature == 'const':
            continue

        # Get coefficient
        coef = model.params[feature]

        # Get mean of feature
        x_mean = X[feature].mean()

        # Calculate elasticity based on model type
        if model_type == 'linear-linear':
            # Standard OLS: elasticity = coefficient * (x_mean / y_mean)
            elasticity = coef * (x_mean / y_mean)

        elif model_type == 'log-log':
            # Log-Log model: elasticity is directly the coefficient
            elasticity = coef

        elif model_type == 'semi-log':
            # Semi-Log model (log X, linear y): elasticity = coefficient * (1 / y_mean)
            elasticity = coef * (1 / y_mean)

        elif model_type == 'log-linear':
            # Log-Linear model (linear X, log y): elasticity = coefficient * x_mean
            elasticity = coef * x_mean

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Store elasticity
        # Remove '_log' suffix if present for consistent comparison
        original_feature = feature.replace('_log', '')
        elasticities[original_feature] = elasticity

    return elasticities


def evaluate_model_performance(y_true, y_pred, model_name):
    """Evaluate model performance with multiple metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    print(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    return {
        "r2": r2,
        "rmse": rmse,
        "mape": mape
    }

def plot_actual_vs_predicted(actual_sales, actual_units,
                             pred_baseline_sales, pred_baseline_units,
                             pred_loglog_sales, pred_loglog_units,
                             dates=None, title="Model Comparison"):
    # Create subplots for Sales and Units
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Sales subplot
    ax1.plot(dates, actual_sales, 'b-', label='Actual Sales', linewidth=2)
    ax1.plot(dates, pred_baseline_sales, 'r-', label='Baseline Sales', alpha=0.7)
    ax1.plot(dates, pred_loglog_sales, 'g-', label='Log-Log Sales', alpha=0.7)
    ax1.set_title('Sales Comparison')
    ax1.set_ylabel('Sales')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Units subplot
    ax2.plot(dates, actual_units, 'b-', label='Actual Units', linewidth=2)
    ax2.plot(dates, pred_baseline_units, 'r-', label='Baseline Units', alpha=0.7)
    ax2.plot(dates, pred_loglog_units, 'g-', label='Log-Log Units', alpha=0.7)
    ax2.set_title('Units Sold Comparison')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Units')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()


def plot_response_curves(model, X, model_type, feature_cols,
                         current_values, min_pct=50, max_pct=200):
    """
    Plot response curves for each feature to visualize diminishing returns.

    Args:
        model: Fitted statsmodels model
        X: Features data
        model_type: Type of model ('linear-linear', 'log-log', etc.)
        feature_cols: List of feature column names
        current_values: Dictionary mapping features to their current values
        min_pct: Minimum percentage of current value to simulate
        max_pct: Maximum percentage of current value to simulate
    """
    plt.figure(figsize=(15, 10))

    # Number of rows and columns for subplots
    n_cols = 2
    n_rows = (len(feature_cols) + 1) // n_cols

    for i, feature in enumerate(feature_cols):
        ax = plt.subplot(n_rows, n_cols, i + 1)

        # Current value of the feature
        current_val = current_values[feature.replace('_log', '')]

        # Range of values to simulate
        spend_range = np.linspace(current_val * min_pct / 100,
                                  current_val * max_pct / 100,
                                  100)

        # Calculate predicted outcomes for the range
        outcomes = []

        for spend in spend_range:
            # Create a copy of X for prediction
            X_copy = X.copy()

            # Modify the specific feature value
            if model_type == 'log-log' or model_type == 'semi-log':
                # For log models, we need to log-transform the spend
                X_copy.loc[0, feature] = np.log1p(spend)
            else:
                X_copy.loc[0, feature] = spend

            # Predict
            pred = model.predict(X_copy)[0]

            # Convert prediction back to original scale if needed
            if model_type == 'log-log' or model_type == 'log-linear':
                pred = np.expm1(pred)

            outcomes.append(pred)

        # Plot the response curve
        ax.plot(spend_range, outcomes, 'b-')

        # Mark the current value
        current_outcome_idx = np.abs(spend_range - current_val).argmin()
        current_outcome = outcomes[current_outcome_idx]
        ax.scatter([current_val], [current_outcome], color='red', s=50,
                   label='Current Value')

        # Calculate elasticity at current value
        if model_type == 'log-log':
            elasticity = model.params[feature]
        elif model_type == 'semi-log':
            elasticity = model.params[feature] * (1 / np.mean(np.expm1(model.endog)))
        elif model_type == 'linear-linear':
            elasticity = model.params[feature] * (current_val / current_outcome)

        # Add title and labels
        feature_name = feature.replace('_log', '')
        ax.set_title(f"{feature_name} Response (Elasticity: {elasticity:.4f})")
        ax.set_xlabel(f"{feature_name} Spend")
        ax.set_ylabel("Predicted Sales")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('response_curves.png')
    plt.show()


def run_log_log_model(data_path):
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Diagnostic print statements
    print("\n--- Data Diagnostic Information ---")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nColumn Data Types:")
    print(df.dtypes)

    print("\nFirst few rows:")
    print(df.head())

    print("\nBasic statistical summary:")
    print(df.describe())
    """
    Run the MMM analysis using a log-log model.

    Args:
        data_path: Path to the data file

    Returns:
        Tuple of (baseline_results, log_log_results)
    """
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Convert Date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        # Sort by date to ensure chronological order
        df = df.sort_values('Date').reset_index(drop=True)

    # Identify target column (support both 'Sales' and 'Revenue')
    target_sales_col = 'Sales' if 'Sales' in df.columns else 'Revenue'
    target_units_col = 'Units Sold'

    # Validate both target columns exist
    if target_sales_col not in df.columns or target_units_col not in df.columns:
        raise ValueError(f"Missing required columns. Need '{target_sales_col}' and '{target_units_col}'.")

    # Identify media channels
    feature_cols = [col for col in df.columns if '_Spend' in col]

    print(f"Media channels: {feature_cols}")
    print(f"Target variable: {target_sales_col}")

    # Split data into train and test sets (80/20 split)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Store current values for response curves
    current_values = {col: df[col].mean() for col in feature_cols}

    # Apply transformations
    transformed_train_df, log_target_col, log_feature_cols = apply_log_transformations(
        train_df, target_sales_col, feature_cols)

    transformed_test_df, _, _ = apply_log_transformations(
        test_df, target_sales_col, feature_cols)

    # ------------------------------------------------------------------------
    # Baseline Model (Linear-Linear)
    # ------------------------------------------------------------------------
    print("\nTraining baseline model (linear-linear)...")

    # Prepare X and y for baseline model
    X_train_base = sm.add_constant(train_df[feature_cols])
    y_train_base = train_df[target_sales_col]

    # Fit baseline model
    baseline_model = sm.OLS(y_train_base, X_train_base).fit()
    print(baseline_model.summary())

    # Make predictions on test set
    X_test_base = sm.add_constant(test_df[feature_cols])
    y_test_base = test_df[target_sales_col]
    baseline_predictions = baseline_model.predict(X_test_base)

    # Evaluate baseline model
    baseline_metrics = evaluate_model_performance(
        y_test_base, baseline_predictions, "Baseline Model")

    # Calculate baseline elasticities
    baseline_elasticities = calculate_elasticities(
        baseline_model, X_train_base, y_train_base, model_type='linear-linear')

    # ------------------------------------------------------------------------
    # Log-Log Model
    # ------------------------------------------------------------------------
    print("\nTraining log-log model...")

    # Prepare X and y for log-log model
    X_train_log = sm.add_constant(transformed_train_df[log_feature_cols])
    y_train_log = transformed_train_df[log_target_col]

    # Fit log-log model
    log_log_model = sm.OLS(y_train_log, X_train_log).fit()
    print(log_log_model.summary())

    # Make predictions on test set (log scale)
    X_test_log = sm.add_constant(transformed_test_df[log_feature_cols])
    y_test_log = transformed_test_df[log_target_col]
    log_log_predictions_log = log_log_model.predict(X_test_log)

    # Back-transform predictions to original scale for comparison
    log_log_predictions = np.expm1(log_log_predictions_log)

    # Evaluate log-log model (comparing in original scale)
    log_log_metrics = evaluate_model_performance(
        test_df[target_sales_col], log_log_predictions, "Log-Log Model")

    # Calculate log-log elasticities
    log_log_elasticities = calculate_elasticities(
        log_log_model, X_train_log, y_train_log, model_type='log-log')

    # ------------------------------------------------------------------------
    # Compare Results
    # ------------------------------------------------------------------------
    print("\nTest Set Performance Comparison:")
    print(
        f"Baseline Model      - R²: {baseline_metrics['r2']:.4f}, RMSE: {baseline_metrics['rmse']:.2f}, MAPE: {baseline_metrics['mape']:.2f}%")
    print(
        f"Log-Log Model       - R²: {log_log_metrics['r2']:.4f}, RMSE: {log_log_metrics['rmse']:.2f}, MAPE: {log_log_metrics['mape']:.2f}%")

    # Compare elasticities
    print("\nElasticity Comparison:")
    print("Channel               | Baseline | Log-Log   | % Difference")
    print("---------------------|----------|-----------|-------------")

    for channel in baseline_elasticities.keys():
        base_elas = baseline_elasticities[channel]
        log_elas = log_log_elasticities[channel]
        pct_diff = (log_elas / base_elas - 1) * 100 if base_elas != 0 else float('inf')

        print(f"{channel:20} | {base_elas:.4f}   | {log_elas:.4f}    | {pct_diff:+.2f}%")

    # Plot actual vs predicted for both models
    plot_actual_vs_predicted(
        test_df[target_sales_col],  # actual_sales
        test_df[target_units_col],  # actual_units
        baseline_predictions,  # pred_baseline_sales
        baseline_predictions,  # pred_baseline_units
        log_log_predictions,  # pred_loglog_sales
        log_log_predictions,  # pred_loglog_units
        dates=test_df['Date'] if 'Date' in test_df.columns else None,
        title="Actual vs. Predicted Sales and Units (Test Set)"
    )

    # Plot response curves for log-log model
    # First, create a single row DataFrame for prediction
    X_pred = X_train_log.iloc[[0]].copy()
    plot_response_curves(
        log_log_model,
        X_pred,
        'log-log',
        log_feature_cols,
        current_values
    )

    # In the result dictionary, include both projections
    return {
        'baseline': {
            'model': baseline_model,
            'metrics': baseline_metrics,
            'elasticities': baseline_elasticities,
            'predictions': {
                'sales': baseline_predictions,
                'units': baseline_predictions  # You might want a separate prediction logic
            }
        },
        'log_log': {
            'model': log_log_model,
            'metrics': log_log_metrics,
            'elasticities': log_log_elasticities,
            'predictions': {
                'sales': log_log_predictions,
                'units': log_log_predictions  # Again, potentially more sophisticated later
            }
        }
    }


# Execute the analysis if the script is run directly
if __name__ == "__main__":
    # Specify the data file path
    data_file = "C:\_econometricModel\data\mmm_data.csv"  # Update this to your actual data path

    # Run the analysis
    results = run_log_log_model(data_file)

    print("\nAnalysis complete. Results and visualizations saved.")
