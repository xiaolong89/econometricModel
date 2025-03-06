"""
Simplified MMM Optimization Script that integrates all improvements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
import sys
import logging
import time
import warnings
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

# Import core MMM only - we'll define our own utilities
from mmm.core import MarketingMixModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define utility functions that might not exist in your package
def apply_adstock(series, decay_rate=0.7, lag_weight=0.3, max_lag=10, decay_type='geometric'):
    """
    Apply adstock transformation to a media variable.

    Args:
        series: Series containing media variable
        decay_rate: Rate of decay (between 0 and 1)
        lag_weight: Weight applied to lagged effects
        max_lag: Maximum number of lag periods to consider
        decay_type: Type of decay function ('geometric', 'weibull', or 'delayed')

    Returns:
        Series containing adstocked values
    """
    # Convert to numpy for efficiency
    x = series.values
    n = len(x)
    adstocked = np.zeros(n)

    # Create weights based on decay pattern
    if decay_type == 'geometric':
        # Standard geometric decay (exponential)
        weights = np.array([decay_rate ** i for i in range(max_lag + 1)])
    elif decay_type == 'weibull':
        # Weibull decay (allows for S-shaped response)
        shape = 2.0  # Shape parameter > 1 gives delayed peak
        scale = 1.0 / (-np.log(decay_rate))  # Scale parameter based on decay rate
        weights = np.array([np.exp(-(i / scale) ** shape) for i in range(max_lag + 1)])
    elif decay_type == 'delayed':
        # Delayed adstock (peak effect not immediate)
        peak = 2  # Peak effect at lag 2
        weights = np.array([
            (i / peak) * decay_rate ** (i - peak) if i <= peak else decay_rate ** (i - peak)
            for i in range(max_lag + 1)
        ])
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")

    # Normalize weights
    weights = weights / weights.sum()

    # Apply adstock transformation
    for i in range(n):
        adstocked[i] = x[i]  # Current effect

        # Add lagged effects
        for lag in range(1, min(i + 1, max_lag + 1)):
            adstocked[i] += x[i - lag] * weights[lag] * lag_weight

    return pd.Series(adstocked, index=series.index)


def check_stationarity(series):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.

    Args:
        series: Time series to check

    Returns:
        Tuple of (is_stationary, adf_statistic, p_value)
    """
    result = adfuller(series.dropna())
    is_stationary = result[1] <= 0.05

    logger.info(f"ADF Statistic: {result[0]}")
    logger.info(f"p-value: {result[1]}")

    if is_stationary:
        logger.info("Target variable is stationary")
    else:
        logger.warning("Target variable may not be stationary. Consider differencing or transformations.")

    return is_stationary, result[0], result[1]


def make_stationary(df, target_col, transformation_type='log'):
    """
    Transform the target variable to make it stationary.

    Args:
        df: DataFrame containing the data
        target_col: Name of the target variable column
        transformation_type: Type of transformation ('log', 'diff', 'log_diff')

    Returns:
        Tuple of (transformed_df, new_target_name, transformation_info)
    """
    transformed_df = df.copy()
    is_stationary, _, _ = check_stationarity(df[target_col])
    transformation_info = {'original': target_col}

    if is_stationary:
        logger.info(f"{target_col} is already stationary. No transformation needed.")
        return transformed_df, target_col, transformation_info

    # Try log transformation
    if transformation_type in ['log', 'log_diff']:
        log_col = f"{target_col}_log"
        transformed_df[log_col] = np.log1p(df[target_col])

        logger.info(f"Applied log transformation to {target_col}")
        return transformed_df, log_col, transformation_info

    # Try differencing (or log differencing)
    if transformation_type in ['diff', 'log_diff']:
        if transformation_type == 'diff':
            diff_col = f"{target_col}_diff"
            transformed_df[diff_col] = df[target_col].diff().fillna(0)
        else:
            diff_col = f"{target_col}_log_diff"
            transformed_df[diff_col] = transformed_df[f"{target_col}_log"].diff().fillna(0)

        logger.info(f"Applied differencing to make {target_col} stationary")
        return transformed_df, diff_col, transformation_info

    # If all attempts failed, use the last transformation tried
    logger.warning("Could not achieve stationarity. Using original variable.")
    return transformed_df, target_col, transformation_info


def orthogonalize_features(df, feature_cols, method='qr'):
    """
    Orthogonalize highly correlated features to address multicollinearity.

    Args:
        df: DataFrame with features
        feature_cols: List of feature columns to orthogonalize
        method: Orthogonalization method ('qr' or 'residualization')

    Returns:
        DataFrame with orthogonalized features
    """
    result_df = df.copy()

    # Standardize features
    features_data = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_data)

    if method == 'qr':
        # QR decomposition for orthogonalization
        Q, R = np.linalg.qr(X_scaled)

        # Create orthogonalized features
        for i, col in enumerate(feature_cols):
            result_df[f"{col}_ortho"] = Q[:, i]

        # Log correlation of orthogonalized features
        ortho_cols = [f"{col}_ortho" for col in feature_cols]
        if len(ortho_cols) >= 2:  # Need at least 2 columns for correlation
            corr_matrix = result_df[ortho_cols].corr()
            max_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
            # Remove self-correlations (which are always 1.0)
            max_corr = max_corr[max_corr < 1.0]
            logger.info(
                f"Max correlation between orthogonalized features: {max_corr.iloc[0] if len(max_corr) > 0 else 0}")

    return result_df


def analyze_media_correlation(X):
    """
    Analyze correlation between media channels to detect multicollinearity.

    Args:
        X: DataFrame containing media variables

    Returns:
        Tuple of correlation matrix and high correlation pairs
    """
    # Calculate correlation matrix
    correlation = X.corr()

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

    return correlation, high_corr_pairs


def add_careful_interactions(df, media_cols, target_col, max_interactions=2):
    """
    Add interaction terms between key media channels with care to avoid
    introducing multicollinearity.

    Args:
        df: DataFrame to modify
        media_cols: List of media column names
        target_col: Target variable name
        max_interactions: Maximum number of interaction terms to add

    Returns:
        Updated DataFrame and list of added interaction columns
    """
    # Calculate individual correlations with the target
    correlations = {}
    for col in media_cols:
        correlations[col] = df[col].corr(df[target_col])

    # Sort media channels by correlation strength
    sorted_channels = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    # Select top channels
    top_channels = [ch[0] for ch in sorted_channels[:4]]

    # Candidate interaction pairs from top channels
    candidate_pairs = []
    for i, col1 in enumerate(top_channels):
        base_col1 = col1.split('_')[0]
        for j, col2 in enumerate(top_channels[i + 1:], i + 1):
            base_col2 = col2.split('_')[0]

            # Check for multicollinearity
            corr = df[col1].corr(df[col2])

            # Only add interactions where correlation is moderate
            if abs(corr) < 0.6:  # Lower threshold to be safer
                candidate_pairs.append((col1, col2, base_col1, base_col2, abs(corr)))

    # Sort by lowest correlation to minimize multicollinearity
    candidate_pairs.sort(key=lambda x: x[4])

    # Add the interactions, limiting to max_interactions
    interaction_columns = []
    for i, (col1, col2, base_col1, base_col2, _) in enumerate(candidate_pairs):
        if i >= max_interactions:
            break

        # Create interaction name using base names
        interaction_name = f"{base_col1}_{base_col2}_interaction"

        # Create interaction
        df[interaction_name] = df[col1] * df[col2]
        interaction_columns.append(interaction_name)
        print(f"Added interaction: {interaction_name}")

        # Check correlation with original variables
        corr1 = df[interaction_name].corr(df[col1])
        corr2 = df[interaction_name].corr(df[col2])
        print(f"  Correlation with {col1}: {corr1:.3f}")
        print(f"  Correlation with {col2}: {corr2:.3f}")

    return df, interaction_columns


def add_seasonality(df):
    """
    Add time-based effects and seasonality indicators.

    Args:
        df: DataFrame to modify

    Returns:
        Updated DataFrame and list of added seasonality columns
    """
    # Extract date components if date column exists
    if 'date' in df.columns:
        # Quarter dummies
        if 'quarter' not in df.columns:
            df['quarter'] = pd.to_datetime(df['date']).dt.quarter

        df['quarter_2'] = (df['quarter'] == 2).astype(int)
        df['quarter_3'] = (df['quarter'] == 3).astype(int)
        df['quarter_4'] = (df['quarter'] == 4).astype(int)

        # Time trend
        df['time_trend'] = np.arange(len(df))

        # Add holiday indicators
        holiday_dates = pd.to_datetime(df['date'])
        df['holiday_blackfriday'] = ((holiday_dates.dt.month == 11) & (holiday_dates.dt.day >= 20)).astype(int)
        df['holiday_christmas'] = (holiday_dates.dt.month == 12).astype(int)
        df['holiday_summer'] = (holiday_dates.dt.month.isin([7, 8])).astype(int)

        seasonal_columns = [
            'quarter_2', 'quarter_3', 'quarter_4',
            'time_trend',
            'holiday_blackfriday', 'holiday_christmas', 'holiday_summer'
        ]

        print(f"Added seasonality features: {', '.join(seasonal_columns)}")
        return df, seasonal_columns
    else:
        print("No date column found, couldn't add seasonality")
        return df, []


def check_multicollinearity(X):
    """
    Calculate Variance Inflation Factor (VIF) to detect multicollinearity.

    Args:
        X: DataFrame with features

    Returns:
        DataFrame with VIF values
    """
    # Add constant
    X_with_const = sm.add_constant(X)

    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_with_const.columns
    vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i)
                       for i in range(X_with_const.shape[1])]

    # Keep only features, remove constant
    vif_data = vif_data[vif_data['Feature'] != 'const'].reset_index(drop=True)

    # Sort by VIF
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

    # Print high VIF features
    high_vif = vif_data[vif_data['VIF'] > 10]
    if len(high_vif) > 0:
        print("\nHigh multicollinearity detected in features:")
        print(high_vif.to_string(index=False))

    return vif_data


def build_optimized_mmm_model(data, target='revenue', handle_multicollinearity=True,
                              check_target_stationarity=True):
    """
    Build an optimized MMM model incorporating all improvements.

    Args:
        data: DataFrame with media data
        target: Target variable name
        handle_multicollinearity: Whether to orthogonalize features
        check_target_stationarity: Whether to check and transform for stationarity

    Returns:
        Tuple of (MarketingMixModel, results, test_metrics)
    """
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

    # Make a copy to avoid modifying the original data
    data_copy = data.copy()

    # Check and handle target stationarity if requested
    target_transformed = target
    if check_target_stationarity:
        # Check stationarity
        is_stationary, adf_stat, p_value = check_stationarity(data_copy[target])

        if not is_stationary:
            print(f"\nTarget variable may not be stationary (ADF p-value: {p_value:.4f})")
            data_copy, target_transformed, transformation_info = make_stationary(
                data_copy, target, transformation_type='log'
            )
            print(f"Transformed target to {target_transformed}")
        else:
            print(f"\nTarget variable is stationary (ADF p-value: {p_value:.4f})")

    # Create log-transformed media variables
    for col in media_cols:
        if col in data_copy.columns:
            # Add small constant to avoid log(0)
            min_nonzero = max(data_copy[col][data_copy[col] > 0].min() * 0.1, 0.01)
            data_copy[f"{col}_log"] = np.log1p(data_copy[col] + min_nonzero)

    log_media_cols = [f"{col}_log" for col in media_cols if col in data_copy.columns]

    # Split data chronologically into train/test (80/20)
    if 'date' in data_copy.columns:
        data_copy = data_copy.sort_values('date')

    train_size = int(len(data_copy) * 0.8)
    train_data = data_copy.iloc[:train_size].copy()
    test_data = data_copy.iloc[train_size:].copy()

    print(f"Train set: {len(train_data)} observations")
    print(f"Test set: {len(test_data)} observations")

    # Initialize the model
    mmm = MarketingMixModel()
    mmm.load_data_from_dataframe(train_data)

    # Preprocess data with log-transformed media
    print("\nPreprocessing data...")
    mmm.preprocess_data(
        target=target_transformed,
        date_col='date' if 'date' in train_data.columns else None,
        media_cols=log_media_cols,
        control_cols=control_cols
    )

    # Apply adstock transformations with better defaults
    print("Applying adstock transformations...")

    # Use better, channel-specific decay rates
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

    # Apply adstock using our custom function
    adstocked_cols = []
    for col in log_media_cols:
        if col in decay_rates and col in max_lags:
            decayed_col = f"{col}_adstocked"
            mmm.preprocessed_data[decayed_col] = apply_adstock(
                mmm.preprocessed_data[col],
                decay_rate=decay_rates.get(col, 0.5),
                lag_weight=0.3,
                max_lag=max_lags.get(col, 4),
                decay_type='geometric'
            )

            # Replace original feature with adstocked version
            if col in mmm.feature_names:
                idx = mmm.feature_names.index(col)
                mmm.feature_names[idx] = decayed_col
            else:
                mmm.feature_names.append(decayed_col)

            adstocked_cols.append(decayed_col)
            print(
                f"Applied adstock to {col} with decay_rate={decay_rates.get(col, 0.5)}, max_lag={max_lags.get(col, 4)}")

    # Analyze media channel correlation
    print("\nAnalyzing media channel correlation...")
    correlation, high_corr_pairs = analyze_media_correlation(mmm.preprocessed_data[adstocked_cols])

    # Check multicollinearity
    vif_before = check_multicollinearity(mmm.preprocessed_data[mmm.feature_names])

    # Handle multicollinearity if requested
    if handle_multicollinearity and len(high_corr_pairs) > 0:
        # Identify features with high VIF
        high_vif_features = vif_before[vif_before['VIF'] > 10]['Feature'].tolist()

        if high_vif_features:
            print(f"\nOrthogonalizing {len(high_vif_features)} high-VIF features")

            # Apply orthogonalization
            mmm.preprocessed_data = orthogonalize_features(
                mmm.preprocessed_data,
                high_vif_features,
                method='qr'  # QR decomposition for orthogonalization
            )

            # Update feature names to use orthogonalized versions
            for i, feature in enumerate(mmm.feature_names):
                if feature in high_vif_features:
                    mmm.feature_names[i] = f"{feature}_ortho"

            # Check VIF after orthogonalization
            ortho_features = [f"{f}_ortho" if f in high_vif_features else f for f in mmm.feature_names]
            vif_after = check_multicollinearity(mmm.preprocessed_data[ortho_features])

            # Report improvement
            mean_vif_before = vif_before['VIF'].mean()
            mean_vif_after = vif_after['VIF'].mean()
            print(f"Mean VIF reduced from {mean_vif_before:.2f} to {mean_vif_after:.2f}")

    # Add careful interaction terms - limited to just 2
    print("\nAdding channel interactions...")
    # Use modified feature names (with orthogonalized versions if needed)
    current_features = mmm.feature_names.copy()
    media_features = [f for f in current_features if
                      any(m in f for m in ['tv', 'digital', 'search', 'social', 'video', 'email'])]

    mmm.preprocessed_data, interaction_cols = add_careful_interactions(
        mmm.preprocessed_data,
        media_features,
        target_transformed,
        max_interactions=2  # Limit to just 2 interactions to avoid multicollinearity
    )
    mmm.feature_names.extend(interaction_cols)

    # Add seasonality
    print("\nAdding seasonal effects...")
    mmm.preprocessed_data, seasonal_cols = add_seasonality(mmm.preprocessed_data)
    mmm.feature_names.extend(seasonal_cols)

    # Update X with new features
    mmm.X = mmm.preprocessed_data[mmm.feature_names]

    # Fit the model
    print("\nFitting the optimized model...")
    # Add constant for statsmodels
    mmm.X = sm.add_constant(mmm.X)
    mmm.feature_names = ['const'] + mmm.feature_names
    results = mmm.fit_model()

    print("\nModel Summary:")
    print(results.summary())

    # Calculate elasticities more carefully
    print("\nCalculating elasticities...")
    elasticities = {}

    # Map feature-to-original for elasticity calculation
    # This ensures log-transformed and adstocked features are mapped back
    feature_to_original = {}
    for col in mmm.feature_names:
        # Skip constant
        if col == 'const':
            continue

        # Map back to original media channel
        original_col = col
        for transform in ['_log', '_adstocked', '_ortho']:
            if transform in col:
                original_col = col.split(transform)[0]

        feature_to_original[col] = original_col

    for feature in mmm.feature_names:
        if feature == 'const':
            continue

        # Get original media channel
        original = feature_to_original.get(feature, feature)

        # Only calculate for media columns
        if not any(media in original for media in ['tv_spend', 'digital_display_spend', 'search_spend',
                                                   'social_media_spend', 'video_spend', 'email_spend']):
            continue

        # Get coefficient
        if feature in results.params:
            coef = results.params[feature]

            # Check if the feature is an interaction
            if 'interaction' in feature:
                # For interactions, use the coefficient * mean(X) / mean(y) formula
                feature_mean = mmm.X[feature].mean()
                target_mean = mmm.y.mean()
                elasticity = coef * feature_mean / target_mean
                calculation = f"coefficient ({coef:.2f}) * (mean X ({feature_mean:.2f}) / mean Y ({target_mean:.2f}))"
            elif '_log' in feature:
                # For log-transformed variables, coefficient is the elasticity
                elasticity = coef
                calculation = f"coefficient (direct elasticity)"
            else:
                # Linear variables
                feature_mean = mmm.X[feature].mean()
                target_mean = mmm.y.mean()
                elasticity = coef * feature_mean / target_mean
                calculation = f"coefficient ({coef:.2f}) * (mean X ({feature_mean:.2f}) / mean Y ({target_mean:.2f}))"

            # Store with detailed logs
            original_media = original.replace('_log', '')
            elasticities[original_media] = elasticity

            logger.info(f"Elasticity for {original_media} (using {feature}): {elasticity:.4f}")
            logger.info(f"  Calculation: {calculation}")

    # Print elasticity comparison
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

    # Evaluate on test set
    print("\nEvaluating on test set...")

    # Create a simpler model for test set evaluation
    test_features = control_cols + log_media_cols

    # Use simple model on training data for prediction
    train_X = sm.add_constant(mmm.preprocessed_data[test_features])
    train_y = mmm.preprocessed_data[target_transformed]

    simple_model = sm.OLS(train_y, train_X).fit()

    # Apply the same transformations to test data
    test_X = sm.add_constant(test_data[test_features])
    test_y = test_data[target_transformed] if target_transformed in test_data.columns else test_data[target]

    # Predict on test data
    test_predictions = simple_model.predict(test_X)

    # Calculate metrics
    test_r2 = r2_score(test_y, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_y, test_predictions))
    test_mape = mean_absolute_percentage_error(test_y, test_predictions) * 100

    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAPE: {test_mape:.2f}%")

    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    if 'date' in test_data.columns:
        plt.plot(test_data['date'], test_y, 'b-', label='Actual')
        plt.plot(test_data['date'], test_predictions, 'r--', label='Predicted')
        plt.xlabel('Date')
    else:
        plt.plot(test_y.index, test_y, 'b-', label='Actual')
        plt.plot(test_predictions.index, test_predictions, 'r--', label='Predicted')
        plt.xlabel('Index')

    plt.title('Test Set: Actual vs. Predicted Revenue')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_predictions.png')

    # Plot media coefficients from the full model
    media_effects = []

    for col in mmm.feature_names:
        if any(media in col for media in
               ['tv', 'digital', 'search', 'social', 'video', 'email']) and col in results.params:
            coef = results.params[col]
            p_val = results.pvalues[col]
            significant = p_val < 0.05

            media_effects.append({
                'Channel': col,
                'Coefficient': coef,
                'P-value': p_val,
                'Significant': significant
            })

    if media_effects:
        media_df = pd.DataFrame(media_effects)
        media_df = media_df.sort_values('Coefficient', key=abs, ascending=False)

        plt.figure(figsize=(12, 6))
        colors = ['green' if c > 0 else 'red' for c in media_df['Coefficient']]
        bars = plt.bar(media_df['Channel'], media_df['Coefficient'], color=colors)

        # Add significance markers
        for i, bar in enumerate(bars):
            if media_df.iloc[i]['Significant']:
                plt.text(bar.get_x() + bar.get_width() / 2,
                         0.001 + bar.get_height() if bar.get_height() > 0 else bar.get_height() - 0.001,
                         '*',
                         ha='center', va='bottom' if bar.get_height() > 0 else 'top',
                         fontsize=16)

        plt.title('Media Channel Coefficients')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('media_coefficients.png')

    # Return results
    test_metrics = {
        'r2': test_r2,
        'rmse': test_rmse,
        'mape': test_mape,
        'elasticities': elasticities
    }

    print("\nAnalysis complete. Output visualizations saved to:")
    print("- media_correlation.png")
    print("- test_predictions.png")
    print("- media_coefficients.png")

    return mmm, results, test_metrics


if __name__ == "__main__":
    start_time = time.time()

    # Load the data
    data_path = Path(__file__).parent / 'mock_marketing_data.csv'
    data = pd.read_csv(data_path)

    # Convert date to datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])

    print("=====================================")
    print("  MMM OPTIMIZATION ANALYSIS")
    print("=====================================")
    print(f"Data loaded: {len(data)} observations from {data['date'].min()} to {data['date'].max()}")

    # Build the optimized model
    mmm, results, metrics = build_optimized_mmm_model(
        data,
        target='revenue',
        handle_multicollinearity=True,  # Apply orthogonalization
        check_target_stationarity=True  # Check and handle stationarity
    )

    end_time = time.time()
    print(f"\nAnalysis completed in {(end_time - start_time) / 60:.1f} minutes")
