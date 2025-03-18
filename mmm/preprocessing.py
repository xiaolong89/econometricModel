"""
Data preprocessing functions for Marketing Mix Modeling.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import logging
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt

from mmm.utils import parse_date_column

logger = logging.getLogger(__name__)


def detect_media_columns(df):
    """
    Automatically detect media columns in the dataset.

    Args:
        df: DataFrame to analyze

    Returns:
        List of detected media column names
    """
    potential_media_keywords = [
        'spend', 'marketing spend', 'units sold', 'units', 'tv', 'radio', 'digital', 'social',
        'search', 'display', 'video', 'email', 'print',
        'outdoor', 'media', 'facebook', 'google', 'twitter',
        'tiktok', 'youtube', 'programmatic', 'ad_'
    ]

    media_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in potential_media_keywords):
            # Exclude Date and known non-media columns
            if col not in ['Date', 'Sales', 'Revenue', 'Units Sold']:
                media_cols.append(col)

    return media_cols


def detect_control_columns(df, target_col, date_col, media_cols):
    """
    Automatically detect control columns in the dataset.

    Args:
        df: DataFrame to analyze
        target_col: Name of the target variable
        date_col: Name of the date column
        media_cols: List of media column names

    Returns:
        List of detected control column names
    """
    potential_control_keywords = [
        'price', 'promotion', 'discount', 'holiday',
        'season', 'temperature', 'weather', 'competitor',
        'economic', 'gdp', 'unemployment', 'income',
        'inflation', 'consumer', 'sentiment', 'confidence'
    ]

    control_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in potential_control_keywords):
            if col != target_col and col != date_col and col not in media_cols:
                control_cols.append(col)

    return control_cols


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
        is_log_stationary, _, _ = check_stationarity(transformed_df[log_col])
        transformation_info['log'] = log_col

        if is_log_stationary:
            logger.info(f"Log transformation made {target_col} stationary.")
            return transformed_df, log_col, transformation_info

    # Try differencing (or log differencing)
    if transformation_type in ['diff', 'log_diff']:
        if transformation_type == 'diff':
            diff_col = f"{target_col}_diff"
            transformed_df[diff_col] = df[target_col].diff().fillna(0)
        else:
            diff_col = f"{target_col}_log_diff"
            transformed_df[diff_col] = transformed_df[f"{target_col}_log"].diff().fillna(0)

        is_diff_stationary, _, _ = check_stationarity(transformed_df[diff_col][1:])  # Skip first row
        transformation_info['diff'] = diff_col

        if is_diff_stationary:
            logger.info(f"Differencing made {target_col} stationary.")
            return transformed_df, diff_col, transformation_info

    # If all attempts failed, use the last transformation tried
    if transformation_type == 'log_diff':
        logger.warning("Could not achieve stationarity. Using log differencing anyway.")
        return transformed_df, f"{target_col}_log_diff", transformation_info
    elif transformation_type == 'log':
        logger.warning("Could not achieve stationarity. Using log transformation anyway.")
        return transformed_df, f"{target_col}_log", transformation_info
    else:
        logger.warning("Could not achieve stationarity. Using differencing anyway.")
        return transformed_df, f"{target_col}_diff", transformation_info


def add_seasonality_features(df, date_col=None):
    """
    Add seasonality features to the dataset.

    Args:
        df: DataFrame to modify
        date_col: Name of the date column (defaults to 'Date' or 'date')

    Returns:
        DataFrame with added seasonality features
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Determine date column if not specified
    if date_col is None:
        date_col = 'Date' if 'Date' in df.columns else 'date'

    # Convert date column to datetime if it's not already
    if pd.api.types.is_string_dtype(result_df[date_col]):
        result_df[date_col] = pd.to_datetime(result_df[date_col], format='%m/%d/%Y')

    # Extract datetime components
    result_df['month'] = result_df[date_col].dt.month
    result_df['quarter'] = result_df[date_col].dt.quarter
    result_df['year'] = result_df[date_col].dt.year
    result_df['week_of_year'] = result_df[date_col].dt.isocalendar().week

    # Add time trend
    result_df['time_trend'] = range(len(result_df))

    # Create quarter dummies (drop first to avoid multicollinearity)
    quarter_dummies = pd.get_dummies(result_df['quarter'], prefix='quarter', drop_first=True)
    result_df = pd.concat([result_df, quarter_dummies], axis=1)

    # Holiday periods
    # Define common holiday periods
    result_df['holiday_blackfriday'] = ((result_df['month'] == 11) & (result_df[date_col].dt.day >= 20)).astype(int)
    result_df['holiday_christmas'] = ((result_df['month'] == 12) & (result_df[date_col].dt.day >= 10)).astype(int)
    result_df['holiday_summer'] = ((result_df['month'] >= 6) & (result_df['month'] <= 8)).astype(int)

    return result_df


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
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == 'qr':
        # QR decomposition for orthogonalization
        Q, R = np.linalg.qr(X_scaled)

        # Create orthogonalized features
        for i, col in enumerate(feature_cols):
            result_df[f"{col}_ortho"] = Q[:, i]

        # Log correlation of orthogonalized features
        ortho_cols = [f"{col}_ortho" for col in feature_cols]
        corr_matrix = result_df[ortho_cols].corr()
        logger.info(f"Max correlation between orthogonalized features: {corr_matrix.abs().max().max()}")

    elif method == 'residualization':
        # Sequential residualization (Gram-Schmidt-like process)
        feature_importance = {}

        # First, get a rough estimate of feature importance
        for col in feature_cols:
            X = sm.add_constant(df[col])
            model = sm.OLS(df[df.columns[-1]], X).fit()  # Assuming target is last column
            feature_importance[col] = model.rsquared

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        sorted_feature_names = [item[0] for item in sorted_features]

        # Keep first feature as is (most important)
        result_df[f"{sorted_feature_names[0]}_ortho"] = df[sorted_feature_names[0]]

        # Residualize each subsequent feature
        for i in range(1, len(sorted_feature_names)):
            # Current feature
            curr_feature = sorted_feature_names[i]

            # Features to regress against (previous orthogonalized features)
            prev_ortho_features = [f"{feat}_ortho" for feat in sorted_feature_names[:i]]

            # Residualize
            X = sm.add_constant(result_df[prev_ortho_features])
            model = sm.OLS(df[curr_feature], X).fit()

            # Store residuals as orthogonalized feature
            result_df[f"{curr_feature}_ortho"] = model.resid

    return result_df


def check_multicollinearity(df, feature_cols, vif_threshold=10):
    """
    Check for multicollinearity using Variance Inflation Factor (VIF).

    Args:
        df: DataFrame with features
        feature_cols: List of feature columns to check
        vif_threshold: Threshold above which to flag high VIF

    Returns:
        DataFrame with VIF values for each feature
    """
    # Create a DataFrame to store VIF values
    vif_data = pd.DataFrame()
    vif_data['Feature'] = feature_cols

    # Calculate VIF for each feature
    vif_values = []
    for i, feature in enumerate(feature_cols):
        # Features excluding the current one
        other_features = feature_cols.copy()
        other_features.remove(feature)

        # Regression of current feature on all other features
        y = df[feature]
        X = sm.add_constant(df[other_features])

        # Handle perfect multicollinearity case
        try:
            model = sm.OLS(y, X).fit()
            r_squared = model.rsquared
            vif = 1 / (1 - r_squared)
        except:
            vif = float('inf')  # Indicates perfect multicollinearity

        vif_values.append(vif)

    vif_data['VIF'] = vif_values

    # Flag high VIF values
    high_vif = vif_data[vif_data['VIF'] > vif_threshold]
    if len(high_vif) > 0:
        logger.warning(f"High multicollinearity detected in features:")
        for _, row in high_vif.iterrows():
            logger.warning(f"  {row['Feature']}: VIF = {row['VIF']:.2f}")

    return vif_data


def preprocess_for_modeling(df, target=None, date_col=None, media_cols=None, control_cols=None,
                            make_stationary_flag=True, orthogonalize=True, add_seasonality=True):
    """
    Complete preprocessing pipeline for modeling.

    Args:
        df: DataFrame to process
        target: Name of the target variable (defaults to 'Sales' or 'Revenue')
        date_col: Name of the date column (optional, defaults to 'Date' or 'date')
        media_cols: List of media column names (optional)
        control_cols: List of control column names (optional)
        make_stationary_flag: Whether to transform target for stationarity
        orthogonalize: Whether to orthogonalize features
        add_seasonality: Whether to add seasonality features

    Returns:
        Tuple of (preprocessed_df, X, y, feature_names)
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Determine date column if not specified
    if date_col is None:
        date_col = 'Date' if 'Date' in processed_df.columns else 'date'

    # Handle date column
    if date_col in processed_df.columns:
        processed_df[date_col] = pd.to_datetime(processed_df[date_col], format='%m/%d/%Y')
        processed_df = processed_df.sort_values(by=date_col)

    # Determine target column if not specified
    if target is None:
        target = 'Sales' if 'Sales' in processed_df.columns else 'Revenue'

    # Auto-detect columns if not provided
    if media_cols is None:
        media_cols = detect_media_columns(processed_df)
        logger.info(f"Automatically identified media columns: {media_cols}")

    if control_cols is None and date_col:
        control_cols = detect_control_columns(processed_df, target, date_col, media_cols)
        logger.info(f"Automatically identified control columns: {control_cols}")

    # Add seasonality features if requested
    if add_seasonality and date_col:
        processed_df = add_seasonality_features(processed_df, date_col)
        logger.info("Added seasonality features")

    # Apply transformations to media variables for diminishing returns
    if media_cols:
        processed_df, transformed_media_cols = apply_diminishing_returns_transformations(
            processed_df, media_cols, method='log')
        logger.info(f"Applied diminishing returns transformations to media columns")
    else:
        transformed_media_cols = []

    # Make target stationary if requested
    if make_stationary_flag:
        processed_df, target_transformed, transformation_info = make_stationary(
            processed_df, target, transformation_type='log')
        logger.info(f"Transformed target to {target_transformed}")
    else:
        target_transformed = target

    # Combine feature names
    all_feature_cols = (transformed_media_cols +
                        (control_cols if control_cols else []) +
                        [col for col in processed_df.columns if col.startswith('quarter_') or
                         col.startswith('holiday_')])

    # Check for missing values
    missing_values = processed_df[all_feature_cols + [target_transformed]].isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")

        # Simple imputation
        processed_df[all_feature_cols + [target_transformed]] = processed_df[
            all_feature_cols + [target_transformed]].fillna(
            processed_df[all_feature_cols + [target_transformed]].mean())

        logger.info("Missing values imputed with column means")

    # Check multicollinearity before orthogonalization
    vif_before = check_multicollinearity(processed_df, all_feature_cols)

    # Orthogonalize features if requested
    if orthogonalize:
        high_vif_features = vif_before[vif_before['VIF'] > 10]['Feature'].tolist()
        if high_vif_features:
            processed_df = orthogonalize_features(processed_df, high_vif_features)
            logger.info(f"Orthogonalized {len(high_vif_features)} high-VIF features")

            # Update feature names to use orthogonalized versions
            all_feature_cols = [f"{col}_ortho" if col in high_vif_features else col
                                for col in all_feature_cols]

    # Create X and y
    X = processed_df[all_feature_cols]
    y = processed_df[target_transformed]

    logger.info("Data preprocessing completed successfully")

    return processed_df, X, y, all_feature_cols


def apply_diminishing_returns_transformations(df, media_cols, method='log'):
    """
    Apply transformations to capture diminishing returns effect in media spend.

    Args:
        df: DataFrame with marketing data
        media_cols: List of media spending columns
        method: Transformation method ('log', 'hill', or 'power')

    Returns:
        DataFrame with added transformed columns
    """
    result_df = df.copy()
    transformed_cols = []

    for col in media_cols:
        if method == 'log':
            # Log transformation (simplest approach)
            # Add small constant to avoid log(0)
            min_nonzero = max(df[col][df[col] > 0].min() * 0.1, 0.01)
            result_df[f'{col}_log'] = np.log1p(df[col] + min_nonzero)
            transformed_cols.append(f'{col}_log')

        elif method == 'hill':
            # Hill function (more flexible, better for S-shaped responses)
            # Parameters can be tuned per channel if needed
            scale = df[col].mean() * 2  # Scale parameter
            shape = 0.7  # Shape parameter (controls curve steepness)
            result_df[f'{col}_hill'] = df[col] ** shape / (scale ** shape + df[col] ** shape)
            transformed_cols.append(f'{col}_hill')

        elif method == 'power':
            # Power transformation (simple but effective)
            power = 0.7  # Typical value for media
            min_nonzero = max(df[col][df[col] > 0].min() * 0.1, 0.01)
            result_df[f'{col}_power'] = (df[col] + min_nonzero) ** power
            transformed_cols.append(f'{col}_power')

    return result_df, transformed_cols


def add_seasonality(df):
    """
    Add time-based effects and seasonality indicators.

    Args:
        df: DataFrame to modify

    Returns:
        Updated DataFrame and list of added seasonality columns
    """
    import pandas as pd
    import numpy as np

    # Extract date components if date column exists
    if 'date' in df.columns:
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Quarter dummies
        if 'quarter' not in result_df.columns:
            result_df['quarter'] = pd.to_datetime(result_df['date']).dt.quarter

        result_df['quarter_2'] = (result_df['quarter'] == 2).astype(int)
        result_df['quarter_3'] = (result_df['quarter'] == 3).astype(int)
        result_df['quarter_4'] = (result_df['quarter'] == 4).astype(int)

        # Time trend
        result_df['time_trend'] = np.arange(len(result_df))

        # Add holiday indicators
        holiday_dates = pd.to_datetime(result_df['date'])
        result_df['holiday_blackfriday'] = ((holiday_dates.dt.month == 11) & (holiday_dates.dt.day >= 20)).astype(int)
        result_df['holiday_christmas'] = (holiday_dates.dt.month == 12).astype(int)
        result_df['holiday_summer'] = (holiday_dates.dt.month.isin([7, 8])).astype(int)

        seasonal_columns = [
            'quarter_2', 'quarter_3', 'quarter_4',
            'time_trend',
            'holiday_blackfriday', 'holiday_christmas', 'holiday_summer'
        ]

        return result_df, seasonal_columns
    else:
        print("No date column found, couldn't add seasonality")
        return df, []


def preprocess_data(df):
    """
    Apply basic preprocessing to prepare data for modeling.

    Args:
        df: DataFrame containing the data

    Returns:
        DataFrame with preprocessed data
    """
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Use consistent date parsing
    if 'Date' in data.columns:
        data = parse_date_column(data)
        data.set_index('Date', inplace=True)

    # Convert date to datetime if needed (updated to handle both 'date' and 'Date')
    date_column = 'Date' if 'Date' in data.columns else 'date'
    if date_column in data.columns and data[date_column].dtype == 'object':
        data[date_column] = pd.to_datetime(data[date_column], format='%m/%d/%Y')
        data.set_index(date_column, inplace=True)

    # Log transform revenue or sales (for stationarity & to handle skewness)
    if 'revenue' in data.columns:
        data['log_revenue'] = np.log(data['revenue'])
    elif 'Sales' in data.columns:
        data['log_sales'] = np.log(data['Sales'])

    # Log transform units sold if present
    if 'Units Sold' in data.columns:
        data['log_units'] = np.log(data['Units Sold'])

    # Simple adstock transformations for media channels
    media_channels = ['tv_spend', 'search_spend', 'social_spend', 'display_spend', 'email_spend']
    adstock_params = {
        'tv_spend': 0.7,  # Slower decay for TV
        'search_spend': 0.3,  # Fast decay for search
        'social_spend': 0.5,  # Medium decay for social
        'display_spend': 0.4,  # Medium decay for display
        'email_spend': 0.4  # Medium decay for email
    }

    # Apply adstock to each channel
    for channel in media_channels:
        if channel in data.columns:
            data[f"{channel}_adstock"] = apply_adstock(data[channel], adstock_params[channel])

    # Aggregate channels to reduce multicollinearity
    data['traditional_media'] = data.get('tv_spend_adstock', 0)
    data['digital_paid'] = (data.get('search_spend_adstock', 0) +
                            data.get('display_spend_adstock', 0))
    data['social_media'] = data.get('social_spend_adstock', 0)
    data['owned_media'] = data.get('email_spend_adstock', 0)

    # Add any seasonal indicators if needed
    date_column = 'Date' if 'Date' in data.index.names else 'date'
    if date_column in data.index.names:
        data['month'] = data.index.month
        # Convert to dummy variables
        month_dummies = pd.get_dummies(data['month'], prefix='month', drop_first=True)
        data = pd.concat([data, month_dummies], axis=1)

    return data


def apply_adstock(series, decay_rate=0.5, max_lag=8):
    """
    Apply adstock transformation to a series.

    Args:
        series: Time series to apply adstock to
        decay_rate: Rate at which effect decays over time (0-1)
        max_lag: Maximum number of lagged periods to consider

    Returns:
        Series with adstock transformation applied
    """
    result = series.copy()
    for lag in range(1, max_lag + 1):
        # Shift and apply decay
        lagged = series.shift(lag) * (decay_rate ** lag)
        # Add to result, handling NAs
        result = result.add(lagged, fill_value=0)
    return result


# Only run this code when this file is executed directly, not when imported
if __name__ == "__main__":
    # Example usage and visualization
    try:
        import matplotlib.pyplot as plt

        # Load example data
        df = pd.read_csv('C:\_econometricModel\data\mmm_data.csv')

        # Apply preprocessing
        processed_data = preprocess_data(df)

        # Visualize key series
        plt.figure(figsize=(12, 8))

        # Original revenue vs log-transformed
        plt.subplot(2, 2, 1)
        plt.plot(processed_data.index, processed_data['revenue'], label='Revenue')
        plt.title('Original Revenue')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(processed_data.index, processed_data['log_revenue'], label='Log Revenue')
        plt.title('Log-Transformed Revenue')
        plt.grid(True)

        # Compare original vs adstocked media for one channel
        plt.subplot(2, 2, 3)
        plt.plot(processed_data.index, processed_data['tv_spend'], label='TV Spend')
        plt.title('Original TV Spend')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(processed_data.index, processed_data['tv_spend_adstock'], label='TV Adstock')
        plt.title('TV Spend with Adstock')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Inspect multicollinearity between aggregated channels
        correlation_matrix = processed_data[['traditional_media', 'digital_paid',
                                             'social_media', 'owned_media']].corr()
        print("\nCorrelation Matrix of Aggregated Channels:")
        print(correlation_matrix)

    except Exception as e:
        print(f"Error in example code: {e}")
