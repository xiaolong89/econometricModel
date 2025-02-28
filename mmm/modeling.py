"""
Advanced modeling approaches for Marketing Mix Models.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


def calculate_elasticities(model, X, y, feature_dict=None):
    """
    Calculate elasticities for media variables.

    Args:
        model: Fitted model (statsmodels or sklearn)
        X: Feature matrix
        y: Target variable
        feature_dict: Dictionary mapping feature types to column names

    Returns:
        DataFrame with elasticity calculations
    """
    results = []
    mean_y = y.mean()

    # Handle different model types
    if hasattr(model, 'params'):  # statsmodels
        coefficients = model.params
        feature_names = X.columns
    else:  # sklearn
        coefficients = model.coef_
        if len(coefficients.shape) > 1:
            coefficients = coefficients.flatten()
        feature_names = X.columns

    for i, feature in enumerate(feature_names):
        if feature == 'const':
            continue

        coef = coefficients[i] if hasattr(model, 'params') else coefficients[i]
        mean_x = X[feature].mean()

        # Skip if coefficient is 0
        if coef == 0:
            continue

        # Identify feature type
        feature_type = None
        if feature_dict:
            for ftype, cols in feature_dict.items():
                if feature in cols:
                    feature_type = ftype

        # Calculate elasticity based on feature type and transformation
        if 'log' in feature:
            # For log-transformed variables, elasticity is just the coefficient
            elasticity = coef
            elasticity_method = "coefficient (direct elasticity)"
        else:
            # For linear variables, elasticity is coefficient * mean(X) / mean(Y)
            elasticity = coef * mean_x / mean_y
            elasticity_method = f"coefficient ({coef:.2f}) * (mean X ({mean_x:.2f}) / mean Y ({mean_y:.2f}))"

        # Map back to original variable if needed
        original_feature = feature
        for transform in ['_log', '_hill', '_power', '_adstocked', '_ortho']:
            if transform in feature:
                original_feature = feature.split(transform)[0]
                break

        results.append({
            'feature': feature,
            'original_feature': original_feature,
            'feature_type': feature_type,
            'coefficient': coef,
            'elasticity': elasticity,
            'calculation': elasticity_method
        })

    # Create dataframe and sort by elasticity magnitude
    elasticity_df = pd.DataFrame(results)
    if not elasticity_df.empty:
        elasticity_df = elasticity_df.sort_values('elasticity', key=abs, ascending=False)

    return elasticity_df


def fit_ridge_model(X, y, alphas=None, cv=5, feature_names=None):
    """
    Fit Ridge regression model with cross-validation.

    Args:
        X: Feature matrix
        y: Target variable
        alphas: List of alpha values to try (if None, defaults to a range)
        cv: Number of cross-validation folds
        feature_names: List of feature names (optional)

    Returns:
        Tuple of (fitted model, metrics dictionary)
    """
    # Default alphas if not provided
    if alphas is None:
        alphas = np.logspace(-3, 5, 20)

    # Scale features for better regularization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names if feature_names else X.columns)

    # Use time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)

    # Fit model with cross-validation
    ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_scaled, y)

    # Get optimal alpha
    optimal_alpha = ridge_cv.alpha_
    logger.info(f"Ridge regression optimal alpha: {optimal_alpha}")

    # Final model with optimal alpha
    ridge = Ridge(alpha=optimal_alpha)
    ridge.fit(X_scaled, y)

    # Predictions and metrics
    y_pred = ridge.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = ridge_cv.score(X_scaled, y)

    # Calculate elasticities
    elasticities = calculate_elasticities(ridge, X_scaled_df, y)

    # Cross-validation scores
    cv_scores = cross_val_score(ridge, X_scaled, y, cv=tscv, scoring='r2')

    # Gather metrics
    metrics = {
        'model_type': 'Ridge',
        'optimal_alpha': optimal_alpha,
        'r_squared': r2,
        'rmse': rmse,
        'cv_r_squared': cv_scores.mean(),
        'cv_r_squared_std': cv_scores.std(),
        'coefficients': pd.DataFrame({
            'Feature': feature_names if feature_names else X.columns,
            'Coefficient': ridge.coef_
        }).sort_values('Coefficient', key=abs, ascending=False),
        'elasticities': elasticities
    }

    logger.info(f"Ridge Regression Results:")
    logger.info(f"  R-squared: {r2:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  CV R-squared: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return ridge, metrics, scaler


def fit_lasso_model(X, y, alphas=None, cv=5, feature_names=None):
    """
    Fit Lasso regression model with cross-validation.

    Args:
        X: Feature matrix
        y: Target variable
        alphas: List of alpha values to try (if None, defaults to a range)
        cv: Number of cross-validation folds
        feature_names: List of feature names (optional)

    Returns:
        Tuple of (fitted model, metrics dictionary)
    """
    # Default alphas if not provided
    if alphas is None:
        alphas = np.logspace(-3, 5, 20)

    # Scale features for better regularization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names if feature_names else X.columns)

    # Use time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)

    # Fit model with cross-validation
    lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=10000, selection='random')
    lasso_cv.fit(X_scaled, y)

    # Get optimal alpha
    optimal_alpha = lasso_cv.alpha_
    logger.info(f"Lasso regression optimal alpha: {optimal_alpha}")

    # Final model with optimal alpha
    lasso = Lasso(alpha=optimal_alpha, max_iter=10000)
    lasso.fit(X_scaled, y)

    # Predictions and metrics
    y_pred = lasso.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = lasso_cv.score(X_scaled, y)

    # Calculate elasticities
    elasticities = calculate_elasticities(lasso, X_scaled_df, y)

    # Cross-validation scores
    cv_scores = cross_val_score(lasso, X_scaled, y, cv=tscv, scoring='r2')

    # Count selected features
    selected_features = sum(lasso.coef_ != 0)

    # Gather metrics
    metrics = {
        'model_type': 'Lasso',
        'optimal_alpha': optimal_alpha,
        'r_squared': r2,
        'rmse': rmse,
        'cv_r_squared': cv_scores.mean(),
        'cv_r_squared_std': cv_scores.std(),
        'selected_features': selected_features,
        'total_features': X.shape[1],
        'coefficients': pd.DataFrame({
            'Feature': feature_names if feature_names else X.columns,
            'Coefficient': lasso.coef_
        }).sort_values('Coefficient', key=abs, ascending=False),
        'elasticities': elasticities
    }

    logger.info(f"Lasso Regression Results:")
    logger.info(f"  R-squared: {r2:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  CV R-squared: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"  Selected features: {selected_features} out of {X.shape[1]}")

    return lasso, metrics, scaler


def fit_constrained_model(X, y, feature_constraints=None):
    """
    Fit a constrained regression model to ensure valid elasticities.

    Args:
        X: Feature matrix
        y: Target variable
        feature_constraints: Dictionary mapping features to bounds (lower, upper)

    Returns:
        Tuple of (fitted model, metrics dictionary)
    """
    from scipy.optimize import minimize

    # Default constraints if not provided
    if feature_constraints is None:
        feature_constraints = {}

        # Set default positive constraints for media variables
        for col in X.columns:
            if any(keyword in col.lower() for keyword in ['spend', 'tv', 'digital', 'social', 'search']):
                feature_constraints[col] = (0, None)  # Lower bound 0, no upper bound

    # Objective function to minimize (sum of squared residuals)
    def objective(beta):
        beta_with_intercept = np.insert(beta, 0, 1)  # Add intercept
        y_pred = X_with_const.dot(beta_with_intercept)
        return np.sum((y - y_pred) ** 2)

    # Add constant to X
    X_with_const = sm.add_constant(X)

    # Initial values (OLS estimates)
    initial_model = sm.OLS(y, X_with_const).fit()
    initial_params = initial_model.params.values[1:]  # Exclude intercept

    # Set up bounds for constrained optimization
    bounds = []
    for j, col in enumerate(X.columns):
        if col in feature_constraints:
            bounds.append(feature_constraints[col])
        else:
            bounds.append((None, None))  # No bounds

    # Run optimization
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds
    )

    # Get optimal parameters
    optimal_params = np.insert(result.x, 0, 1)  # Add back intercept

    # Create model-like object for compatibility
    class ConstrainedModel:
        def __init__(self, params, X, y):
            self.params = pd.Series(params, index=X.columns)
            self.X = X
            self.y = y

        def predict(self, X_new):
            return X_new.dot(self.params)

    model = ConstrainedModel(optimal_params, X_with_const, y)

    # Calculate predictions and metrics
    y_pred = model.predict(X_with_const)
    sse = np.sum((y - y_pred) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (sse / sst)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    # Calculate elasticities
    elasticities = calculate_elasticities(model, X, y)

    # Gather metrics
    metrics = {
        'model_type': 'Constrained Regression',
        'r_squared': r2,
        'rmse': rmse,
        'constraint_satisfied': result.success,
        'coefficients': pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.params[1:]  # Exclude intercept
        }).sort_values('Coefficient', key=abs, ascending=False),
        'elasticities': elasticities
    }

    logger.info(f"Constrained Regression Results:")
    logger.info(f"  R-squared: {r2:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  Constraints satisfied: {result.success}")

    return model, metrics


def fit_pca_model(X, y, n_components=None, explained_variance=0.95, feature_names=None):
    """
    Fit a PCA-based model to address multicollinearity.

    Args:
        X: Feature matrix
        y: Target variable
        n_components: Number of components to use (if None, use explained_variance)
        explained_variance: Target explained variance (if n_components is None)
        feature_names: List of feature names (optional)

    Returns:
        Dictionary with fitted model and PCA information
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine number of components
    if n_components is None:
        # Fit PCA to determine number of components needed
        pca_test = PCA()
        pca_test.fit(X_scaled)
        cumulative_variance = np.cumsum(pca_test.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= explained_variance) + 1
        n_components = min(n_components, X.shape[1] - 1)  # Ensure we don't exceed max possible components

        logger.info(f"Selected {n_components} PCA components to explain {explained_variance * 100:.1f}% of variance")

    # Fit PCA with selected number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Create component names for clarity
    component_names = [f"Component_{i + 1}" for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=component_names)

    # Fit OLS on PCA components
    X_pca_with_const = sm.add_constant(X_pca_df)
    model = sm.OLS(y, X_pca_with_const).fit()

    # Calculate metrics
    y_pred = model.predict(X_pca_with_const)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Feature importance based on PCA loadings
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=feature_names if feature_names else X.columns
    )

    # Scale importance by explained variance ratio
    for i, col in enumerate(component_names):
        feature_importance[col] = feature_importance[col] * pca.explained_variance_ratio_[i]

    # Total importance across all components
    feature_importance['Total_Importance'] = feature_importance.abs().sum(axis=1)
    feature_importance = feature_importance.sort_values('Total_Importance', ascending=False)

    # Prepare result dictionary
    result = {
        'model': model,
        'pca': pca,
        'scaler': scaler,
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'total_explained_variance': pca.explained_variance_ratio_.sum(),
        'component_summary': pd.DataFrame({
            'Component': component_names,
            'Explained_Variance': pca.explained_variance_,
            'Explained_Variance_Ratio': pca.explained_variance_ratio_,
            'Cumulative_Variance_Ratio': np.cumsum(pca.explained_variance_ratio_)
        }),
        'feature_importance': feature_importance,
        'r_squared': model.rsquared,
        'adjusted_r_squared': model.rsquared_adj,
        'rmse': rmse
    }

    logger.info(f"PCA-OLS Model Results:")
    logger.info(f"  R-squared: {model.rsquared:.4f}")
    logger.info(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    return result


def evaluate_model(model, X_test, y_test, scaler=None, inverse_transform_func=None):
    """
    Evaluate a model on test data.

    Args:
        model: Fitted model (statsmodels or sklearn)
        X_test: Test features
        y_test: Test target values
        scaler: Feature scaler (if used during training)
        inverse_transform_func: Function to inverse transform predictions (if target was transformed)

    Returns:
        Dictionary of evaluation metrics
    """
    # Scale features if scaler provided
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    # Handle different model types for prediction
    if hasattr(model, 'predict'):
        # For sklearn models or statsmodels
        if hasattr(model, 'params') and not isinstance(X_test_scaled, pd.DataFrame):
            # For statsmodels with numpy array input
            X_test_const = sm.add_constant(X_test_scaled)
            y_pred = model.predict(X_test_const)
        else:
            # For sklearn or statsmodels with DataFrame input
            y_pred = model.predict(X_test_scaled)
    else:
        # For custom model objects
        X_test_const = sm.add_constant(X_test_scaled)
        y_pred = model.predict(X_test_const)

    # Inverse transform predictions if needed
    if inverse_transform_func:
        y_pred = inverse_transform_func(y_pred)
        # Assuming y_test might also need to be inverse transformed
        if hasattr(y_test, 'name') and '_log' in y_test.name:
            y_test_original = inverse_transform_func(y_test)
        else:
            y_test_original = y_test
    else:
        y_test_original = y_test

    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)

    # Calculate R-squared
    ss_total = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
    ss_residual = np.sum((y_test_original - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    # Calculate MAPE, handling zeros and small values
    y_test_masked = np.where(y_test_original < 0.01, 0.01, y_test_original)
    mape = mean_absolute_percentage_error(y_test_masked, y_pred)

    # Return metrics
    metrics = {
        'r_squared': r2,
        'rmse': rmse,
        'mse': mse,
        'mape': mape,
        'mean_predicted': np.mean(y_pred),
        'mean_actual': np.mean(y_test_original),
        'predictions': y_pred,
        'actuals': y_test_original
    }

    logger.info(f"Model Evaluation Results:")
    logger.info(f"  Test R-squared: {r2:.4f}")
    logger.info(f"  Test RMSE: {rmse:.4f}")
    logger.info(f"  Test MAPE: {mape:.4f}")

    return metrics


def map_elasticities_to_original_channels(model, pca, channel_groups, media_cols):
    """Transform PCA component elasticities back to original media channels."""

    # Get media component columns
    component_cols = [col for col in model.params.index if 'media_component' in col]

    # Extract component coefficients
    component_coeffs = {}
    for col in component_cols:
        if col in model.params:
            component_coeffs[col] = model.params[col]

    # Map back to standardized inputs using PCA loadings
    log_channel_elasticities = {}
    for col in media_cols:
        elasticity = 0
        for comp_idx, comp_name in enumerate(component_cols):
            if comp_name in component_coeffs:
                # Sum the contribution from each component
                elasticity += component_coeffs[comp_name] * pca.components_[comp_idx, media_cols.index(col)]
        log_channel_elasticities[col] = elasticity

    # Map from log channels to original channel groups
    channel_group_elasticities = {}
    for log_col, elasticity in log_channel_elasticities.items():
        # Remove _log suffix
        group = log_col.replace('_log', '')
        channel_group_elasticities[group] = elasticity

    # Map to original individual channels based on their weight in the group
    original_channel_elasticities = {}
    for group, channels in channel_groups.items():
        group_elasticity = channel_group_elasticities.get(group, 0)
        # Distribute elasticity equally among channels in the group
        # This is simplified - in a real implementation, we might weight by spend
        for channel in channels:
            original_channel_elasticities[channel] = group_elasticity / len(channels)

    # Create a DataFrame for the results
    import pandas as pd
    elasticity_df = pd.DataFrame({
        'Channel': list(original_channel_elasticities.keys()),
        'Elasticity': list(original_channel_elasticities.values())
    }).sort_values('Elasticity', ascending=False)

    return elasticity_df, original_channel_elasticities


def build_mmm(data, target='log_revenue'):
    """Build the MMM using OLS"""
    # Prepare model variables
    media_vars = ['traditional_media', 'digital_paid', 'social_media', 'owned_media']

    # Add control variables and seasonality if available
    control_vars = [col for col in data.columns if col.startswith('month_')]
    if 'price_index' in data.columns:
        control_vars.append('price_index')
    if 'competitor_price_index' in data.columns:
        control_vars.append('competitor_price_index')

    # Combine all predictors
    all_predictors = media_vars + control_vars

    # Create design matrix with constant
    X = sm.add_constant(data[all_predictors])
    y = data[target]

    # Fit OLS model
    model = sm.OLS(y, X).fit()

    # Check VIF for multicollinearity
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return model, vif_data


def validate_model(model, data, target='log_revenue'):
    """Validate model performance"""
    # Prepare features for prediction
    features = list(model.params.index)
    if 'const' in features:
        features.remove('const')
        X = sm.add_constant(data[features])
    else:
        X = data[features]

    # Make predictions
    preds = model.predict(X)

    # Convert back to original scale if using log transform
    if target == 'log_revenue':
        actual = np.exp(data[target])
        predicted = np.exp(preds)
    else:
        actual = data[target]
        predicted = preds

    # Calculate performance metrics
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # Visualize fit
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, actual, label='Actual')
    plt.plot(data.index, predicted, label='Predicted')
    plt.title(f'Model Fit - R²: {model.rsquared:.4f}, MAPE: {mape:.2f}%')
    plt.legend()
    plt.grid(True)

    # Residual plot
    plt.figure(figsize=(12, 6))
    residuals = actual - predicted
    plt.scatter(predicted, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)

    return {
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'mape': mape,
        'predictions': predicted,
        'actual': actual
    }


def apply_constraints(model, data, media_vars):
    """Apply non-negativity constraints to media coefficients if needed"""
    # Check if any media coefficients are negative
    negative_coefs = [var for var in media_vars if model.params[var] < 0]

    if not negative_coefs:
        print("No negative coefficients found. Using original model.")
        return model

    print(f"Found negative coefficients for: {negative_coefs}")

    # Apply constraints using scipy.optimize
    from scipy.optimize import minimize

    # Prepare data
    features = list(model.params.index)
    if 'const' in features:
        features.remove('const')
        X = sm.add_constant(data[features])
    else:
        X = data[features]

    y = data['log_revenue']

    # Objective function (sum of squared residuals)
    def objective(params):
        return np.sum((y - X.dot(params)) ** 2)

    # Initial values
    initial = model.params.values

    # Bounds: non-negative for media variables, unrestricted for others
    bounds = []
    for i, var in enumerate(model.params.index):
        if var in media_vars:
            bounds.append((0, None))  # Non-negative for media
        else:
            bounds.append((None, None))  # Unrestricted for others

    # Optimize
    result = minimize(objective, initial, bounds=bounds)

    # Create constrained model
    constrained_params = pd.Series(result.x, index=model.params.index)

    # Print comparison
    comparison = pd.DataFrame({
        'Original': model.params,
        'Constrained': constrained_params
    })
    print("\nCoefficient Comparison:")
    print(comparison)

    # Create a new model object with constrained parameters
    # (This is a simple approach - a more complete approach would recalculate all statistics)
    constrained_model = model
    constrained_model.params = constrained_params

    return constrained_model


# Only run this code when executing this file directly
if __name__ == "__main__":
    # Test code
    try:
        import pandas as pd
        from mmm.preprocessing import preprocess_data

        # Load test data
        df = pd.read_csv('data/synthetic_advertising_data_v2.csv')
        processed_data = preprocess_data(df)

        # Build and validate model
        model, vif_data = build_mmm(processed_data)
        validation_results = validate_model(model, processed_data)

        # Print results
        print("\nModel Summary:")
        print(model.summary())

        print("\nVIF Values:")
        print(vif_data.sort_values('VIF', ascending=False))

        print(f"\nModel Performance:")
        print(f"R²: {validation_results['r_squared']:.4f}")
        print(f"Adjusted R²: {validation_results['adj_r_squared']:.4f}")
        print(f"MAPE: {validation_results['mape']:.2f}%")

        # Check for negative coefficients and apply constraints if needed
        media_vars = ['traditional_media', 'digital_paid', 'social_media', 'owned_media']
        constrained_model = apply_constraints(model, processed_data, media_vars)

        # Validate constrained model if different from original
        if constrained_model is not model:
            constrained_validation = validate_model(constrained_model, processed_data)
            print("\nConstrained Model Performance:")
            print(f"R²: {constrained_validation['r_squared']:.4f}")
            print(f"MAPE: {constrained_validation['mape']:.2f}%")
    except Exception as e:
        print(f"Error in test code: {e}")
