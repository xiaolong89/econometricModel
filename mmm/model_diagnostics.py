"""
Model Diagnostics for Marketing Mix Models.

This module provides comprehensive diagnostic tools for validating MMM models,
including residual analysis, stability assessment, sensitivity testing, and
advanced validation techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import itertools
import warnings
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress some common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', 'Maximum number of iterations has been exceeded')


# ---------------------------------------------------------------------------------------
# RESIDUAL ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------------------

def analyze_residuals(model, X, y, plot=True, test_autocorr=True, test_heteroskedasticity=True):
    """
    Comprehensive analysis of model residuals.

    Args:
        model: Fitted statsmodels OLS model
        X: Feature matrix used for prediction
        y: Actual target values
        plot: Whether to create visualization plots
        test_autocorr: Whether to test for autocorrelation
        test_heteroskedasticity: Whether to test for heteroskedasticity

    Returns:
        Dictionary with residual analysis results
    """
    # Add constant if needed
    if isinstance(X, pd.DataFrame) and 'const' not in X.columns and not sm.add_constant(X).equals(X):
        X_pred = sm.add_constant(X)
    else:
        X_pred = X

    # Get predictions and residuals
    preds = model.predict(X_pred)
    residuals = y - preds

    # Initialize results dictionary
    results = {
        'residuals': residuals,
        'predictions': preds,
        'residual_statistics': {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max()
        }
    }

    # Normality test (Shapiro-Wilk)
    try:
        shapiro_test = stats.shapiro(residuals)
        results['normality_test'] = {
            'statistic': shapiro_test.statistic,
            'p_value': shapiro_test.pvalue,
            'is_normal': shapiro_test.pvalue > 0.05
        }
    except Exception as e:
        logger.warning(f"Error in normality test: {str(e)}")
        results['normality_test'] = {
            'error': str(e)
        }

    # Autocorrelation test (Durbin-Watson)
    if test_autocorr:
        try:
            dw_statistic = durbin_watson(residuals)
            results['autocorrelation_test'] = {
                'durbin_watson': dw_statistic,
                'has_autocorrelation': dw_statistic < 1.5 or dw_statistic > 2.5
            }

            # Ljung-Box test for autocorrelation (up to lag 10 or n/5, whichever is smaller)
            max_lag = min(10, len(residuals) // 5)
            if max_lag > 1:  # Need at least 2 lags for the test
                lb_test = acorr_ljungbox(residuals, lags=max_lag)
                results['ljung_box_test'] = {
                    'statistic': lb_test.iloc[0, 0],
                    'p_value': lb_test.iloc[0, 1],
                    'has_autocorrelation': lb_test.iloc[0, 1] < 0.05
                }
        except Exception as e:
            logger.warning(f"Error in autocorrelation test: {str(e)}")
            results['autocorrelation_test'] = {
                'error': str(e)
            }

    # Heteroskedasticity test (Breusch-Pagan)
    if test_heteroskedasticity:
        try:
            bp_test = het_breuschpagan(residuals, X_pred)
            results['heteroskedasticity_test'] = {
                'lm_statistic': bp_test[0],
                'lm_p_value': bp_test[1],
                'f_statistic': bp_test[2],
                'f_p_value': bp_test[3],
                'has_heteroskedasticity': bp_test[1] < 0.05
            }
        except Exception as e:
            logger.warning(f"Error in heteroskedasticity test: {str(e)}")
            results['heteroskedasticity_test'] = {
                'error': str(e)
            }

    # Create plots if requested
    if plot:
        try:
            # Create subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))

            # 1. Residuals vs Predicted
            axs[0, 0].scatter(preds, residuals, alpha=0.5)
            axs[0, 0].axhline(y=0, color='r', linestyle='-')
            axs[0, 0].set_xlabel('Predicted Values')
            axs[0, 0].set_ylabel('Residuals')
            axs[0, 0].set_title('Residuals vs Predicted')
            axs[0, 0].grid(True, alpha=0.3)

            # Add a lowess smoothed line if possible
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                z = lowess(residuals, preds)
                axs[0, 0].plot(z[:, 0], z[:, 1], 'r--', linewidth=1)
            except:
                pass

            # 2. Histogram of residuals
            axs[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
            axs[0, 1].set_xlabel('Residual Value')
            axs[0, 1].set_ylabel('Frequency')
            axs[0, 1].set_title('Histogram of Residuals')

            # Add normal distribution curve
            x = np.linspace(min(residuals), max(residuals), 100)
            axs[0, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()) * len(residuals) * (
                        max(residuals) - min(residuals)) / 20,
                           'r-', linewidth=2)

            # 3. Q-Q Plot
            stats.probplot(residuals, plot=axs[1, 0])
            axs[1, 0].set_title('Q-Q Plot of Residuals')

            # 4. Residuals over index (time series pattern check)
            axs[1, 1].plot(range(len(residuals)), residuals)
            axs[1, 1].axhline(y=0, color='r', linestyle='-')
            axs[1, 1].set_xlabel('Observation Index')
            axs[1, 1].set_ylabel('Residuals')
            axs[1, 1].set_title('Residuals Over Time')
            axs[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.suptitle("Residual Analysis", fontsize=16)
            plt.subplots_adjust(top=0.92)

            # Save plot
            plt.savefig('residual_analysis.png')
            results['plots'] = {'residual_analysis': 'residual_analysis.png'}
        except Exception as e:
            logger.warning(f"Error creating residual plots: {str(e)}")

    # Format results for easier interpretation
    results['summary'] = {
        'mean_residual': f"{results['residual_statistics']['mean']:.4f}",
        'residual_issues': []
    }

    # Check if residuals show any issues
    if 'normality_test' in results and not results['normality_test'].get('is_normal', True):
        results['summary']['residual_issues'].append(
            f"Non-normal residuals (p={results['normality_test']['p_value']:.4f})"
        )

    if 'autocorrelation_test' in results and results['autocorrelation_test'].get('has_autocorrelation', False):
        results['summary']['residual_issues'].append(
            f"Autocorrelation detected (DW={results['autocorrelation_test']['durbin_watson']:.4f})"
        )

    if 'heteroskedasticity_test' in results and results['heteroskedasticity_test'].get('has_heteroskedasticity', False):
        results['summary']['residual_issues'].append(
            f"Heteroskedasticity detected (p={results['heteroskedasticity_test']['lm_p_value']:.4f})"
        )

    return results


def plot_coefficient_stability(model, X, y, feature_names=None):
    """
    Plot coefficients with confidence intervals to visualize stability.

    Args:
        model: Fitted statsmodels OLS model
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names (if None, use model.params.index)

    Returns:
        Matplotlib figure
    """
    # Get coefficient names and values
    if feature_names is None:
        feature_names = model.params.index

    # Exclude intercept for plotting
    if 'const' in feature_names:
        coef_names = [f for f in feature_names if f != 'const']
    else:
        coef_names = feature_names

    coef_values = [model.params[name] for name in coef_names]

    # Get confidence intervals
    conf_int = model.conf_int()
    lower_bounds = [conf_int.loc[name, 0] for name in coef_names]
    upper_bounds = [conf_int.loc[name, 1] for name in coef_names]

    # Calculate confidence interval width for error bars
    error_below = [coef - lower for coef, lower in zip(coef_values, lower_bounds)]
    error_above = [upper - coef for coef, upper in zip(coef_values, upper_bounds)]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort coefficients by absolute value
    indices = np.argsort(np.abs(coef_values))[::-1]
    sorted_names = [coef_names[i] for i in indices]
    sorted_coefs = [coef_values[i] for i in indices]
    sorted_err_below = [error_below[i] for i in indices]
    sorted_err_above = [error_above[i] for i in indices]

    # Plot coefficients with error bars
    colors = ['green' if c > 0 else 'red' for c in sorted_coefs]
    ax.errorbar(
        range(len(sorted_coefs)),
        sorted_coefs,
        yerr=[sorted_err_below, sorted_err_above],
        fmt='o',
        capsize=5,
        color=colors,
        ecolor='black',
        linewidth=1,
        elinewidth=1,
        markeredgewidth=1
    )

    # Add labels and grid
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('Coefficient Stability Analysis')
    ax.set_ylabel('Coefficient Value')
    ax.grid(True, axis='y', alpha=0.3)

    # Check if any confidence intervals cross zero
    for i, name in enumerate(sorted_names):
        if (lower_bounds[indices[i]] < 0 and upper_bounds[indices[i]] > 0):
            ax.plot(i, sorted_coefs[i], 'ko', markersize=10, alpha=0.3)

    # Add note about circles indicating statistical insignificance
    ax.annotate('* Black circles indicate statistically insignificant coefficients (CI crosses zero)',
                xy=(0, -0.15), xycoords='axes fraction', fontsize=10)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------
# STABILITY ASSESSMENT FUNCTIONS
# ---------------------------------------------------------------------------------------

def rolling_window_analysis(df, target, features, window_size=None, step_size=1, model_type='linear-linear'):
    """
    Perform rolling window analysis to assess coefficient stability over time.

    Args:
        df: DataFrame with the data
        target: Target variable name
        features: List of feature variable names
        window_size: Size of the rolling window (default: 1/3 of data)
        step_size: Number of observations to move forward each step
        model_type: Type of model ('linear-linear', 'log-log', etc.)

    Returns:
        Dictionary with rolling window analysis results
    """
    # Set default window size if not provided
    if window_size is None:
        window_size = max(len(df) // 3, 10)  # At least 10 observations

    if window_size >= len(df):
        raise ValueError(f"Window size ({window_size}) must be smaller than the data length ({len(df)})")

    # Pre-transform data if using log-log model
    if model_type == 'log-log':
        df_transformed = df.copy()
        df_transformed[target] = np.log1p(df[target])
        for feature in features:
            df_transformed[feature] = np.log1p(df[feature])
    else:
        df_transformed = df.copy()

    # Initialize storage for coefficients and metrics
    windows = []
    coefficients = {feature: [] for feature in features}
    coefficients['const'] = []  # For intercept
    r_squared = []
    predictions = []

    # Loop through windows
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        windows.append((start, end))

        # Get window data
        window_data = df_transformed.iloc[start:end]

        # Prepare X and y
        X = sm.add_constant(window_data[features])
        y = window_data[target]

        # Fit model
        try:
            model = sm.OLS(y, X).fit()

            # Store coefficients
            for feature in features:
                coefficients[feature].append(model.params[feature])
            coefficients['const'].append(model.params['const'])

            # Store R-squared
            r_squared.append(model.rsquared)

            # Make predictions for the next observation if available
            if end < len(df):
                if model_type == 'log-log':
                    X_next = sm.add_constant(np.log1p(df.iloc[end:end + 1][features]))
                    pred = np.expm1(model.predict(X_next))
                else:
                    X_next = sm.add_constant(df.iloc[end:end + 1][features])
                    pred = model.predict(X_next)
                predictions.append((end, pred[0], df.iloc[end][target]))
        except Exception as e:
            logger.warning(f"Error fitting model for window {start}-{end}: {str(e)}")
            # Fill with NaN for this window
            for feature in features:
                coefficients[feature].append(np.nan)
            coefficients['const'].append(np.nan)
            r_squared.append(np.nan)

    # Convert to DataFrames for easier analysis
    coef_df = pd.DataFrame(coefficients, index=[w[0] for w in windows])
    r2_df = pd.Series(r_squared, index=[w[0] for w in windows])

    # Create predictions DataFrame if we have predictions
    if predictions:
        pred_df = pd.DataFrame(predictions, columns=['index', 'predicted', 'actual'])
    else:
        pred_df = pd.DataFrame(columns=['index', 'predicted', 'actual'])

    # Calculate coefficient stability metrics
    stability_metrics = {}
    for feature in features + ['const']:
        stability_metrics[feature] = {
            'mean': coef_df[feature].mean(),
            'std': coef_df[feature].std(),
            'cv': coef_df[feature].std() / abs(coef_df[feature].mean()) if coef_df[feature].mean() != 0 else np.nan,
            'min': coef_df[feature].min(),
            'max': coef_df[feature].max(),
            'range': coef_df[feature].max() - coef_df[feature].min()
        }

    # Create plots
    fig, axs = plt.subplots(len(features) + 2, 1, figsize=(12, 3 * (len(features) + 2)))

    # Plot coefficients
    for i, feature in enumerate(features + ['const']):
        if i < len(axs) - 1:  # Make sure we have enough subplots
            ax = axs[i]
            ax.plot(coef_df.index, coef_df[feature])
            ax.set_ylabel(feature)
            if i == 0:
                ax.set_title('Coefficient Stability Over Rolling Windows')
            ax.grid(True, alpha=0.3)

            # Add original coefficient as reference line
            try:
                X_full = sm.add_constant(df_transformed[features])
                model_full = sm.OLS(df_transformed[target], X_full).fit()
                full_coef = model_full.params[feature]
                ax.axhline(y=full_coef, color='r', linestyle='--',
                           label=f'Full model: {full_coef:.4f}')
                ax.legend()
            except:
                pass

    # Plot R-squared
    axs[-1].plot(r2_df.index, r2_df.values)
    axs[-1].set_ylabel('R²')
    axs[-1].set_xlabel('Window Start Index')
    axs[-1].grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig('coefficient_stability.png')

    # Calculate out-of-sample prediction accuracy if we have predictions
    if len(pred_df) > 0:
        oos_rmse = np.sqrt(mean_squared_error(pred_df['actual'], pred_df['predicted']))
        oos_mape = mean_absolute_percentage_error(pred_df['actual'], pred_df['predicted']) * 100

        # Plot predictions vs actuals
        plt.figure(figsize=(10, 6))
        plt.plot(pred_df['index'], pred_df['actual'], 'b-', label='Actual')
        plt.plot(pred_df['index'], pred_df['predicted'], 'r--', label='Predicted')
        plt.title(f'Out-of-Sample Predictions (RMSE: {oos_rmse:.2f}, MAPE: {oos_mape:.2f}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('oos_predictions.png')
    else:
        oos_rmse = None
        oos_mape = None

    # Return results
    return {
        'coefficient_stability': coef_df,
        'r_squared_stability': r2_df,
        'stability_metrics': stability_metrics,
        'out_of_sample_metrics': {
            'rmse': oos_rmse,
            'mape': oos_mape
        },
        'plots': {
            'coefficient_stability': 'coefficient_stability.png',
            'oos_predictions': 'oos_predictions.png' if len(pred_df) > 0 else None
        }
    }


def leave_one_out_analysis(df, target, features, model_type='linear-linear'):
    """
    Perform leave-one-out analysis to detect influential observations.

    Args:
        df: DataFrame with the data
        target: Target variable name
        features: List of feature variable names
        model_type: Type of model ('linear-linear', 'log-log', etc.)

    Returns:
        Dictionary with leave-one-out analysis results
    """
    # Transform data if using log-log model
    if model_type == 'log-log':
        df_transformed = df.copy()
        df_transformed[target] = np.log1p(df[target])
        for feature in features:
            df_transformed[feature] = np.log1p(df[feature])
    else:
        df_transformed = df

    # Fit full model for reference
    X_full = sm.add_constant(df_transformed[features])
    y_full = df_transformed[target]
    full_model = sm.OLS(y_full, X_full).fit()

    # Initialize storage for results
    n_obs = len(df)
    coefficient_impacts = {feature: [] for feature in features}
    coefficient_impacts['const'] = []
    prediction_impacts = []

    # Identify potential influential observations
    for i in range(n_obs):
        # Create leave-one-out sample
        loo_df = df_transformed.drop(i)

        # Prepare X and y
        X_loo = sm.add_constant(loo_df[features])
        y_loo = loo_df[target]

        # Fit model
        try:
            loo_model = sm.OLS(y_loo, X_loo).fit()

            # Calculate coefficient % change
            for feature in features + ['const']:
                pct_change = (loo_model.params[feature] - full_model.params[feature]) / full_model.params[feature] * 100
                coefficient_impacts[feature].append(pct_change)

            # Calculate prediction impact
            X_i = sm.add_constant(df_transformed.iloc[i:i + 1][features])
            full_pred = full_model.predict(X_i)[0]
            loo_pred = loo_model.predict(X_i)[0]
            pred_diff = (loo_pred - full_pred) / full_pred * 100

            prediction_impacts.append({
                'index': i,
                'actual': df.iloc[i][target],
                'full_pred': np.expm1(full_pred) if model_type == 'log-log' else full_pred,
                'loo_pred': np.expm1(loo_pred) if model_type == 'log-log' else loo_pred,
                'pct_diff': pred_diff
            })
        except:
            # Skip if model fails to fit
            continue

    # Convert to DataFrame
    coef_impact_df = pd.DataFrame(coefficient_impacts)
    pred_impact_df = pd.DataFrame(prediction_impacts)

    # Calculate impact statistics
    impact_stats = {}
    for feature in features + ['const']:
        impact_stats[feature] = {
            'mean_impact': coef_impact_df[feature].mean(),
            'max_impact': coef_impact_df[feature].abs().max(),
            'std_impact': coef_impact_df[feature].std()
        }

    # Identify influential observations (those with impact > 1 std dev)
    influential_obs = []
    threshold = 1.5  # Standard deviations

    for i in range(len(coef_impact_df)):
        influential_features = []
        for feature in features + ['const']:
            if abs(coef_impact_df.iloc[i][feature]) > threshold * coef_impact_df[feature].std():
                influential_features.append(feature)

        if influential_features:
            influential_obs.append({
                'index': i,
                'features_affected': influential_features,
                'max_impact': max([abs(coef_impact_df.iloc[i][f]) for f in influential_features])
            })

    # Sort influential observations by max impact
    influential_obs = sorted(influential_obs, key=lambda x: x['max_impact'], reverse=True)

    # Create plots
    if len(influential_obs) > 0:
        plt.figure(figsize=(12, 6))

        # Plot the coefficient impact distribution (boxplot)
        plt.subplot(1, 2, 1)
        coef_impact_df.boxplot()
        plt.title('Coefficient Impact Distribution')
        plt.ylabel('% Change in Coefficient')
        plt.grid(True, alpha=0.3)

        # Plot prediction impact for influential observations
        plt.subplot(1, 2, 2)

        # Get indices of top influential observations
        top_influential = [obs['index'] for obs in influential_obs[:min(10, len(influential_obs))]]

        # Filter prediction impacts for these observations
        if len(pred_impact_df) > 0:
            top_pred_impacts = pred_impact_df[pred_impact_df['index'].isin(top_influential)]

            if len(top_pred_impacts) > 0:
                plt.bar(range(len(top_pred_impacts)), top_pred_impacts['pct_diff'])
                plt.xticks(range(len(top_pred_impacts)), top_pred_impacts['index'])
                plt.title('Prediction Impact of Influential Observations')
                plt.xlabel('Observation Index')
                plt.ylabel('% Change in Prediction')
                plt.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('influential_observations.png')

    # Return results
    return {
        'coefficient_impacts': coef_impact_df,
        'prediction_impacts': pred_impact_df,
        'impact_statistics': impact_stats,
        'influential_observations': influential_obs,
        'plots': {
            'influential_observations': 'influential_observations.png' if len(influential_obs) > 0 else None
        }
    }


def jackknife_resampling(df, target, features, n_samples=100, sample_pct=0.8, model_type='linear-linear'):
    """
    Perform jackknife resampling to estimate coefficient variability.

    Args:
        df: DataFrame with the data
        target: Target variable name
        features: List of feature variable names
        n_samples: Number of jackknife samples to create
        sample_pct: Percentage of data to include in each sample
        model_type: Type of model ('linear-linear', 'log-log', etc.)

    Returns:
        Dictionary with jackknife analysis results
    """
    # Transform data if using log-log model
    if model_type == 'log-log':
        df_transformed = df.copy()
        df_transformed[target] = np.log1p(df[target])
        for feature in features:
            df_transformed[feature] = np.log1p(df[feature])
    else:
        df_transformed = df

    # Initialize storage for coefficient distributions
    n_obs = len(df)
    sample_size = int(n_obs * sample_pct)
    coefficient_samples = {feature: [] for feature in features}
    coefficient_samples['const'] = []
    r_squared_samples = []

    # Create jackknife samples
    for _ in range(n_samples):
        # Create random sample
        sample_indices = np.random.choice(n_obs, size=sample_size, replace=False)
        sample_df = df_transformed.iloc[sample_indices]

        # Prepare X and y
        X_sample = sm.add_constant(sample_df[features])
        y_sample = sample_df[target]

        # Fit model
        try:
            sample_model = sm.OLS(y_sample, X_sample).fit()

            # Store coefficients
            for feature in features + ['const']:
                coefficient_samples[feature].append(sample_model.params[feature])

            # Store R-squared
            r_squared_samples.append(sample_model.rsquared)
        except:
            # Skip if model fails to fit
            continue

    # Convert to DataFrame
    coef_df = pd.DataFrame(coefficient_samples)
    r2_series = pd.Series(r_squared_samples)

    # Calculate statistics
    coef_stats = {}
    for feature in features + ['const']:
        coef_stats[feature] = {
            'mean': coef_df[feature].mean(),
            'median': coef_df[feature].median(),
            'std': coef_df[feature].std(),
            'cv': coef_df[feature].std() / abs(coef_df[feature].mean()) if coef_df[feature].mean() != 0 else np.nan,
            '5th_pct': coef_df[feature].quantile(0.05),
            '95th_pct': coef_df[feature].quantile(0.95)
        }

    # Calculate R-squared statistics
    r2_stats = {
        'mean': r2_series.mean(),
        'median': r2_series.median(),
        'std': r2_series.std(),
        '5th_pct': r2_series.quantile(0.05),
        '95th_pct': r2_series.quantile(0.95)
    }

    # Create plots
    plt.figure(figsize=(12, 10))

    # Plot coefficient distributions
    n_features = len(features) + 1  # +1 for const
    n_cols = 2
    n_rows = (n_features + 1) // n_cols + 1  # +1 for R-squared

    for i, feature in enumerate(features + ['const']):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(coef_df[feature], kde=True)
        plt.title(f'{feature} Coefficient Distribution')
        plt.axvline(x=coef_df[feature].mean(), color='r', linestyle='--',
                    label=f'Mean: {coef_df[feature].mean():.4f}')
        plt.axvline(x=coef_df[feature].quantile(0.05), color='g', linestyle=':',
                    label='5th/95th pct')
        plt.axvline(x=coef_df[feature].quantile(0.95), color='g', linestyle=':')
        plt.grid(True, alpha=0.3)
        plt.legend()

    # Plot R-squared distribution
    plt.subplot(n_rows, n_cols, n_features + 1)
    sns.histplot(r2_series, kde=True)
    plt.title('R² Distribution')
    plt.axvline(x=r2_series.mean(), color='r', linestyle='--',
                label=f'Mean: {r2_series.mean():.4f}')
    plt.axvline(x=r2_series.quantile(0.05), color='g', linestyle=':',
                label='5th/95th pct')
    plt.axvline(x=r2_series.quantile(0.95), color='g', linestyle=':')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('jackknife_distributions.png')

    # Return results
    return {
        'coefficient_distributions': coef_df,
        'r_squared_distribution': r2_series,
        'coefficient_statistics': coef_stats,
        'r_squared_statistics': r2_stats,
        'plots': {
            'jackknife_distributions': 'jackknife_distributions.png'
        }
    }


# ---------------------------------------------------------------------------------------
# SENSITIVITY TESTING FUNCTIONS
# ---------------------------------------------------------------------------------------

def monte_carlo_simulation(df, target, features, n_simulations=100, noise_level=0.1, model_type='linear-linear'):
    """
    Perform Monte Carlo simulations with parameter perturbation.

    Args:
        df: DataFrame with the data
        target: Target variable name
        features: List of feature variable names
        n_simulations: Number of simulations to run
        noise_level: Level of noise to add to features (as proportion of std dev)
        model_type: Type of model ('linear-linear', 'log-log', etc.)

    Returns:
        Dictionary with Monte Carlo simulation results
    """
    # Transform data if using log-log model
    if model_type == 'log-log':
        df_transformed = df.copy()
        df_transformed[target] = np.log1p(df[target])
        for feature in features:
            df_transformed[feature] = np.log1p(df[feature])
    else:
        df_transformed = df

    # Fit base model for reference
    X_base = sm.add_constant(df_transformed[features])
    y_base = df_transformed[target]
    base_model = sm.OLS(y_base, X_base).fit()

    # Calculate feature standard deviations for noise addition
    feature_stds = {feature: df_transformed[feature].std() for feature in features}

    # Initialize storage for results
    coefficient_results = {feature: [] for feature in features}
    coefficient_results['const'] = []
    r_squared_results = []
    elasticity_results = {feature: [] for feature in features}

    # Run simulations
    for i in range(n_simulations):
        # Create perturbed data
        perturbed_df = df_transformed.copy()

        # Add noise to each feature
        for feature in features:
            noise = np.random.normal(0, feature_stds[feature] * noise_level, len(df))
            perturbed_df[feature] = df_transformed[feature] + noise

        # Prepare X and y
        X_sim = sm.add_constant(perturbed_df[features])
        y_sim = df_transformed[target]  # Target doesn't change

        # Fit model
        try:
            sim_model = sm.OLS(y_sim, X_sim).fit()

            # Store coefficients
            for feature in features + ['const']:
                coefficient_results[feature].append(sim_model.params[feature])

            # Store R-squared
            r_squared_results.append(sim_model.rsquared)

            # Calculate and store elasticities
            for feature in features:
                if model_type == 'log-log':
                    # For log-log model, elasticity is the coefficient
                    elasticity = sim_model.params[feature]
                else:
                    # For linear model, elasticity = coef * (mean_x / mean_y)
                    mean_x = perturbed_df[feature].mean()
                    mean_y = y_sim.mean()
                    elasticity = sim_model.params[feature] * (mean_x / mean_y)

                elasticity_results[feature].append(elasticity)
        except Exception as e:
            logger.warning(f"Error in simulation {i}: {str(e)}")
            # Skip this simulation
            continue

    # Convert to DataFrames
    coef_df = pd.DataFrame(coefficient_results)
    r2_series = pd.Series(r_squared_results)
    elas_df = pd.DataFrame(elasticity_results)

    # Calculate statistics
    coef_stats = {}
    for feature in features + ['const']:
        coef_stats[feature] = {
            'mean': coef_df[feature].mean(),
            'std': coef_df[feature].std(),
            'cv': coef_df[feature].std() / abs(coef_df[feature].mean()) if coef_df[feature].mean() != 0 else np.nan,
            'base_value': base_model.params[feature],
            'pct_diff_from_base': (coef_df[feature].mean() - base_model.params[feature]) / base_model.params[
                feature] * 100 if base_model.params[feature] != 0 else np.nan
        }

    # Calculate elasticity statistics
    elasticity_stats = {}
    for feature in features:
        # Calculate base elasticity
        if model_type == 'log-log':
            base_elasticity = base_model.params[feature]
        else:
            mean_x = df_transformed[feature].mean()
            mean_y = y_base.mean()
            base_elasticity = base_model.params[feature] * (mean_x / mean_y)

        elasticity_stats[feature] = {
            'mean': elas_df[feature].mean(),
            'std': elas_df[feature].std(),
            'cv': elas_df[feature].std() / abs(elas_df[feature].mean()) if elas_df[feature].mean() != 0 else np.nan,
            'base_value': base_elasticity,
            'pct_diff_from_base': (elas_df[
                                       feature].mean() - base_elasticity) / base_elasticity * 100 if base_elasticity != 0 else np.nan
        }

    # Create plots
    plt.figure(figsize=(12, 10))

    # Plot coefficient distributions
    n_coefs = len(features) + 1  # +1 for const
    n_rows = (n_coefs + len(features)) // 2

    for i, feature in enumerate(features + ['const']):
        plt.subplot(n_rows, 2, i + 1)
        sns.histplot(coef_df[feature], kde=True)
        plt.axvline(x=base_model.params[feature], color='r', linestyle='--',
                    label=f'Base: {base_model.params[feature]:.4f}')
        plt.title(f'{feature} Coefficient Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()

    # Plot elasticity distributions
    for i, feature in enumerate(features):
        plt.subplot(n_rows, 2, i + n_coefs + 1)
        sns.histplot(elas_df[feature], kde=True)

        # Calculate base elasticity again
        if model_type == 'log-log':
            base_elasticity = base_model.params[feature]
        else:
            mean_x = df_transformed[feature].mean()
            mean_y = y_base.mean()
            base_elasticity = base_model.params[feature] * (mean_x / mean_y)

        plt.axvline(x=base_elasticity, color='r', linestyle='--',
                    label=f'Base: {base_elasticity:.4f}')
        plt.title(f'{feature} Elasticity Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig('monte_carlo_distributions.png')

    # Return results
    return {
        'coefficient_distributions': coef_df,
        'r_squared_distribution': r2_series,
        'elasticity_distributions': elas_df,
        'coefficient_statistics': coef_stats,
        'elasticity_statistics': elasticity_stats,
        'plots': {
            'monte_carlo_distributions': 'monte_carlo_distributions.png'
        }
    }


def model_specification_comparison(df, target, features, compare_specs=None):
    """
    Compare different model specifications to assess elasticity robustness.

    Args:
        df: DataFrame with the data
        target: Target variable name
        features: List of feature variable names
        compare_specs: List of specifications to compare (if None, compare default set)

    Returns:
        Dictionary with model specification comparison results
    """
    # Define default specifications to compare if not provided
    if compare_specs is None:
        compare_specs = [
            {'name': 'Linear-Linear', 'target_transform': None, 'feature_transform': None},
            {'name': 'Log-Log', 'target_transform': 'log', 'feature_transform': 'log'},
            {'name': 'Semi-Log', 'target_transform': None, 'feature_transform': 'log'},
            {'name': 'Log-Linear', 'target_transform': 'log', 'feature_transform': None}
        ]

    # Initialize storage for results
    models = {}
    metrics = {}
    elasticities = {}

    # Test each specification
    for spec in compare_specs:
        spec_name = spec['name']
        target_transform = spec['target_transform']
        feature_transform = spec['feature_transform']

        # Create transformed data
        transformed_df = df.copy()

        # Transform target if needed
        if target_transform == 'log':
            transformed_df[f"{target}_log"] = np.log1p(df[target])
            target_col = f"{target}_log"
        else:
            target_col = target

        # Transform features if needed
        if feature_transform == 'log':
            transformed_features = []
            for feature in features:
                transformed_df[f"{feature}_log"] = np.log1p(df[feature])
                transformed_features.append(f"{feature}_log")
        else:
            transformed_features = features

        # Prepare X and y
        X = sm.add_constant(transformed_df[transformed_features])
        y = transformed_df[target_col]

        # Fit model
        try:
            model = sm.OLS(y, X).fit()
            models[spec_name] = model

            # Calculate metrics
            metrics[spec_name] = {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'aic': model.aic,
                'bic': model.bic
            }

            # Make predictions on original scale for comparison
            preds = model.predict(X)
            if target_transform == 'log':
                preds = np.expm1(preds)
                actuals = df[target]
            else:
                actuals = df[target]

            # Calculate prediction metrics
            metrics[spec_name]['rmse'] = np.sqrt(mean_squared_error(actuals, preds))
            metrics[spec_name]['mape'] = mean_absolute_percentage_error(actuals, preds) * 100

            # Calculate elasticities
            model_elasticities = {}
            for i, feature in enumerate(features):
                if feature_transform == 'log' and target_transform == 'log':
                    # Log-Log: elasticity is coefficient
                    elasticity = model.params[transformed_features[i]]
                elif feature_transform == 'log' and target_transform is None:
                    # Semi-Log: elasticity = coef * (1 / mean(y))
                    elasticity = model.params[transformed_features[i]] * (1 / df[target].mean())
                elif feature_transform is None and target_transform == 'log':
                    # Log-Linear: elasticity = coef * mean(x)
                    elasticity = model.params[feature] * df[feature].mean()
                else:
                    # Linear-Linear: elasticity = coef * (mean(x) / mean(y))
                    elasticity = model.params[feature] * (df[feature].mean() / df[target].mean())

                model_elasticities[feature] = elasticity

            elasticities[spec_name] = model_elasticities
        except Exception as e:
            logger.warning(f"Error fitting {spec_name} model: {str(e)}")
            metrics[spec_name] = {
                'error': str(e)
            }

    # Convert metrics to DataFrame for easier comparison
    metrics_df = pd.DataFrame(metrics).T

    # Create elasticity comparison DataFrame
    elasticity_comparison = {}
    for feature in features:
        elasticity_comparison[feature] = {spec_name: elasticities[spec_name][feature]
                                          for spec_name in elasticities.keys()
                                          if feature in elasticities[spec_name]}

    elasticity_df = pd.DataFrame(elasticity_comparison)

    # Calculate coefficient of variation across models
    elasticity_stats = {}
    for feature in features:
        feature_elasticities = elasticity_df[feature].dropna()
        if len(feature_elasticities) > 0:
            elasticity_stats[feature] = {
                'mean': feature_elasticities.mean(),
                'std': feature_elasticities.std(),
                'cv': feature_elasticities.std() / abs(
                    feature_elasticities.mean()) if feature_elasticities.mean() != 0 else np.nan,
                'min': feature_elasticities.min(),
                'max': feature_elasticities.max(),
                'range': feature_elasticities.max() - feature_elasticities.min()
            }

    # Create plots
    plt.figure(figsize=(12, 10))

    # Plot elasticity comparison
    plt.subplot(2, 1, 1)
    elasticity_df.plot(kind='bar', ax=plt.gca())
    plt.title('Elasticity Comparison Across Model Specifications')
    plt.ylabel('Elasticity')
    plt.grid(True, axis='y', alpha=0.3)

    # Plot model fitness metrics
    plt.subplot(2, 1, 2)
    if 'rmse' in metrics_df.columns and 'mape' in metrics_df.columns:
        metrics_plot = pd.DataFrame({
            'RMSE': metrics_df['rmse'],
            'MAPE (%)': metrics_df['mape']
        })
        metrics_plot.plot(kind='bar', ax=plt.gca())
        plt.title('Model Fit Metrics')
        plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_specification_comparison.png')

    # Return results
    return {
        'models': models,
        'metrics': metrics_df,
        'elasticities': elasticity_df,
        'elasticity_statistics': elasticity_stats,
        'plots': {
            'model_specification_comparison': 'model_specification_comparison.png'
        }
    }


def outlier_impact_analysis(df, target, features, outlier_threshold=2, model_type='linear-linear'):
    """
    Analyze the impact of outliers on model results.

    Args:
        df: DataFrame with the data
        target: Target variable name
        features: List of feature variable names
        outlier_threshold: Z-score threshold for outlier detection
        model_type: Type of model ('linear-linear', 'log-log', etc.)

    Returns:
        Dictionary with outlier impact analysis results
    """
    # Transform data if using log-log model
    if model_type == 'log-log':
        df_transformed = df.copy()
        df_transformed[target] = np.log1p(df[target])
        for feature in features:
            df_transformed[feature] = np.log1p(df[feature])
    else:
        df_transformed = df.copy()

    # Detect outliers in each feature
    outliers = {}
    all_outlier_indices = set()

    for feature in features + [target]:
        # Calculate z-scores
        z_scores = np.abs((df_transformed[feature] - df_transformed[feature].mean()) / df_transformed[feature].std())

        # Find outliers
        feature_outliers = df_transformed.index[z_scores > outlier_threshold].tolist()
        outliers[feature] = feature_outliers
        all_outlier_indices.update(feature_outliers)

    # Convert to list and sort
    all_outlier_indices = sorted(list(all_outlier_indices))

    # Create a DataFrame without outliers
    df_no_outliers = df_transformed.drop(all_outlier_indices)

    # Fit models with and without outliers
    X_full = sm.add_constant(df_transformed[features])
    y_full = df_transformed[target]

    X_no_outliers = sm.add_constant(df_no_outliers[features])
    y_no_outliers = df_no_outliers[target]

    # Fit full model (with outliers)
    full_model = sm.OLS(y_full, X_full).fit()

    # Fit model without outliers
    no_outliers_model = sm.OLS(y_no_outliers, X_no_outliers).fit()

    # Calculate elasticities for both models
    full_elasticities = {}
    no_outliers_elasticities = {}

    for feature in features:
        if model_type == 'log-log':
            # For log-log model, elasticity is the coefficient
            full_elasticities[feature] = full_model.params[feature]
            no_outliers_elasticities[feature] = no_outliers_model.params[feature]
        else:
            # For linear model, elasticity = coef * (mean_x / mean_y)
            full_mean_x = df_transformed[feature].mean()
            full_mean_y = y_full.mean()
            full_elasticities[feature] = full_model.params[feature] * (full_mean_x / full_mean_y)

            no_outliers_mean_x = df_no_outliers[feature].mean()
            no_outliers_mean_y = y_no_outliers.mean()
            no_outliers_elasticities[feature] = no_outliers_model.params[feature] * (
                        no_outliers_mean_x / no_outliers_mean_y)

    # Calculate percentage differences
    elasticity_diffs = {}
    for feature in features:
        if full_elasticities[feature] != 0:
            pct_diff = (no_outliers_elasticities[feature] - full_elasticities[feature]) / full_elasticities[
                feature] * 100
        else:
            pct_diff = np.nan

        elasticity_diffs[feature] = pct_diff

    # Calculate prediction metrics for both models
    full_preds = full_model.predict(X_full)
    no_outliers_preds_full = no_outliers_model.predict(X_full)  # Predict on full dataset

    # Back-transform if needed
    if model_type == 'log-log':
        full_preds_original = np.expm1(full_preds)
        no_outliers_preds_full_original = np.expm1(no_outliers_preds_full)
        y_original = df[target]
    else:
        full_preds_original = full_preds
        no_outliers_preds_full_original = no_outliers_preds_full
        y_original = df[target]

    # Calculate metrics
    full_metrics = {
        'r_squared': full_model.rsquared,
        'adj_r_squared': full_model.rsquared_adj,
        'rmse': np.sqrt(mean_squared_error(y_original, full_preds_original)),
        'mape': mean_absolute_percentage_error(y_original, full_preds_original) * 100
    }

    no_outliers_metrics = {
        'r_squared': no_outliers_model.rsquared,
        'adj_r_squared': no_outliers_model.rsquared_adj,
        'rmse': np.sqrt(mean_squared_error(y_original, no_outliers_preds_full_original)),
        'mape': mean_absolute_percentage_error(y_original, no_outliers_preds_full_original) * 100
    }

    # Create visualizations
    plt.figure(figsize=(12, 10))

    # Plot elasticity comparison
    plt.subplot(2, 2, 1)
    elasticity_comparison = pd.DataFrame({
        'With Outliers': full_elasticities,
        'Without Outliers': no_outliers_elasticities
    }).T
    elasticity_comparison.plot(kind='bar', ax=plt.gca())
    plt.title('Elasticity Comparison')
    plt.grid(True, axis='y', alpha=0.3)

    # Plot percentage differences
    plt.subplot(2, 2, 2)
    plt.bar(features, [elasticity_diffs[f] for f in features])
    plt.title('Elasticity % Change After Removing Outliers')
    plt.grid(True, axis='y', alpha=0.3)

    # Plot actual vs predicted
    plt.subplot(2, 2, 3)
    plt.scatter(y_original, full_preds_original, alpha=0.5, label='With Outliers')
    plt.scatter(y_original, no_outliers_preds_full_original, alpha=0.5, label='Without Outliers')
    plt.plot([min(y_original), max(y_original)], [min(y_original), max(y_original)], 'k--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot metrics comparison
    plt.subplot(2, 2, 4)
    metrics_comparison = pd.DataFrame({
        'With Outliers': {'RMSE': full_metrics['rmse'], 'MAPE (%)': full_metrics['mape']},
        'Without Outliers': {'RMSE': no_outliers_metrics['rmse'], 'MAPE (%)': no_outliers_metrics['mape']}
    }).T
    metrics_comparison.plot(kind='bar', ax=plt.gca())
    plt.title('Prediction Metrics')
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('outlier_impact_analysis.png')

    # Return results
    return {
        'outliers': {
            'count': len(all_outlier_indices),
            'indices': all_outlier_indices,
            'by_feature': outliers
        },
        'models': {
            'full': full_model,
            'no_outliers': no_outliers_model
        },
        'elasticities': {
            'full': full_elasticities,
            'no_outliers': no_outliers_elasticities,
            'pct_diff': elasticity_diffs
        },
        'metrics': {
            'full': full_metrics,
            'no_outliers': no_outliers_metrics
        },
        'plots': {
            'outlier_impact_analysis': 'outlier_impact_analysis.png'
        }
    }


# ---------------------------------------------------------------------------------------
# MODEL VALIDATION EXTENSIONS
# ---------------------------------------------------------------------------------------

def time_based_validation(df, target, features, time_col=None, n_splits=5, model_type='linear-linear'):
    """
    Perform time-based cross-validation.

    Args:
        df: DataFrame with the data
        target: Target variable name
        features: List of feature variable names
        time_col: Column to use for time ordering (if None, use DataFrame index)
        n_splits: Number of time splits to create
        model_type: Type of model ('linear-linear', 'log-log', etc.)

    Returns:
        Dictionary with time-based validation results
    """
    # Sort data by time if time column provided
    if time_col is not None and time_col in df.columns:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
    else:
        df_sorted = df.copy()

    # Transform data if using log-log model
    if model_type == 'log-log':
        df_transformed = df_sorted.copy()
        df_transformed[target] = np.log1p(df_sorted[target])
        for feature in features:
            df_transformed[feature] = np.log1p(df_sorted[feature])
    else:
        df_transformed = df_sorted

    # Create time series splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize storage for results
    fold_metrics = []
    fold_elasticities = []

    # Training size for each fold
    fold_sizes = []

    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(tscv.split(df_transformed)):
        train_data = df_transformed.iloc[train_idx]
        test_data = df_transformed.iloc[test_idx]

        # Record fold sizes
        fold_sizes.append(len(train_data))

        # Prepare X and y for training
        X_train = sm.add_constant(train_data[features])
        y_train = train_data[target]

        # Prepare X and y for testing
        X_test = sm.add_constant(test_data[features])
        y_test = test_data[target]

        # Fit model
        try:
            model = sm.OLS(y_train, X_train).fit()

            # Make predictions
            test_preds = model.predict(X_test)

            # Calculate metrics (in original scale if log-transformed)
            if model_type == 'log-log':
                test_preds_original = np.expm1(test_preds)
                y_test_original = np.expm1(y_test)
            else:
                test_preds_original = test_preds
                y_test_original = y_test

            r2 = r2_score(y_test_original, test_preds_original)
            rmse = np.sqrt(mean_squared_error(y_test_original, test_preds_original))
            mape = mean_absolute_percentage_error(y_test_original, test_preds_original) * 100

            # Store metrics
            fold_metrics.append({
                'fold': i,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'r_squared': r2,
                'rmse': rmse,
                'mape': mape
            })

            # Calculate elasticities
            fold_elas = {}
            for feature in features:
                if model_type == 'log-log':
                    # For log-log model, elasticity is the coefficient
                    elasticity = model.params[feature]
                else:
                    # For linear model, elasticity = coef * (mean_x / mean_y)
                    mean_x = train_data[feature].mean()
                    mean_y = y_train.mean()
                    elasticity = model.params[feature] * (mean_x / mean_y)

                fold_elas[feature] = elasticity

            fold_elasticities.append({
                'fold': i,
                **fold_elas
            })
        except Exception as e:
            logger.warning(f"Error in fold {i}: {str(e)}")
            # Skip this fold
            continue

    # Convert to DataFrames
    metrics_df = pd.DataFrame(fold_metrics)
    elasticities_df = pd.DataFrame(fold_elasticities)

    # Calculate summary statistics
    metrics_summary = {
        'r_squared': {
            'mean': metrics_df['r_squared'].mean(),
            'std': metrics_df['r_squared'].std(),
            'min': metrics_df['r_squared'].min(),
            'max': metrics_df['r_squared'].max()
        },
        'rmse': {
            'mean': metrics_df['rmse'].mean(),
            'std': metrics_df['rmse'].std(),
            'min': metrics_df['rmse'].min(),
            'max': metrics_df['rmse'].max()
        },
        'mape': {
            'mean': metrics_df['mape'].mean(),
            'std': metrics_df['mape'].std(),
            'min': metrics_df['mape'].min(),
            'max': metrics_df['mape'].max()
        }
    }

    # Calculate elasticity stability
    elasticity_stability = {}
    for feature in features:
        if feature in elasticities_df.columns:
            elasticity_stability[feature] = {
                'mean': elasticities_df[feature].mean(),
                'std': elasticities_df[feature].std(),
                'cv': elasticities_df[feature].std() / abs(elasticities_df[feature].mean()) if elasticities_df[
                                                                                                   feature].mean() != 0 else np.nan,
                'min': elasticities_df[feature].min(),
                'max': elasticities_df[feature].max()
            }

    # Create plots
    plt.figure(figsize=(12, 10))

    # Plot metrics across folds
    plt.subplot(2, 1, 1)
    plt.plot(metrics_df['fold'], metrics_df['r_squared'], 'b-o', label='R²')
    plt.plot(metrics_df['fold'], metrics_df['mape'] / 100, 'r-o', label='MAPE (%)')
    plt.xticks(metrics_df['fold'])
    plt.title('Model Performance Across Validation Folds')
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot elasticities across folds
    plt.subplot(2, 1, 2)
    for feature in features:
        if feature in elasticities_df.columns:
            plt.plot(elasticities_df['fold'], elasticities_df[feature], 'o-', label=feature)

    plt.xticks(elasticities_df['fold'])
    plt.title('Elasticity Stability Across Validation Folds')
    plt.xlabel('Fold')
    plt.ylabel('Elasticity')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('time_based_validation.png')

    # Return results
    return {
        'metrics': metrics_df,
        'elasticities': elasticities_df,
        'metrics_summary': metrics_summary,
        'elasticity_stability': elasticity_stability,
        'plots': {
            'time_based_validation': 'time_based_validation.png'
        }
    }


def prediction_confidence_intervals(model, X, y, alpha=0.05, model_type='linear-linear'):
    """
    Calculate prediction confidence intervals.

    Args:
        model: Fitted statsmodels OLS model
        X: Feature matrix
        y: Target variable
        alpha: Significance level for confidence intervals
        model_type: Type of model ('linear-linear', 'log-log', etc.)

    Returns:
        Dictionary with prediction interval results
    """
    # Make predictions
    predictions = model.predict(X)

    # Get prediction intervals
    pred_intervals = model.get_prediction(X).conf_int(alpha=alpha)
    lower_bound = pred_intervals[:, 0]
    upper_bound = pred_intervals[:, 1]

    # Calculate interval widths
    interval_widths = upper_bound - lower_bound

    # Check if target was log-transformed
    if model_type == 'log-log' or model_type == 'log-linear':
        # Back-transform predictions and intervals
        predictions_original = np.expm1(predictions)
        lower_bound_original = np.expm1(lower_bound)
        upper_bound_original = np.expm1(upper_bound)
        y_original = np.expm1(y) if isinstance(y, pd.Series) or isinstance(y, np.ndarray) else y
    else:
        predictions_original = predictions
        lower_bound_original = lower_bound
        upper_bound_original = upper_bound
        y_original = y

    # Calculate percentage of actual values within intervals
    within_interval = ((y_original >= lower_bound_original) & (y_original <= upper_bound_original)).mean() * 100

    # Calculate average interval width as percentage of predicted value
    avg_pct_width = (interval_widths / predictions).mean() * 100

    # Create visualization
    plt.figure(figsize=(12, 6))

    # Sort by prediction for clearer visualization
    sort_idx = np.argsort(predictions)
    sorted_preds = predictions[sort_idx]
    sorted_lower = lower_bound[sort_idx]
    sorted_upper = upper_bound[sort_idx]
    sorted_y = y.iloc[sort_idx] if isinstance(y, pd.Series) else y[sort_idx]

    # Plot predictions with confidence intervals
    plt.plot(range(len(sorted_preds)), sorted_preds, 'b-', label='Predicted')
    plt.fill_between(range(len(sorted_preds)), sorted_lower, sorted_upper,
                     color='blue', alpha=0.2, label=f'{(1 - alpha) * 100}% CI')
    plt.scatter(range(len(sorted_y)), sorted_y,
                color='r', alpha=0.5, label='Actual')

    plt.title(f'Predictions with {(1 - alpha) * 100}% Confidence Intervals')
    plt.xlabel('Observation Index (sorted by prediction)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('prediction_intervals.png')

    # Return results
    return {
        'predictions': predictions,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'interval_widths': interval_widths,
        'pct_within_interval': within_interval,
        'avg_interval_width_pct': avg_pct_width,
        'plots': {
            'prediction_intervals': 'prediction_intervals.png'
        }
    }


# ---------------------------------------------------------------------------------------
# COMPREHENSIVE DIAGNOSTIC SUITE
# ---------------------------------------------------------------------------------------

def run_comprehensive_diagnostics(df, target, features, model_type='linear-linear',
                                  time_col=None, validation_splits=5):
    """
    Run a comprehensive suite of model diagnostics.

    Args:
        df: DataFrame with the data
        target: Target variable name
        features: List of feature variable names
        model_type: Type of model ('linear-linear', 'log-log', etc.)
        time_col: Column to use for time ordering
        validation_splits: Number of splits for time-based validation

    Returns:
        Dictionary with all diagnostic results
    """
    logger.info("Starting comprehensive diagnostics...")

    # Transform data if needed
    if model_type == 'log-log':
        df_transformed = df.copy()
        df_transformed[f"{target}_log"] = np.log1p(df[target])
        target_col = f"{target}_log"

        transformed_features = []
        for feature in features:
            df_transformed[f"{feature}_log"] = np.log1p(df[feature])
            transformed_features.append(f"{feature}_log")
    else:
        df_transformed = df
        target_col = target
        transformed_features = features

    # Fit base model
    X = sm.add_constant(df_transformed[transformed_features])
    y = df_transformed[target_col]

    logger.info("Fitting base model...")
    model = sm.OLS(y, X).fit()

    results = {
        'model': model,
        'model_type': model_type,
        'feature_names': transformed_features,
        'target': target_col
    }

    # 1. Residual Analysis
    logger.info("Performing residual analysis...")
    results['residual_analysis'] = analyze_residuals(model, X, y, plot=True)

    # 2. Coefficient Stability Plot
    logger.info("Creating coefficient stability plot...")
    coef_stability_fig = plot_coefficient_stability(model, X, y, transformed_features)
    results['coefficient_stability_fig'] = coef_stability_fig
    plt.savefig('coefficient_stability.png')
    results['plots'] = {'coefficient_stability': 'coefficient_stability.png'}

    # 3. Stability Assessment: Rolling Window Analysis
    logger.info("Performing rolling window analysis...")
    try:
        results['rolling_window_analysis'] = rolling_window_analysis(
            df_transformed, target_col, transformed_features, model_type=model_type)
        results['plots'].update(results['rolling_window_analysis']['plots'])
    except Exception as e:
        logger.warning(f"Error in rolling window analysis: {str(e)}")
        results['rolling_window_analysis'] = {'error': str(e)}

    # 4. Stability Assessment: Leave-One-Out Analysis
    logger.info("Performing leave-one-out analysis...")
    try:
        results['leave_one_out_analysis'] = leave_one_out_analysis(
            df_transformed, target_col, transformed_features, model_type=model_type)
        results['plots'].update(results['leave_one_out_analysis'].get('plots', {}))
    except Exception as e:
        logger.warning(f"Error in leave-one-out analysis: {str(e)}")
        results['leave_one_out_analysis'] = {'error': str(e)}

    # 5. Stability Assessment: Jackknife Resampling
    logger.info("Performing jackknife resampling...")
    try:
        results['jackknife_analysis'] = jackknife_resampling(
            df_transformed, target_col, transformed_features, n_samples=100, model_type=model_type)
        results['plots'].update(results['jackknife_analysis']['plots'])
    except Exception as e:
        logger.warning(f"Error in jackknife analysis: {str(e)}")
        results['jackknife_analysis'] = {'error': str(e)}

    # 6. Sensitivity Testing: Monte Carlo Simulation
    logger.info("Performing Monte Carlo simulation...")
    try:
        results['monte_carlo_simulation'] = monte_carlo_simulation(
            df_transformed, target_col, transformed_features, n_simulations=100, model_type=model_type)
        results['plots'].update(results['monte_carlo_simulation']['plots'])
    except Exception as e:
        logger.warning(f"Error in Monte Carlo simulation: {str(e)}")
        results['monte_carlo_simulation'] = {'error': str(e)}

    # 7. Sensitivity Testing: Model Specification Comparison
    logger.info("Comparing model specifications...")
    try:
        results['specification_comparison'] = model_specification_comparison(df, target, features)
        results['plots'].update(results['specification_comparison']['plots'])
    except Exception as e:
        logger.warning(f"Error in model specification comparison: {str(e)}")
        results['specification_comparison'] = {'error': str(e)}

    # 8. Sensitivity Testing: Outlier Impact Analysis
    logger.info("Analyzing outlier impact...")
    try:
        results['outlier_impact'] = outlier_impact_analysis(
            df, target, features, model_type=model_type)
        results['plots'].update(results['outlier_impact']['plots'])
    except Exception as e:
        logger.warning(f"Error in outlier impact analysis: {str(e)}")
        results['outlier_impact'] = {'error': str(e)}

    # 9. Model Validation: Time-Based Validation
    logger.info("Performing time-based validation...")
    try:
        results['time_validation'] = time_based_validation(
            df, target, features, time_col=time_col, n_splits=validation_splits, model_type=model_type)
        results['plots'].update(results['time_validation']['plots'])
    except Exception as e:
        logger.warning(f"Error in time-based validation: {str(e)}")
        results['time_validation'] = {'error': str(e)}

    # 10. Model Validation: Prediction Confidence Intervals
    logger.info("Calculating prediction confidence intervals...")
    try:
        results['prediction_intervals'] = prediction_confidence_intervals(
            model, X, y, model_type=model_type)
        results['plots'].update(results['prediction_intervals']['plots'])
    except Exception as e:
        logger.warning(f"Error in prediction confidence intervals: {str(e)}")
        results['prediction_intervals'] = {'error': str(e)}

    # Generate diagnostic summary
    logger.info("Generating diagnostic summary...")
    results['diagnostic_summary'] = generate_diagnostic_summary(results)

    logger.info("Comprehensive diagnostics completed")
    return results


def generate_diagnostic_summary(diagnostic_results):
    """
    Generate a summary of diagnostic findings.

    Args:
        diagnostic_results: Results dictionary from comprehensive diagnostics

    Returns:
        Dictionary with diagnostic summary
    """
    summary = {
        'model_quality': {
            'issues': [],
            'recommendations': []
        },
        'stability': {
            'issues': [],
            'recommendations': []
        },
        'elasticity_reliability': {
            'issues': [],
            'recommendations': []
        }
    }

    # Check residuals
    if 'residual_analysis' in diagnostic_results:
        res_analysis = diagnostic_results['residual_analysis']

        # Check for residual issues
        if 'summary' in res_analysis and 'residual_issues' in res_analysis['summary'] and res_analysis['summary'][
            'residual_issues']:
            summary['model_quality']['issues'].extend(res_analysis['summary']['residual_issues'])

            # Add recommendations based on specific issues
            for issue in res_analysis['summary']['residual_issues']:
                if 'Non-normal residuals' in issue:
                    summary['model_quality']['recommendations'].append(
                        "Try transforming the target variable (e.g., log, sqrt) to address non-normal residuals")

                if 'Autocorrelation detected' in issue:
                    summary['model_quality']['recommendations'].append(
                        "Consider adding lagged variables or using a model that accounts for autocorrelation")

                if 'Heteroskedasticity detected' in issue:
                    summary['model_quality']['recommendations'].append(
                        "Use robust standard errors or transform variables to address heteroskedasticity")

    # Check coefficient stability
    unstable_coefficients = []

    # From jackknife analysis
    if 'jackknife_analysis' in diagnostic_results and 'coefficient_statistics' in diagnostic_results[
        'jackknife_analysis']:
        jack_stats = diagnostic_results['jackknife_analysis']['coefficient_statistics']
        for feature, stats in jack_stats.items():
            if 'cv' in stats and stats['cv'] is not None and stats['cv'] > 0.5:  # >50% CV indicates instability
                unstable_coefficients.append((feature, stats['cv']))

    # From Monte Carlo
    if 'monte_carlo_simulation' in diagnostic_results and 'coefficient_statistics' in diagnostic_results[
        'monte_carlo_simulation']:
        mc_stats = diagnostic_results['monte_carlo_simulation']['coefficient_statistics']
        for feature, stats in mc_stats.items():
            if 'cv' in stats and stats['cv'] is not None and stats['cv'] > 0.5:
                if not any(feat == feature for feat, _ in unstable_coefficients):
                    unstable_coefficients.append((feature, stats['cv']))

    # From rolling window analysis
    if 'rolling_window_analysis' in diagnostic_results and 'stability_metrics' in diagnostic_results[
        'rolling_window_analysis']:
        rw_stats = diagnostic_results['rolling_window_analysis']['stability_metrics']
        for feature, stats in rw_stats.items():
            if 'cv' in stats and stats['cv'] is not None and stats['cv'] > 0.5:
                if not any(feat == feature for feat, _ in unstable_coefficients):
                    unstable_coefficients.append((feature, stats['cv']))

    # Add stability issues if found
    if unstable_coefficients:
        unstable_coefficients.sort(key=lambda x: x[1], reverse=True)
        for feature, cv in unstable_coefficients[:3]:  # Top 3 most unstable
            summary['stability']['issues'].append(
                f"Unstable coefficient for {feature} (CV: {cv:.2f})")

        summary['stability']['recommendations'].append(
            "Consider model regularization (Ridge, Lasso) to stabilize coefficients")
        summary['stability']['recommendations'].append(
            "Evaluate feature collinearity and consider removing or combining highly correlated features")

    # Check elasticity consistency across specifications
    if 'specification_comparison' in diagnostic_results and 'elasticity_statistics' in diagnostic_results[
        'specification_comparison']:
        spec_stats = diagnostic_results['specification_comparison']['elasticity_statistics']
        inconsistent_elasticities = []

        for feature, stats in spec_stats.items():
            if 'cv' in stats and stats['cv'] is not None and stats['cv'] > 0.3:  # >30% CV indicates inconsistency
                inconsistent_elasticities.append((feature, stats['cv']))

        if inconsistent_elasticities:
            inconsistent_elasticities.sort(key=lambda x: x[1], reverse=True)
            for feature, cv in inconsistent_elasticities[:3]:  # Top 3 most inconsistent
                summary['elasticity_reliability']['issues'].append(
                    f"Inconsistent elasticity for {feature} across model specifications (CV: {cv:.2f})")

            summary['elasticity_reliability']['recommendations'].append(
                "Use log-log specification for more stable elasticity estimation")
            summary['elasticity_reliability']['recommendations'].append(
                "Report elasticity ranges instead of point estimates")

    # Check time-based validation
    if 'time_validation' in diagnostic_results and 'metrics_summary' in diagnostic_results['time_validation']:
        tv_metrics = diagnostic_results['time_validation']['metrics_summary']

        if 'r_squared' in tv_metrics and 'std' in tv_metrics['r_squared'] and tv_metrics['r_squared']['std'] > 0.1:
            summary['model_quality']['issues'].append(
                f"Inconsistent model performance across time periods (R² std: {tv_metrics['r_squared']['std']:.2f})")

            summary['model_quality']['recommendations'].append(
                "Consider adding time-based features to capture temporal patterns")
            summary['model_quality']['recommendations'].append(
                "Evaluate model on the most recent time period for forecasting")

    # Check outlier impact
    if 'outlier_impact' in diagnostic_results and 'elasticities' in diagnostic_results['outlier_impact']:
        oi_elasticities = diagnostic_results['outlier_impact']['elasticities']

        if 'pct_diff' in oi_elasticities:
            significant_impacts = []
            for feature, pct_diff in oi_elasticities['pct_diff'].items():
                if abs(pct_diff) > 20:  # >20% change indicates significant impact
                    significant_impacts.append((feature, pct_diff))

            if significant_impacts:
                significant_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                for feature, pct_diff in significant_impacts[:3]:  # Top 3 most impacted
                    summary['elasticity_reliability']['issues'].append(
                        f"Elasticity for {feature} highly sensitive to outliers ({pct_diff:.1f}% change when removed)")

                summary['elasticity_reliability']['recommendations'].append(
                    "Consider robust regression or outlier treatment techniques")
                summary['elasticity_reliability']['recommendations'].append(
                    "Investigate outliers to determine if they represent valid business patterns")

    # Add general recommendations if we have issues
    if summary['model_quality']['issues'] or summary['stability']['issues'] or summary['elasticity_reliability'][
        'issues']:
        if diagnostic_results.get('model_type') == 'linear-linear' and not any(
                'log-log' in rec for rec in summary['elasticity_reliability']['recommendations']):
            summary['elasticity_reliability']['recommendations'].append(
                "Consider testing log-log specification which often provides more stable elasticities")

    # Add a confidence assessment
    summary['overall_assessment'] = generate_confidence_assessment(summary)

    return summary


def generate_confidence_assessment(summary):
    """
    Generate an overall confidence assessment based on diagnostic summary.

    Args:
        summary: Diagnostic summary dictionary

    Returns:
        Dictionary with confidence assessment
    """
    # Count issues in each category
    model_issues = len(summary['model_quality']['issues'])
    stability_issues = len(summary['stability']['issues'])
    elasticity_issues = len(summary['elasticity_reliability']['issues'])

    total_issues = model_issues + stability_issues + elasticity_issues

    # Determine confidence level
    confidence_level = "High"
    confidence_score = 100

    if total_issues == 0:
        confidence_level = "Very High"
        confidence_score = 95
        assessment = "Model diagnostics show excellent performance with no significant issues detected."
    elif total_issues <= 2:
        confidence_level = "High"
        confidence_score = 80
        assessment = "Model performs well with minor issues that are unlikely to significantly impact results."
    elif total_issues <= 5:
        confidence_level = "Moderate"
        confidence_score = 60
        assessment = "Model shows acceptable performance but has several issues that should be addressed for improved reliability."
    elif total_issues <= 8:
        confidence_level = "Low"
        confidence_score = 40
        assessment = "Model has multiple issues that may significantly impact reliability. Consider substantial revisions."
    else:
        confidence_level = "Very Low"
        confidence_score = 20
        assessment = "Model has critical issues that severely compromise reliability. Major revisions required."

    # Add specific dimension assessments
    dimension_scores = {}

    if model_issues == 0:
        dimension_scores['model_quality'] = "Excellent"
    elif model_issues == 1:
        dimension_scores['model_quality'] = "Good"
    elif model_issues == 2:
        dimension_scores['model_quality'] = "Acceptable"
    else:
        dimension_scores['model_quality'] = "Poor"

    if stability_issues == 0:
        dimension_scores['stability'] = "Excellent"
    elif stability_issues == 1:
        dimension_scores['stability'] = "Good"
    elif stability_issues == 2:
        dimension_scores['stability'] = "Acceptable"
    else:
        dimension_scores['stability'] = "Poor"

    if elasticity_issues == 0:
        dimension_scores['elasticity_reliability'] = "Excellent"
    elif elasticity_issues == 1:
        dimension_scores['elasticity_reliability'] = "Good"
    elif elasticity_issues == 2:
        dimension_scores['elasticity_reliability'] = "Acceptable"
    else:
        dimension_scores['elasticity_reliability'] = "Poor"

    return {
        'confidence_level': confidence_level,
        'confidence_score': confidence_score,
        'assessment': assessment,
        'dimension_scores': dimension_scores,
        'issue_counts': {
            'model_quality': model_issues,
            'stability': stability_issues,
            'elasticity_reliability': elasticity_issues,
            'total': total_issues
        }
    }


# ---------------------------------------------------------------------------------------
# MAIN EXECUTION FUNCTION
# ---------------------------------------------------------------------------------------

def main_diagnostic_workflow(df, target, features, model_type='linear-linear', time_col=None):
    """
    Execute the main diagnostic workflow and generate report.

    Args:
        df: DataFrame with the data
        target: Target variable name
        features: List of feature variable names
        model_type: Type of model ('linear-linear', 'log-log', etc.)
        time_col: Column to use for time ordering

    Returns:
        Dictionary with diagnostic results and report path
    """
    logger.info(f"Starting main diagnostic workflow for {model_type} model")

    # Run comprehensive diagnostics
    results = run_comprehensive_diagnostics(df, target, features, model_type, time_col)

    # Generate HTML report
    report_path = generate_diagnostic_report(results, model_type)

    logger.info(f"Diagnostics completed. Report saved to {report_path}")

    return {
        'results': results,
        'report_path': report_path
    }


def generate_diagnostic_report(results, model_type):
    """
    Generate HTML report from diagnostic results.

    Args:
        results: Comprehensive diagnostic results
        model_type: Type of model

    Returns:
        Path to the generated report
    """
    # Basic implementation - in a real-world scenario, this would create a more
    # sophisticated HTML report with interactive elements

    report_path = f"mmm_diagnostics_{model_type}.html"

    # Get summary for report
    summary = results['diagnostic_summary']

    # Create simple HTML report
    with open(report_path, 'w') as f:
        f.write(f"<html><head><title>MMM Diagnostics Report - {model_type}</title>")
        f.write("<style>body{font-family:Arial,sans-serif;margin:20px;line-height:1.6}")
        f.write("h1,h2,h3{color:#333}.section{margin-bottom:30px;border:1px solid #ddd;padding:20px;border-radius:5px}")
        f.write(".issue{color:#d9534f}.recommendation{color:#5cb85c}")
        f.write(
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background-color:#f2f2f2}")
        f.write(".plot-container{margin:20px 0}</style></head><body>")

        # Header
        f.write(f"<h1>Marketing Mix Model Diagnostics Report</h1>")
        f.write(f"<p>Model Type: {model_type}</p>")

        # Overall Assessment
        f.write("<div class='section'>")
        f.write("<h2>Overall Assessment</h2>")
        f.write(
            f"<p><strong>Confidence Level:</strong> {summary['overall_assessment']['confidence_level']} ({summary['overall_assessment']['confidence_score']}%)</p>")
        f.write(f"<p>{summary['overall_assessment']['assessment']}</p>")

        # Dimension scores
        f.write("<h3>Dimension Scores</h3>")
        f.write("<table>")
        f.write("<tr><th>Dimension</th><th>Rating</th><th>Issues</th></tr>")
        for dim, score in summary['overall_assessment']['dimension_scores'].items():
            f.write(
                f"<tr><td>{dim.replace('_', ' ').title()}</td><td>{score}</td><td>{summary['overall_assessment']['issue_counts'][dim]}</td></tr>")
        f.write("</table>")
        f.write("</div>")

        # Issues and Recommendations
        for section in ['model_quality', 'stability', 'elasticity_reliability']:
            f.write("<div class='section'>")
            f.write(f"<h2>{section.replace('_', ' ').title()}</h2>")

            if summary[section]['issues']:
                f.write("<h3>Issues</h3><ul>")
                for issue in summary[section]['issues']:
                    f.write(f"<li class='issue'>{issue}</li>")
                f.write("</ul>")
            else:
                f.write("<p>No issues detected.</p>")

            if summary[section]['recommendations']:
                f.write("<h3>Recommendations</h3><ul>")
                for rec in summary[section]['recommendations']:
                    f.write(f"<li class='recommendation'>{rec}</li>")
                f.write("</ul>")

            f.write("</div>")

        # Diagnostic Plots
        f.write("<div class='section'>")
        f.write("<h2>Diagnostic Plots</h2>")

        for plot_name, plot_path in results.get('plots', {}).items():
            if plot_path:
                f.write("<div class='plot-container'>")
                f.write(f"<h3>{plot_name.replace('_', ' ').title()}</h3>")
                f.write(f"<img src='{plot_path}' alt='{plot_name}' style='max-width:100%'>")
                f.write("</div>")

        f.write("</div>")

        # Close HTML
        f.write("</body></html>")

    return report_path


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Load sample data
    df = pd.read_csv("/data/mmm_data.csv")

    # Define target and features
    target = 'Sales'
    features = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']

    # Run diagnostics for both model types
    results_linear = main_diagnostic_workflow(df, target, features, model_type='linear-linear')
    results_loglog = main_diagnostic_workflow(df, target, features, model_type='log-log')

    print(f"Linear model report: {results_linear['report_path']}")
    print(f"Log-log model report: {results_loglog['report_path']}")
