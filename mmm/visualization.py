"""
Visualization functions for Marketing Mix Models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs. Predicted Values"):
    """
    Plot actual vs. predicted values.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot values
    ax.plot(range(len(y_true)), y_true, 'b-', label='Actual')
    ax.plot(range(len(y_pred)), y_pred, 'r-', label='Predicted')

    # Add zero line
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Add labels and legend
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3)

    return fig


def plot_channel_contributions(coefficients, feature_values, feature_names=None):
    """
    Plot the contribution of each channel to the target variable.

    Args:
        coefficients: Model coefficients
        feature_values: Feature values (typically mean values)
        feature_names: Names of features (optional)

    Returns:
        matplotlib figure
    """
    if feature_names is None:
        feature_names = [f'Feature {i + 1}' for i in range(len(coefficients))]

    # Calculate contributions
    contributions = coefficients * feature_values

    # Create DataFrame for plotting
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': contributions,
        'Abs_Contribution': np.abs(contributions)
    }).sort_values('Abs_Contribution', ascending=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color positive and negative effects differently
    colors = ['green' if c >= 0 else 'red' for c in contrib_df['Contribution']]

    # Create bar plot
    ax.bar(contrib_df['Feature'], contrib_df['Contribution'], color=colors)

    # Add labels
    ax.set_title('Channel Contributions')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Contribution')

    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')

    # Add grid
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    return fig


def plot_correlation_matrix(data, title="Feature Correlation Matrix"):
    """
    Plot correlation matrix heatmap.

    Args:
        data: DataFrame or correlation matrix
        title: Plot title

    Returns:
        matplotlib figure
    """
    # Calculate correlation matrix if needed
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if data.shape[0] != data.shape[1]:
        corr = data.corr()
    else:
        corr = data

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)

    # Add title
    ax.set_title(title)

    plt.tight_layout()

    return fig


def plot_model_comparison(model_results, metric='r2'):
    """
    Plot comparison of different models based on a performance metric.

    Args:
        model_results: Dictionary mapping model names to performance metrics
        metric: Metric to compare

    Returns:
        matplotlib figure
    """
    # Extract model names and metric values
    models = list(model_results.keys())
    values = [model_results[model][metric] for model in models]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar plot
    ax.bar(models, values)

    # Add labels
    ax.set_title(f'Model Comparison ({metric.upper()})')
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())

    # Add values on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center')

    # Add grid
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    return fig


def plot_media_saturation(mmm, channel, points=100, range_pct=(50, 200)):
    """
    Plot the saturation curve for a given media channel.
    Shows how incremental spending affects incremental revenue.

    Args:
        mmm: The fitted MarketingMixModel instance
        channel: Media channel name (original, before transformations)
        points: Number of points to plot
        range_pct: Range of spending to simulate as percentage of current

    Returns:
        matplotlib figure
    """
    if mmm.results is None:
        raise ValueError("Model not fitted. Call fit_model() first.")

    # Get average spending for this channel
    avg_spend = mmm.preprocessed_data[channel].mean()

    # Get all transformations of this channel used in the model
    channel_vars = []
    for col in mmm.feature_names:
        if channel in col and col in mmm.results.params:
            channel_vars.append(col)

    if not channel_vars:
        raise ValueError(f"Channel {channel} not found in model features")

    # Create range of spending values
    min_pct, max_pct = range_pct
    spends = np.linspace(avg_spend * min_pct / 100, avg_spend * max_pct / 100, points)
    responses = np.zeros_like(spends)

    # Simulate response for each spending level
    for i, spend in enumerate(spends):
        # Initialize incremental response
        response = 0

        for var in channel_vars:
            # Get the coefficient
            coef = mmm.results.params[var]

            # Apply appropriate transformation to the spend
            if '_adstocked' in var:
                # Simple approximation of adstock effect
                if '_log' in var:
                    transformed = np.log1p(spend)
                elif '_hill' in var:
                    # Extract parameters from transformation name or use defaults
                    shape = 0.7
                    scale = avg_spend * 2
                    transformed = spend ** shape / (scale ** shape + spend ** shape)
                elif '_power' in var:
                    transformed = spend ** 0.7
                else:
                    transformed = spend

                # Apply coefficient
                response += coef * transformed

        responses[i] = response

    # Calculate marginal return on investment
    mroi = np.zeros_like(spends)
    mroi[1:] = np.diff(responses) / np.diff(spends)
    mroi[0] = mroi[1]  # Set first value same as second

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot total response curve
    ax1.plot(spends, responses, 'b-')
    ax1.set_title(f'Response Curve: {channel}')
    ax1.set_xlabel('Spend ($)')
    ax1.set_ylabel('Incremental Revenue ($)')
    ax1.axvline(x=avg_spend, color='r', linestyle='--', alpha=0.7, label='Current Avg Spend')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot marginal ROI curve
    ax2.plot(spends, mroi, 'g-')
    ax2.set_title(f'Marginal ROI: {channel}')
    ax2.set_xlabel('Spend ($)')
    ax2.set_ylabel('Marginal ROI ($/$ spent)')
    ax2.axvline(x=avg_spend, color='r', linestyle='--', alpha=0.7, label='Current Avg Spend')
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='ROI = 1.0')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig
