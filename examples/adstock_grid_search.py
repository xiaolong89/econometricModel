"""
Implement a grid search for optimal adstock parameters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
from pathlib import Path
import sys
import logging
import time

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from mmm.core import MarketingMixModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def grid_search_adstock_parameters(base_data, target='revenue', media_cols=None, control_cols=None):
    """
    Perform grid search to find optimal adstock parameters for each media channel.

    Args:
        base_data: Original DataFrame with media data
        target: Target variable name
        media_cols: List of media columns to optimize
        control_cols: Control variables to include in the model

    Returns:
        DataFrame with results and dictionary of optimal parameters
    """
    if media_cols is None:
        media_cols = [
            'tv_spend',
            'digital_display_spend',
            'search_spend',
            'social_media_spend',
            'video_spend',
            'email_spend'
        ]

    if control_cols is None:
        control_cols = [
            'price_index',
            'competitor_price_index',
            'gdp_index',
            'consumer_confidence'
        ]

    # Create log-transformed media columns
    data = base_data.copy()
    for col in media_cols:
        data[f"{col}_log"] = np.log1p(data[col])

    # Parameters to search
    decay_rates = {
        'tv_spend_log': [0.7, 0.8, 0.85, 0.9],
        'digital_display_spend_log': [0.5, 0.6, 0.7, 0.8],
        'search_spend_log': [0.2, 0.3, 0.4, 0.5],
        'social_media_spend_log': [0.4, 0.5, 0.6, 0.7],
        'video_spend_log': [0.6, 0.7, 0.8, 0.9],
        'email_spend_log': [0.3, 0.4, 0.5, 0.6]
    }

    max_lags = {
        'tv_spend_log': [4, 6, 8, 10],
        'digital_display_spend_log': [3, 4, 5, 6],
        'search_spend_log': [1, 2, 3, 4],
        'social_media_spend_log': [3, 4, 5, 6],
        'video_spend_log': [3, 4, 5, 6],
        'email_spend_log': [2, 3, 4, 5]
    }

    # Storage for results
    all_results = []

    # First, determine baseline performance with default parameters
    mmm = MarketingMixModel()
    mmm.load_data_from_dataframe(data)

    # Create column lists
    transformed_cols = [f"{col}_log" for col in media_cols]

    # Preprocess data
    mmm.preprocess_data(
        target=target,
        date_col='date' if 'date' in data.columns else None,
        media_cols=transformed_cols,
        control_cols=control_cols
    )

    # Apply default adstock transformations
    default_decay = {col: 0.7 for col in transformed_cols}
    default_max_lag = {col: 4 for col in transformed_cols}

    mmm.apply_adstock_to_all_media(
        media_cols=transformed_cols,
        decay_rates=default_decay,
        max_lags=default_max_lag
    )

    # Fit model
    baseline_results = mmm.fit_model()
    baseline_r2 = baseline_results.rsquared

    logger.info(f"Baseline R² with default parameters: {baseline_r2:.4f}")

    # Now optimize each channel's parameters one at a time
    best_params = {}
    best_r2 = baseline_r2

    for channel in media_cols:
        channel_log = f"{channel}_log"
        logger.info(f"Optimizing parameters for {channel}...")

        channel_best_r2 = -np.inf
        channel_best_params = {}

        # Try all combinations of decay rate and max lag
        for decay in decay_rates[channel_log]:
            for lag in max_lags[channel_log]:
                # Start with current best parameters
                current_decay = best_params.copy() if best_params else default_decay.copy()
                current_lag = copy.deepcopy(default_max_lag)

                # Update just this channel's parameters
                current_decay[channel_log] = decay
                current_lag[channel_log] = lag

                # Create a new model
                grid_mmm = MarketingMixModel()
                grid_mmm.load_data_from_dataframe(data)

                # Preprocess data
                grid_mmm.preprocess_data(
                    target=target,
                    date_col='date' if 'date' in data.columns else None,
                    media_cols=transformed_cols,
                    control_cols=control_cols
                )

                # Apply adstock with current parameters
                grid_mmm.apply_adstock_to_all_media(
                    media_cols=transformed_cols,
                    decay_rates=current_decay,
                    max_lags=current_lag
                )

                # Fit model
                try:
                    grid_results = grid_mmm.fit_model()
                    grid_r2 = grid_results.rsquared

                    # Store results
                    all_results.append({
                        'channel': channel,
                        'decay_rate': decay,
                        'max_lag': lag,
                        'r2': grid_r2
                    })

                    # Update best parameters if improvement
                    if grid_r2 > channel_best_r2:
                        channel_best_r2 = grid_r2
                        channel_best_params = {
                            'decay_rate': decay,
                            'max_lag': lag
                        }

                        logger.info(f"  New best for {channel}: decay={decay}, lag={lag}, R²={grid_r2:.4f}")
                except Exception as e:
                    logger.warning(f"Error with {channel}, decay={decay}, lag={lag}: {e}")

        # Update best parameters with this channel's best
        if channel_best_params:
            best_params[channel_log] = channel_best_params['decay_rate']
            default_max_lag[channel_log] = channel_best_params['max_lag']

            # If we improved overall R², update best R²
            if channel_best_r2 > best_r2:
                best_r2 = channel_best_r2

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Create dictionary of optimal parameters
    optimal_decay_rates = {col: best_params.get(col, default_decay[col]) for col in transformed_cols}
    optimal_max_lags = {col: default_max_lag[col] for col in transformed_cols}

    # Print optimal parameters
    print("\nOptimal Adstock Parameters:")
    for col in transformed_cols:
        print(f"{col}: decay_rate={optimal_decay_rates[col]}, max_lag={optimal_max_lags[col]}")

    print(f"\nImprovement in R²: {baseline_r2:.4f} → {best_r2:.4f} (+{(best_r2 - baseline_r2) * 100:.2f}%)")

    # Create heatmaps for each channel
    for channel in media_cols:
        channel_log = f"{channel}_log"
        channel_results = results_df[results_df['channel'] == channel].copy()

        if len(channel_results) > 0:
            # Reshape for heatmap
            pivot = channel_results.pivot_table(
                index='decay_rate',
                columns='max_lag',
                values='r2'
            )

            # Plot heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.4f')
            plt.title(f'R² by Adstock Parameters: {channel}')
            plt.tight_layout()
            plt.savefig(f'{channel}_adstock_heatmap.png')
            plt.close()

    # Return optimal parameters and results
    return {
        'optimal_decay_rates': optimal_decay_rates,
        'optimal_max_lags': optimal_max_lags,
        'best_r2': best_r2,
        'baseline_r2': baseline_r2,
        'results_df': results_df
    }


# You can run this as a separate script:

if __name__ == "__main__":
    # Load data
    data_path = Path(__file__).parent / 'mock_marketing_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])

    # Run grid search
    print("Starting adstock parameter grid search...")
    start_time = time.time()
    grid_results = grid_search_adstock_parameters(data)
    end_time = time.time()

    print(f"Grid search completed in {(end_time - start_time) / 60:.1f} minutes")

    # Use the optimal parameters in your main model
    optimal_decay_rates = grid_results['optimal_decay_rates']
    optimal_max_lags = grid_results['optimal_max_lags']

    print("Optimal parameters found and saved to PNG heatmaps.")