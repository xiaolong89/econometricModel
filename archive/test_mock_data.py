"""
Test script for evaluating MMM performance with clean mock data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import logging
import statsmodels.api as sm

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from mmm.core import MarketingMixModel
from mmm.modeling import fit_ridge_model, fit_lasso_model, fit_pca_model
from mmm.utils import create_train_test_split, evaluate_model_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_with_mock_data():
    """Run tests using the mock marketing data"""

    # ---- Load Mock Data ----
    logger.info("Loading mock marketing data...")
    data_path = Path(__file__).parent / 'mock_marketing_data.csv'
    mock_data = pd.read_csv(data_path)

    # Convert date column to datetime
    mock_data['date'] = pd.to_datetime(mock_data['date'])

    # Print dataset info
    logger.info(f"Data shape: {mock_data.shape}")
    logger.info(f"Date range: {mock_data['date'].min()} to {mock_data['date'].max()}")

    # ---- Basic Model Test ----
    logger.info("\n--- TESTING BASIC MODEL ---")

    # Initialize the MMM
    basic_mmm = MarketingMixModel()

    # Load data from dataframe
    logger.info("Loading data into MMM...")
    basic_mmm.load_data_from_dataframe(mock_data)

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

    # Preprocess data
    logger.info("Preprocessing data...")
    basic_mmm.preprocess_data(
        target='revenue',
        date_col='date',
        media_cols=media_cols,
        control_cols=control_cols
    )

    # Apply adstock transformations
    logger.info("Applying adstock transformations...")

    # Custom adstock parameters for each channel
    decay_rates = {
        'tv_spend': 0.85,  # TV has longer effect
        'digital_display_spend': 0.7,
        'search_spend': 0.3,  # Search has more immediate effect
        'social_media_spend': 0.6,
        'video_spend': 0.75,
        'email_spend': 0.4
    }

    max_lags = {
        'tv_spend': 8,  # TV has longer lag
        'digital_display_spend': 4,
        'search_spend': 2,  # Search has shorter lag
        'social_media_spend': 5,
        'video_spend': 6,
        'email_spend': 3
    }

    basic_mmm.apply_adstock_to_all_media(
        media_cols=media_cols,
        decay_rates=decay_rates,
        max_lags=max_lags
    )

    # Fit the model
    logger.info("Fitting the basic model...")
    basic_results = basic_mmm.fit_model()
    print("\nBasic Model Summary:")
    print(basic_results.summary())

    # Calculate elasticities
    logger.info("Calculating elasticities...")
    basic_elasticities = basic_mmm.calculate_elasticities()

    # ---- Advanced Model Test ----
    logger.info("\n--- TESTING ADVANCED MODEL ---")

    # Initialize a new MMM for advanced methods
    adv_mmm = MarketingMixModel()
    adv_mmm.load_data_from_dataframe(mock_data)

    # Preprocess with the same parameters
    adv_mmm.preprocess_data(
        target='revenue',
        date_col='date',
        media_cols=media_cols,
        control_cols=control_cols
    )

    # Apply the same adstock transformations
    adv_mmm.apply_adstock_to_all_media(
        media_cols=media_cols,
        decay_rates=decay_rates,
        max_lags=max_lags
    )

    # Create train/test split
    logger.info("Creating train/test split...")
    train_data, test_data = create_train_test_split(
        adv_mmm.preprocessed_data,
        train_frac=0.8,
        date_col='date'
    )

    # Extract X and y for training and testing
    X_train = train_data[adv_mmm.feature_names]
    y_train = train_data[adv_mmm.target]
    X_test = test_data[adv_mmm.feature_names]
    y_test = test_data[adv_mmm.target]

    # Fit OLS model for baseline
    adv_mmm.X = X_train
    adv_mmm.y = y_train
    ols_results = adv_mmm.fit_model()

    # Make predictions on test set
    X_test_const = sm.add_constant(X_test)
    ols_preds = ols_results.predict(X_test_const)
    ols_metrics = evaluate_model_performance(y_test, ols_preds)

    # Fit Ridge model
    logger.info("Fitting Ridge model...")
    ridge_results = fit_ridge_model(X_train, y_train,
                                    alphas=[0.1, 1.0, 10.0, 100.0],
                                    feature_names=adv_mmm.feature_names)

    # Get best Ridge model
    best_ridge_alpha = max(ridge_results.keys(),
                           key=lambda alpha: ridge_results[alpha]['r_squared'])
    best_ridge = ridge_results[best_ridge_alpha]

    # Make predictions with Ridge
    X_test_scaled = best_ridge['scaler'].transform(X_test)
    ridge_preds = best_ridge['model'].predict(X_test_scaled)
    ridge_metrics = evaluate_model_performance(y_test, ridge_preds)

    # Fit Lasso model
    logger.info("Fitting Lasso model...")
    lasso_results = fit_lasso_model(X_train, y_train,
                                    alphas=[0.1, 1.0, 10.0, 100.0],
                                    feature_names=adv_mmm.feature_names)

    # Get best Lasso model
    best_lasso_alpha = max(lasso_results.keys(),
                           key=lambda alpha: lasso_results[alpha]['r_squared'])
    best_lasso = lasso_results[best_lasso_alpha]

    # Make predictions with Lasso
    X_test_scaled = best_lasso['scaler'].transform(X_test)
    lasso_preds = best_lasso['model'].predict(X_test_scaled)
    lasso_metrics = evaluate_model_performance(y_test, lasso_preds)

    # Try PCA model
    try:
        logger.info("Fitting PCA model...")
        pca_results = fit_pca_model(X_train, y_train, explained_variance=0.95)

        # Make predictions with PCA
        X_test_scaled = pca_results['scaler'].transform(X_test)
        X_test_pca = pca_results['pca'].transform(X_test_scaled)
        X_test_pca_const = np.c_[np.ones(X_test_pca.shape[0]), X_test_pca]
        pca_preds = pca_results['model'].predict(X_test_pca_const)
        pca_metrics = evaluate_model_performance(y_test, pca_preds)
    except Exception as e:
        logger.warning(f"PCA model failed: {e}")
        pca_metrics = None

    # Compare models
    logger.info("\n--- MODEL COMPARISON ---")
    print("\nTest Set Performance:")
    print(f"OLS R²: {ols_metrics['r2']:.4f}, RMSE: {ols_metrics['rmse']:.2f}")
    print(f"Ridge (α={best_ridge_alpha}) R²: {ridge_metrics['r2']:.4f}, RMSE: {ridge_metrics['rmse']:.2f}")
    print(f"Lasso (α={best_lasso_alpha}) R²: {lasso_metrics['r2']:.4f}, RMSE: {lasso_metrics['rmse']:.2f}")
    if pca_metrics:
        print(f"PCA R²: {pca_metrics['r2']:.4f}, RMSE: {pca_metrics['rmse']:.2f}")

    # Compare elasticities
    logger.info("\n--- ELASTICITY COMPARISON ---")

    # True elasticities from the mock data generation
    true_elasticities = {
        'tv_spend': 0.20,
        'digital_display_spend': 0.15,
        'search_spend': 0.25,
        'social_media_spend': 0.18,
        'video_spend': 0.12,
        'email_spend': 0.05
    }

    # Extract estimated elasticities from models
    estimated_elasticities = {}
    for channel in true_elasticities.keys():
        # For OLS, use the elasticities calculated by the MMM
        ols_value = basic_elasticities.get(f'{channel}_adstocked', 0)

        # For Ridge, calculate elasticity using coefficient and means
        ridge_coef_idx = adv_mmm.feature_names.index(f'{channel}_adstocked')
        ridge_coef = best_ridge['model'].coef_[ridge_coef_idx]
        ridge_value = ridge_coef * (X_train[f'{channel}_adstocked'].mean() / y_train.mean())

        # For Lasso, calculate elasticity using coefficient and means
        lasso_coef_idx = adv_mmm.feature_names.index(f'{channel}_adstocked')
        lasso_coef = best_lasso['model'].coef_[lasso_coef_idx]
        lasso_value = lasso_coef * (X_train[f'{channel}_adstocked'].mean() / y_train.mean())

        # Store results
        estimated_elasticities[channel] = {
            'true': true_elasticities[channel],
            'ols': ols_value,
            'ridge': ridge_value,
            'lasso': lasso_value
        }

    # Print elasticity comparison
    print("\nElasticity Comparison (True vs. Estimated):")
    for channel, values in estimated_elasticities.items():
        print(
            f"{channel}: True={values['true']:.3f}, OLS={values['ols']:.3f}, Ridge={values['ridge']:.3f}, Lasso={values['lasso']:.3f}")

    # Plot elasticity comparison
    plt.figure(figsize=(14, 8))
    width = 0.2
    x = np.arange(len(true_elasticities))

    plt.bar(x - 1.5 * width, [true_elasticities[c] for c in true_elasticities.keys()], width, label='True')
    plt.bar(x - 0.5 * width, [estimated_elasticities[c]['ols'] for c in true_elasticities.keys()], width, label='OLS')
    plt.bar(x + 0.5 * width, [estimated_elasticities[c]['ridge'] for c in true_elasticities.keys()], width,
            label='Ridge')
    plt.bar(x + 1.5 * width, [estimated_elasticities[c]['lasso'] for c in true_elasticities.keys()], width,
            label='Lasso')

    plt.xlabel('Channel')
    plt.ylabel('Elasticity')
    plt.title('True vs. Estimated Elasticities by Channel')
    plt.xticks(x, true_elasticities.keys(), rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('elasticity_comparison.png')
    plt.show()

    # Plot actual vs. predicted for best model
    plt.figure(figsize=(12, 6))

    # Get best model predictions based on R²
    best_model = max(
        [('OLS', ols_metrics['r2'], ols_preds),
         ('Ridge', ridge_metrics['r2'], ridge_preds),
         ('Lasso', lasso_metrics['r2'], lasso_preds)],
        key=lambda x: x[1]
    )

    model_name, r2, predictions = best_model

    plt.plot(test_data['date'], y_test, 'b-', label='Actual')
    plt.plot(test_data['date'], predictions, 'r-', label=f'Predicted ({model_name})')
    plt.title(f'Actual vs. Predicted Revenue (Test Set) - {model_name} Model')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.show()

    logger.info(f"Plots saved to current directory.")
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    # Fix potential import error
    import statsmodels.api as sm

    test_with_mock_data()
