"""
Advanced Marketing Mix Model example with regularization and PCA.
"""

import logging
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from mmm.core import MarketingMixModel
from mmm.preprocessing import add_seasonality_features
from mmm.modeling import fit_ridge_model, fit_lasso_model, fit_pca_model
from mmm.utils import calculate_vif, create_train_test_split
from mmm.visualization import plot_correlation_matrix, plot_model_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Advanced MMM example with multiple modeling approaches."""
    try:
        # Initialize the MMM
        mmm = MarketingMixModel()

        # Set paths relative to project structure
        current_dir = Path(__file__).parent.parent
        data_path = str(current_dir / 'data' / 'synthetic_advertising_data_v2.csv')

        # Load and preprocess data
        logger.info("Loading data...")
        mmm.load_data(data_path)

        # Define column groups
        media_cols = [
            'ad_spend_linear_tv',
            'ad_spend_digital',
            'ad_spend_search',
            'ad_spend_social',
            'ad_spend_programmatic'
        ]

        control_cols = [
            'gdp',
            'inflation',
            'consumer_confidence',
            'consumer_sentiment'
        ]

        # Preprocess data
        logger.info("Preprocessing data...")
        mmm.preprocess_data(
            target='sales',
            date_col='week',
            media_cols=media_cols,
            control_cols=control_cols
        )

        # Add seasonality features
        logger.info("Adding seasonality features...")
        if 'week' in mmm.preprocessed_data.columns:
            seasonal_df = add_seasonality_features(mmm.preprocessed_data, 'week')

            # Update preprocessed data with seasonal features
            mmm.preprocessed_data = seasonal_df

            # Add quarter dummies to feature names
            quarter_features = [col for col in seasonal_df.columns if col.startswith('quarter_')]
            mmm.feature_names.extend(quarter_features)

        # Apply customized adstock transformations
        logger.info("Applying adstock transformations...")
        decay_rates = {
            'ad_spend_linear_tv': 0.85,  # TV has longer effect
            'ad_spend_digital': 0.6,
            'ad_spend_search': 0.3,  # Search has more immediate effect
            'ad_spend_social': 0.5,
            'ad_spend_programmatic': 0.7
        }

        lag_weights = {
            'ad_spend_linear_tv': 0.4,  # More weight on lagged effect for TV
            'ad_spend_digital': 0.3,
            'ad_spend_search': 0.1,  # Less weight on lagged effect for search
            'ad_spend_social': 0.2,
            'ad_spend_programmatic': 0.3
        }

        max_lags = {
            'ad_spend_linear_tv': 6,  # Longer lag for TV
            'ad_spend_digital': 4,
            'ad_spend_search': 2,  # Shorter lag for search
            'ad_spend_social': 3,
            'ad_spend_programmatic': 4
        }

        mmm.apply_adstock_to_all_media(
            media_cols=media_cols,
            decay_rates=decay_rates,
            lag_weights=lag_weights,
            max_lags=max_lags
        )

        # Add log transformations for diminishing returns
        logger.info("Adding transformations for diminishing returns...")
        for col in [f"{mc}_adstocked" for mc in media_cols]:
            # Add small constant to handle zeros
            mmm.preprocessed_data[f"{col}_log"] = np.log1p(mmm.preprocessed_data[col])
            mmm.feature_names.append(f"{col}_log")

        # Check for multicollinearity
        logger.info("Checking for multicollinearity...")
        X_check = mmm.preprocessed_data[mmm.feature_names]
        vif_data = calculate_vif(X_check)
        print("\nVariance Inflation Factors:")
        print(vif_data)

        # Plot correlation matrix
        corr_matrix = plot_correlation_matrix(X_check)
        corr_matrix.savefig(current_dir / 'correlation_matrix.png')

        # Split data into train and test sets
        logger.info("Creating train/test split...")
        train_data, test_data = create_train_test_split(
            mmm.preprocessed_data,
            train_frac=0.8,
            date_col='week' if 'week' in mmm.preprocessed_data.columns else None
        )

        # Extract X and y for training and testing
        X_train = train_data[mmm.feature_names]
        y_train = train_data[mmm.target]
        X_test = test_data[mmm.feature_names]
        y_test = test_data[mmm.target]

        # 1. Fit OLS model
        logger.info("Fitting OLS model...")
        mmm.X = X_train
        mmm.y = y_train
        ols_results = mmm.fit_model()

        # Make predictions on test set
        X_test_const = pd.DataFrame(X_test)
        X_test_const.insert(0, 'const', 1)
        ols_preds = ols_results.predict(X_test_const)
        ols_r2 = np.corrcoef(y_test, ols_preds)[0, 1] ** 2

        # 2. Fit Ridge models
        logger.info("Fitting Ridge models...")
        ridge_results = fit_ridge_model(X_train, y_train,
                                        alphas=[0.1, 1.0, 10.0, 100.0],
                                        feature_names=mmm.feature_names)

        # Get best Ridge model
        best_ridge_alpha = max(ridge_results.keys(),
                               key=lambda alpha: ridge_results[alpha]['r_squared'])
        best_ridge = ridge_results[best_ridge_alpha]

        # Make predictions with Ridge
        X_test_scaled = best_ridge['scaler'].transform(X_test)
        ridge_preds = best_ridge['model'].predict(X_test_scaled)
        ridge_r2 = np.corrcoef(y_test, ridge_preds)[0, 1] ** 2

        # 3. Fit Lasso models
        logger.info("Fitting Lasso models...")
        lasso_results = fit_lasso_model(X_train, y_train,
                                        alphas=[0.1, 1.0, 10.0, 100.0],
                                        feature_names=mmm.feature_names)

        # Get best Lasso model
        best_lasso_alpha = max(lasso_results.keys(),
                               key=lambda alpha: lasso_results[alpha]['r_squared'])
        best_lasso = lasso_results[best_lasso_alpha]

        # Make predictions with Lasso
        X_test_scaled = best_lasso['scaler'].transform(X_test)
        lasso_preds = best_lasso['model'].predict(X_test_scaled)
        lasso_r2 = np.corrcoef(y_test, lasso_preds)[0, 1] ** 2

        # 4. Fit PCA model
        logger.info("Fitting PCA model...")
        pca_results = fit_pca_model(X_train, y_train, explained_variance=0.95)

        # Make predictions with PCA
        X_test_scaled = pca_results['scaler'].transform(X_test)
        X_test_pca = pca_results['pca'].transform(X_test_scaled)
        X_test_pca_const = np.c_[np.ones(X_test_pca.shape[0]), X_test_pca]
        pca_preds = pca_results['model'].predict(X_test_pca_const)
        pca_r2 = np.corrcoef(y_test, pca_preds)[0, 1] ** 2

        # Compare models
        logger.info("Comparing models on test data...")
        model_performance = {
            'OLS': {'r2': ols_r2},
            'Ridge': {'r2': ridge_r2},
            'Lasso': {'r2': lasso_r2},
            'PCA': {'r2': pca_r2}
        }

        # Print comparison
        print("\nModel Test Performance:")
        for model, metrics in model_performance.items():
            print(f"{model} Test R²: {metrics['r2']:.4f}")

        # Plot comparison
        comparison_fig = plot_model_comparison(model_performance, metric='r2')
        comparison_fig.savefig(current_dir / 'model_comparison.png')

        # Identify best model
        best_model = max(model_performance.keys(),
                         key=lambda model: model_performance[model]['r2'])

        print(f"\nBest model: {best_model} (Test R²: {model_performance[best_model]['r2']:.4f})")

        # Keep plots open if interactive
        plt.show()

        logger.info("Advanced MMM analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()