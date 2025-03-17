"""
Core implementation of the Marketing Mix Model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import logging
from pathlib import Path
import pickle
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import internal modules
from mmm.preprocessing import detect_media_columns, detect_control_columns, check_stationarity
from mmm.adstock import apply_adstock
from mmm.utils import calculate_vif

class MarketingMixModel:
    """
    A comprehensive Marketing Mix Model (MMM) implementation for measuring the effectiveness
    of marketing channels and optimizing budget allocation.

    This class handles:
    - Data loading and preprocessing
    - Adstock transformations for modeling lagged effects
    - Model specification and fitting
    - Cross-validation and model evaluation
    - Elasticity and ROI calculations
    - Budget optimization
    """

    def __init__(self, data_path: str = None):
        """
        Initialize the MarketingMixModel.

        Args:
            data_path: Path to the input data file (CSV)
        """
        self.data_path = data_path
        self.data = None
        self.model = None
        self.results = None
        self.preprocessed_data = None
        self.X = None
        self.y = None
        self.feature_names = None
        # Add support for multiple targets
        self.target = None
        self.units_target = None
        self.adstock_transformations = {}
        self.cv_results = {}
        self.elasticities = {}
        self.roi_metrics = {}

    def load_data(self, data_path=None):
        """Load data from CSV file."""
        try:
            if data_path:
                self.data_path = data_path

            # Use default paths if none provided
            if not self.data_path:
                # Find project root directory based on common folder structure
                current_dir = Path.cwd()
                data_dir = current_dir / 'data'

                if data_dir.exists():
                    self.data_path = str(data_dir / 'synthetic_advertising_data_v2.csv')
                    logger.info(f"Using default data file: {self.data_path}")
                else:
                    raise FileNotFoundError("Data directory not found. Please provide an explicit data_path.")

            # Load data
            self.data = pd.read_csv(self.data_path)

            # Update date column parsing
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'], format='%m/%d/%Y')

            # Flexible target column detection
            self.target = 'Sales' if 'Sales' in self.data.columns else 'Revenue'
            self.units_target = 'Units Sold'

            logger.info(f"Successfully loaded data from {self.data_path}")
            logger.info(f"Data shape: {self.data.shape}")
            logger.info(f"Target column: {self.target}")
            logger.info(f"Units column: {self.units_target}")

            # Display basic info about the data
            logger.info(f"Columns: {self.data.columns.tolist()}")

            return self.data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def load_data_from_dataframe(self, dataframe):
        """Load data from an existing DataFrame."""
        self.data = dataframe.copy()
        logger.info(f"Data loaded from DataFrame")
        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Columns: {self.data.columns.tolist()}")
        return self.data

    def preprocess_data(self,
                        target=None,
                        date_col='Date',
                        media_cols=None,
                        control_cols=None):
        """Preprocess data for modeling."""
        try:
            if self.data is None:
                raise ValueError("No data loaded. Please call load_data() first.")

            # Import functions from preprocessing module
            from mmm.preprocessing import detect_media_columns, detect_control_columns, check_stationarity

            # Make a copy to avoid modifying the original
            df = self.data.copy()

            # Handle date column
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y')
                df = df.sort_values(by=date_col)
                logger.info(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

            # Set target variable with flexible detection
            if target is None:
                target = 'Sales' if 'Sales' in df.columns else 'Revenue'
            self.target = target

            # Identify media channels if not provided
            if media_cols is None:
                media_cols = detect_media_columns(df)
                logger.info(f"Automatically identified media columns: {media_cols}")

            # Identify control variables if not provided
            if control_cols is None:
                control_cols = detect_control_columns(df, target, date_col, media_cols)
                logger.info(f"Automatically identified control columns: {control_cols}")

            # Store feature names
            self.feature_names = media_cols + control_cols

            # Check for missing values
            missing_values = df[self.feature_names + [target]].isnull().sum()
            if missing_values.sum() > 0:
                logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")

                # Simple imputation - fill with mean for this example
                df[self.feature_names + [target]] = df[self.feature_names + [target]].fillna(
                    df[self.feature_names + [target]].mean())

                logger.info("Missing values imputed with column means")

            # Check for stationarity of the target variable
            check_stationarity(df[target])

            # Store the preprocessed data
            self.preprocessed_data = df

            # Set up X and y for modeling
            self.X = df[self.feature_names]
            self.y = df[target]

            # Add support for Units Sold
            if 'Units Sold' in df.columns:
                self.units_target = 'Units Sold'

            logger.info("Data preprocessing completed successfully")
            return df

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def apply_adstock(self, media_col, decay_rate=0.7, lag_weight=0.3, max_lag=4):
        """Apply adstock transformation to a media column."""
        # Implementation moved to adstock.py
        # This method now serves as a wrapper
        transformed = apply_adstock(
            self.preprocessed_data[media_col].values,
            decay_rate,
            lag_weight,
            max_lag
        )

        # Store the transformation parameters
        self.adstock_transformations[media_col] = {
            'decay_rate': decay_rate,
            'lag_weight': lag_weight,
            'max_lag': max_lag
        }

        return transformed

    def apply_adstock_to_all_media(self,
                                   media_cols=None,
                                   decay_rates=None,
                                   lag_weights=None,
                                   max_lags=None):
        """
        Apply adstock transformations to all media channels.

        Args:
            media_cols: List of media columns to transform
            decay_rates: Dictionary mapping media columns to decay rates
            lag_weights: Dictionary mapping media columns to lag weights
            max_lags: Dictionary mapping media columns to max lags

        Returns:
            DataFrame with transformed media values
        """
        try:
            if self.preprocessed_data is None:
                raise ValueError("No preprocessed data available. Call preprocess_data() first.")

            # Use all media columns if not specified
            if media_cols is None:
                # Get media columns from feature names (assuming control variables come after media)
                all_features = set(self.feature_names)
                if len(all_features) == 0:
                    raise ValueError("No feature names available. Call preprocess_data() first.")

                # Try to automatically identify media columns based on common names
                potential_media_keywords = ['spend', 'tv', 'radio', 'digital', 'social',
                                            'search', 'display', 'video', 'email', 'print',
                                            'outdoor', 'media', 'facebook', 'google', 'twitter',
                                            'tiktok', 'youtube']

                media_cols = []
                for col in self.preprocessed_data.columns:
                    if any(keyword in col.lower() for keyword in potential_media_keywords):
                        if col in all_features:
                            media_cols.append(col)

            # Set default parameters if not provided
            if decay_rates is None:
                decay_rates = {col: 0.7 for col in media_cols}

            if lag_weights is None:
                lag_weights = {col: 0.3 for col in media_cols}

            if max_lags is None:
                max_lags = {col: 4 for col in media_cols}

            # Create a copy of the preprocessed data
            transformed_data = self.preprocessed_data.copy()

            # Apply adstock to each media channel
            for col in media_cols:
                decay_rate = decay_rates.get(col, 0.7)
                lag_weight = lag_weights.get(col, 0.3)
                max_lag = max_lags.get(col, 4)

                logger.info(
                    f"Applying adstock to {col} with decay_rate={decay_rate}, lag_weight={lag_weight}, max_lag={max_lag}")

                transformed_values = self.apply_adstock(col, decay_rate, lag_weight, max_lag)

                # Replace original values with transformed ones
                col_name = f"{col}_adstocked"
                transformed_data[col_name] = transformed_values

                # Update feature names to use the adstocked version
                self.feature_names = [col_name if x == col else x for x in self.feature_names]

            # Update X with transformed features
            self.X = transformed_data[self.feature_names]

            # Update preprocessed data
            self.preprocessed_data = transformed_data

            return transformed_data

        except Exception as e:
            logger.error(f"Error applying adstock transformations: {str(e)}")
            raise

    def fit_model(self, add_intercept=True):
        """
        Fit the MMM model using OLS regression.

        Args:
            add_intercept: Whether to add an intercept term to the model

        Returns:
            Fitted model results
        """
        try:
            if self.X is None or self.y is None:
                raise ValueError("X and y not set. Call preprocess_data() first.")

            # Prepare the design matrix
            X_design = self.X.copy()

            # Add intercept if requested
            if add_intercept:
                X_design = sm.add_constant(X_design)

            # Fit the model
            self.model = sm.OLS(self.y, X_design)
            self.results = self.model.fit()

            # Log model summary
            logger.info("Model fitted successfully")
            logger.info(f"R-squared: {self.results.rsquared:.4f}")
            logger.info(f"Adjusted R-squared: {self.results.rsquared_adj:.4f}")

            # Check for multicollinearity
            if add_intercept:
                # Skip the constant term
                vif_data = calculate_vif(X_design.iloc[:, 1:])
            else:
                vif_data = calculate_vif(X_design)

            high_vif_features = vif_data[vif_data['VIF'] > 10]
            if not high_vif_features.empty:
                logger.warning(f"High multicollinearity detected in features:\n{high_vif_features}")

            return self.results

        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise

    def cross_validate(self, n_splits=5, test_size=4):
        """
        Perform time series cross-validation.

        Args:
            n_splits: Number of cross-validation splits
            test_size: Number of periods in each test set

        Returns:
            Dictionary with cross-validation results
        """
        try:
            if self.X is None or self.y is None:
                raise ValueError("X and y not set. Call preprocess_data() first.")

            # Set up time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

            # Initialize results storage
            rmse_scores = []
            r2_scores = []
            mae_scores = []
            mape_scores = []
            actual_vs_predicted = []

            # Add constant for statsmodels
            X_with_const = sm.add_constant(self.X)

            # Perform cross-validation
            for train_idx, test_idx in tscv.split(X_with_const):
                # Split data
                X_train, X_test = X_with_const.iloc[train_idx], X_with_const.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                # Fit model on training data
                model = sm.OLS(y_train, X_train).fit()

                # Predict on test data
                y_pred = model.predict(X_test)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                mae = np.mean(np.abs(y_test - y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                # Store metrics
                rmse_scores.append(rmse)
                r2_scores.append(r2)
                mae_scores.append(mae)
                mape_scores.append(mape)

                # Store actual vs predicted for plotting
                fold_results = pd.DataFrame({
                    'actual': y_test,
                    'predicted': y_pred,
                    'fold': len(actual_vs_predicted) + 1
                })
                actual_vs_predicted.append(fold_results)

            # Combine all CV results
            all_predictions = pd.concat(actual_vs_predicted)

            # Store cross-validation results
            self.cv_results = {
                'rmse': rmse_scores,
                'r2': r2_scores,
                'mae': mae_scores,
                'mape': mape_scores,
                'predictions': all_predictions
            }

            # Log average metrics
            logger.info(f"Cross-validation results:")
            logger.info(f"Avg RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
            logger.info(f"Avg R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
            logger.info(f"Avg MAE: {np.mean(mae_scores):.4f} (±{np.std(mae_scores):.4f})")
            logger.info(f"Avg MAPE: {np.mean(mape_scores):.2f}% (±{np.std(mape_scores):.2f}%)")

            return self.cv_results

        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise

    def plot_cv_results(self):
        """
        Plot cross-validation results including actual vs. predicted values.
        """
        if not self.cv_results:
            raise ValueError("No cross-validation results available. Run cross_validate() first.")

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot actual vs predicted for each fold
        predictions_df = self.cv_results['predictions']

        # Fix: Sort only by fold to avoid KeyError
        predictions_df = predictions_df.sort_values(['fold'])

        # Get unique folds
        folds = predictions_df['fold'].unique()

        # Plot each fold
        for fold in folds:
            fold_data = predictions_df[predictions_df['fold'] == fold]
            ax1.plot(range(len(fold_data)), fold_data['actual'], 'b-', alpha=0.3)
            ax1.plot(range(len(fold_data)), fold_data['predicted'], 'r-', alpha=0.3)

            # Add fold boundaries
            if fold < max(folds):
                ax1.axvline(x=len(fold_data), color='gray', linestyle='--', alpha=0.5)

        # Add legend
        ax1.plot([], [], 'b-', label='Actual')
        ax1.plot([], [], 'r-', label='Predicted')
        ax1.legend()
        ax1.set_title('Actual vs. Predicted Values Across Folds')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel(self.target)

        # Plot error metrics across folds
        metrics = ['rmse', 'r2', 'mae', 'mape']
        metric_labels = ['RMSE', 'R²', 'MAE', 'MAPE (%)']

        # Create bar chart of average metrics
        avg_metrics = [np.mean(self.cv_results[m]) for m in metrics]
        std_metrics = [np.std(self.cv_results[m]) for m in metrics]

        ax2.bar(metric_labels, avg_metrics, yerr=std_metrics, capsize=10)
        ax2.set_title('Average Cross-Validation Metrics')
        ax2.set_ylabel('Value')
        ax2.grid(axis='y', alpha=0.3)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

    def calculate_elasticities(self):
        """
        Calculate elasticities for all media channels with proper scaling for log-transformed variables.

        Returns:
            Dictionary mapping channels to their elasticities
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit_model() first.")

        # Get coefficients
        coefficients = self.results.params

        # Initialize elasticities dictionary
        elasticities = {}

        # Find all base media columns
        base_media_cols = []
        for col in self.preprocessed_data.columns:
            # Look for standard media spend column naming patterns
            if any(pattern in col.lower() for pattern in
                   ['_spend', 'tv_', 'digital', 'search', 'social', 'video', 'email']):
                # Exclude transformed columns
                if not any(suffix in col for suffix in ['_adstocked', '_log', '_hill', '_power']):
                    base_media_cols.append(col)

        # For each base media column, find the best transformed version to use for elasticity
        for base_col in base_media_cols:
            # Initialize with zero elasticity
            elasticities[base_col] = 0
            elasticity_found = False

            # Try all possible transformation patterns
            possible_patterns = [
                f"{base_col}_log_adstocked",  # Log first, then adstock
                f"{base_col}_adstocked_log",  # Adstock first, then log
                f"{base_col}_log",  # Just log
                f"{base_col}_adstocked",  # Just adstock
                base_col  # Original
            ]

            # Find first pattern that exists in coefficients
            for pattern in possible_patterns:
                if pattern in coefficients:
                    # For log-transformed variables
                    if '_log' in pattern:
                        # The coefficient represents the absolute change in Y for a 100% increase in X
                        # To convert to elasticity (% change in Y for 1% change in X), we need to scale it
                        avg_y = self.y.mean()
                        elasticities[base_col] = coefficients[pattern] / avg_y
                        logger.info(f"Elasticity for {base_col} (using {pattern}): {elasticities[base_col]:.4f}")
                        logger.info(f"  Calculation: coefficient ({coefficients[pattern]:.2f}) / mean Y ({avg_y:.2f})")
                        elasticity_found = True
                        break

                    # For non-log variables (elasticity = coef * X/Y)
                    else:
                        avg_x = self.X[pattern].mean()
                        avg_y = self.y.mean()

                        if avg_x > 0 and avg_y > 0:
                            elasticities[base_col] = coefficients[pattern] * (avg_x / avg_y)
                            logger.info(f"Elasticity for {base_col} (using {pattern}): {elasticities[base_col]:.4f}")
                            logger.info(
                                f"  Calculation: coefficient ({coefficients[pattern]:.2f}) * (mean X ({avg_x:.2f}) / mean Y ({avg_y:.2f}))")
                            elasticity_found = True
                            break

            if not elasticity_found:
                # Try a more flexible search by iterating through all coefficient names
                for coef_name in coefficients.index:
                    # Check if this coefficient relates to the current base column
                    if base_col in coef_name and ('_log' in coef_name or '_adstocked' in coef_name):
                        # For log-transformed variables
                        if '_log' in coef_name:
                            avg_y = self.y.mean()
                            elasticities[base_col] = coefficients[coef_name] / avg_y
                            logger.info(f"Elasticity for {base_col} (using {coef_name}): {elasticities[base_col]:.4f}")
                            logger.info(
                                f"  Calculation: coefficient ({coefficients[coef_name]:.2f}) / mean Y ({avg_y:.2f})")
                            elasticity_found = True
                            break
                        # For other transformations
                        else:
                            avg_x = self.X[coef_name].mean()
                            avg_y = self.y.mean()

                            if avg_x > 0 and avg_y > 0:
                                elasticities[base_col] = coefficients[coef_name] * (avg_x / avg_y)
                                logger.info(
                                    f"Elasticity for {base_col} (using {coef_name}): {elasticities[base_col]:.4f}")
                                logger.info(
                                    f"  Calculation: coefficient ({coefficients[coef_name]:.2f}) * (mean X ({avg_x:.2f}) / mean Y ({avg_y:.2f}))")
                                elasticity_found = True
                                break

            if not elasticity_found:
                logger.warning(f"No suitable coefficient found for {base_col}")

        # Store elasticities
        self.elasticities = elasticities

        return elasticities

    def calculate_roi(self, cost_columns=None):
        """
        Calculate Return on Investment (ROI) for media channels.

        Args:
            cost_columns: List of columns containing cost information. If None,
                         columns with 'spend' in the name will be used.

        Returns:
            Dictionary with ROI metrics for each channel
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit_model() first.")

        if self.preprocessed_data is None:
            raise ValueError("No preprocessed data available.")

        # Identify cost columns if not provided
        if cost_columns is None:
            cost_columns = [col for col in self.preprocessed_data.columns if 'spend' in col.lower()]

            if not cost_columns:
                logger.warning("No spend columns found. Cannot calculate ROI.")
                return {}

        # Calculate ROI metrics
        roi_metrics = {}

        for col in cost_columns:
            # Get corresponding feature name (might be adstocked version)
            feature_col = col
            if f"{col}_adstocked" in self.feature_names:
                feature_col = f"{col}_adstocked"

            if feature_col in self.results.params.index:
                # Get coefficient
                coef = self.results.params[feature_col]

                # Calculate metrics
                total_spend = self.preprocessed_data[col].sum()
                total_effect = coef * self.X[feature_col].sum()

                # Calculate ROI as (effect / spend)
                roi = total_effect / total_spend if total_spend > 0 else 0

                # Calculate ROAS (Return on Ad Spend)
                roas = (total_effect / total_spend) if total_spend > 0 else 0

                # Calculate CPA (Cost Per Acquisition)
                cpa = total_spend / total_effect if total_effect > 0 else float('inf')

                # Store metrics
                roi_metrics[col] = {
                    'coefficient': coef,
                    'total_spend': total_spend,
                    'total_effect': total_effect,
                    'roi': roi,
                    'roas': roas,
                    'cpa': cpa
                }

                logger.info(f"ROI metrics for {col}:")
                logger.info(f"  Coefficient: {coef:.4f}")
                logger.info(f"  Total Spend: {total_spend:.2f}")
                logger.info(f"  Total Effect: {total_effect:.2f}")
                logger.info(f"  ROI: {roi:.4f}")
                logger.info(f"  ROAS: {roas:.4f}")
                logger.info(f"  CPA: {cpa:.4f}")

        # Store ROI metrics
        self.roi_metrics = roi_metrics

        return roi_metrics

    def plot_channel_contributions(self):
        """
        Plot the contribution of each channel to the target variable.
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit_model() first.")

        # Get coefficients
        coefs = self.results.params

        # Remove intercept if present
        if 'const' in coefs:
            coefs = coefs.drop('const')

        # Calculate contribution for each channel
        contributions = {}

        for feature in coefs.index:
            # Contribution = coefficient * average feature value
            contribution = coefs[feature] * self.X[feature].mean()
            contributions[feature] = contribution

        # Convert to DataFrame for plotting
        contrib_df = pd.DataFrame({
            'Channel': contributions.keys(),
            'Contribution': contributions.values()
        })

        # Sort by absolute contribution
        contrib_df['Abs_Contribution'] = contrib_df['Contribution'].abs()
        contrib_df = contrib_df.sort_values('Abs_Contribution', ascending=False)

        # Create a color map (blue for positive, red for negative)
        colors = ['green' if c >= 0 else 'red' for c in contrib_df['Contribution']]

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.bar(contrib_df['Channel'], contrib_df['Contribution'], color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.title('Channel Contributions to ' + self.target)
        plt.xlabel('Channel')
        plt.ylabel('Contribution')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_elasticities(self):
        """
        Plot elasticities for all media channels.
        """
        if not self.elasticities:
            self.calculate_elasticities()

        # Convert to DataFrame for plotting
        elasticity_df = pd.DataFrame({
            'Channel': self.elasticities.keys(),
            'Elasticity': self.elasticities.values()
        })

        # Sort by absolute elasticity
        elasticity_df['Abs_Elasticity'] = elasticity_df['Elasticity'].abs()
        elasticity_df = elasticity_df.sort_values('Abs_Elasticity', ascending=False)

        # Create a color map (blue for positive, red for negative)
        colors = ['blue' if e >= 0 else 'red' for e in elasticity_df['Elasticity']]

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.bar(elasticity_df['Channel'], elasticity_df['Elasticity'], color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.title('Media Channel Elasticities')
        plt.xlabel('Channel')
        plt.ylabel('Elasticity')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def optimize_budget(self, total_budget=None, cost_columns=None):
        """
        Simple budget allocation optimization based on channel elasticities.

        Args:
            total_budget: Total budget to allocate. If None, uses the sum of current spend.
            cost_columns: List of columns containing cost information. If None,
                        columns with 'spend' in the name will be used.

        Returns:
            Dictionary with optimized budget allocation
        """
        if not self.elasticities:
            self.calculate_elasticities()

        # Import optimization function
        from mmm.optimization import simple_budget_allocation

        # Identify cost columns if not provided
        if cost_columns is None:
            cost_columns = [col for col in self.preprocessed_data.columns if 'spend' in col.lower()]

            if not cost_columns:
                logger.warning("No spend columns found. Cannot optimize budget.")
                return {}

        # Get current budget if total_budget not provided
        if total_budget is None:
            total_budget = sum(self.preprocessed_data[cost_columns].sum())

        # Prepare elasticities for optimization
        valid_channels = []
        valid_elasticities = {}

        for col in cost_columns:
            # Find corresponding elasticity (might be for adstocked version)
            elasticity_key = col
            if f"{col}_adstocked" in self.elasticities:
                elasticity_key = f"{col}_adstocked"

            if elasticity_key in self.elasticities:
                valid_channels.append(col)
                valid_elasticities[col] = self.elasticities[elasticity_key]

        if not valid_channels:
            logger.warning("No valid channels with elasticities found. Cannot optimize budget.")
            return {}

        # Use simple budget allocation
        optimized_budget = simple_budget_allocation(valid_elasticities, total_budget)

        # Log the optimized budget
        logger.info("Optimized budget allocation:")
        for channel, budget in optimized_budget.items():
            logger.info(f"  {channel}: {budget:.2f} ({budget / total_budget * 100:.1f}%)")

        return optimized_budget

    def generate_summary_report(self):
        """
        Generate a summary report of the model and its performance.

        Returns:
            String containing the report
        """
        if self.results is None:
            raise ValueError("No model results. Call fit_model() first.")

        # Create report
        report = []
        report.append("# Marketing Mix Model Summary Report")
        report.append("\n## Model Performance")
        report.append(f"- R-squared: {self.results.rsquared:.4f}")
        report.append(f"- Adjusted R-squared: {self.results.rsquared_adj:.4f}")

        if self.cv_results:
            report.append("\n## Cross-Validation Results")
            report.append(
                f"- Avg RMSE: {np.mean(self.cv_results['rmse']):.4f} (±{np.std(self.cv_results['rmse']):.4f})")
            report.append(f"- Avg R²: {np.mean(self.cv_results['r2']):.4f} (±{np.std(self.cv_results['r2']):.4f})")
            report.append(f"- Avg MAE: {np.mean(self.cv_results['mae']):.4f} (±{np.std(self.cv_results['mae']):.4f})")
            report.append(
                f"- Avg MAPE: {np.mean(self.cv_results['mape']):.2f}% (±{np.std(self.cv_results['mape']):.2f}%)")

        report.append("\n## Model Coefficients")
        coefs = self.results.params.sort_values(ascending=False)
        for name, value in coefs.items():
            report.append(f"- {name}: {value:.6f}")

        if self.elasticities:
            report.append("\n## Media Channel Elasticities")
            sorted_elasticities = sorted(self.elasticities.items(), key=lambda x: abs(x[1]), reverse=True)
            for channel, elasticity in sorted_elasticities:
                report.append(f"- {channel}: {elasticity:.4f}")

        if self.roi_metrics:
            report.append("\n## ROI Metrics")
            for channel, metrics in self.roi_metrics.items():
                report.append(f"\n### {channel}")
                report.append(f"- Coefficient: {metrics['coefficient']:.4f}")
                report.append(f"- Total Spend: {metrics['total_spend']:.2f}")
                report.append(f"- Total Effect: {metrics['total_effect']:.2f}")
                report.append(f"- ROI: {metrics['roi']:.4f}")
                report.append(f"- ROAS: {metrics['roas']:.4f}")
                report.append(f"- CPA: {metrics['cpa']:.4f}")

        return "\n".join(report)

    def save_model(self, filepath):
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model
        """
        try:
            if self.results is None:
                raise ValueError("No model results to save. Call fit_model() first.")

            # Create a dictionary with all the necessary components
            model_data = {
                'model_results': self.results,
                'feature_names': self.feature_names,
                'target': self.target,
                'adstock_transformations': self.adstock_transformations,
                'elasticities': self.elasticities,
                'roi_metrics': self.roi_metrics
            }

            # Save using pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, filepath):
        """
        Load a model from a file.

        Args:
            filepath: Path to the saved model file

        Returns:
            Loaded MarketingMixModel instance
        """
        try:
            # Load using pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            # Create a new instance
            mmm = cls()

            # Restore model components
            mmm.results = model_data['model_results']
            mmm.feature_names = model_data['feature_names']
            mmm.target = model_data['target']
            mmm.adstock_transformations = model_data['adstock_transformations']
            mmm.elasticities = model_data.get('elasticities', {})
            mmm.roi_metrics = model_data.get('roi_metrics', {})

            logger.info(f"Model loaded from {filepath}")

            return mmm

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
