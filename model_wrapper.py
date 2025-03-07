"""
Marketing Mix Model Wrapper

This module provides an interface between the Flask API and the MMM implementation.
It handles data loading, model training, results extraction, and budget optimization.
"""

import os
import uuid
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# Import MMM modules
from mmm.core import MarketingMixModel
from mmm.adstock import apply_adstock
from mmm.preprocessing import preprocess_data
from mmm.optimization import optimize_budget_allocation
from mmm.model_diagnostics import run_comprehensive_diagnostics
from examples.basic_mmm import run_log_log_model, calculate_elasticities

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories for storing data and models
DATA_DIR = Path("./data")
MODEL_DIR = Path("./models")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# In-memory storage for simplicity (replace with database in production)
uploaded_files = {}
trained_models = {}


def generate_id(prefix=""):
    """Generate a unique ID with optional prefix"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"


def save_uploaded_file(file_obj, original_filename):
    """
    Save an uploaded file and return file metadata

    Args:
        file_obj: File-like object from request
        original_filename: Original filename from request

    Returns:
        dict: File metadata including ID, path, and preview
    """
    try:
        # Generate a unique file ID and save path
        file_id = generate_id("file")
        file_extension = os.path.splitext(original_filename)[1].lower()

        if file_extension != '.csv':
            raise ValueError("Only CSV files are supported")

        # Define the file path
        file_path = DATA_DIR / f"{file_id}.csv"

        # Save the file
        file_obj.save(file_path)

        # Read the file to extract metadata
        df = pd.read_csv(file_path)

        # Generate preview
        preview = df.head(5).to_dict('records')

        # Store file metadata
        file_metadata = {
            "file_id": file_id,
            "original_filename": original_filename,
            "path": str(file_path),
            "upload_time": datetime.now().isoformat(),
            "columns": df.columns.tolist(),
            "rows": len(df),
            "preview": preview
        }

        # Store in memory
        uploaded_files[file_id] = file_metadata

        return file_metadata

    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise


def load_data(file_id):
    """
    Load data from a previously uploaded file

    Args:
        file_id: ID of the uploaded file

    Returns:
        pandas.DataFrame: Loaded data
    """
    if file_id not in uploaded_files:
        raise ValueError(f"File with ID {file_id} not found")

    file_path = uploaded_files[file_id]["path"]
    return pd.read_csv(file_path)


def preprocess_mmm_data(df, config):
    """
    Preprocess data for MMM training

    Args:
        df: Input DataFrame
        config: Model configuration dictionary

    Returns:
        tuple: (processed_df, features, target)
    """
    # Extract configuration
    target = config.get("target_variable")
    date_col = config.get("date_variable")
    media_vars = config.get("media_variables", [])
    control_vars = config.get("control_variables", [])
    model_type = config.get("model_type", "log-log")

    # Validate required fields
    if not target or not date_col or not media_vars:
        raise ValueError("Missing required configuration: target_variable, date_variable, and media_variables")

    # Convert date column
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

    # Preprocess based on model type
    if model_type == "log-log":
        # Apply log transformation to target and media variables
        df[f"{target}_log"] = np.log1p(df[target])
        target_transformed = f"{target}_log"

        # Transform media variables
        for var in media_vars:
            df[f"{var}_log"] = np.log1p(df[var])

        media_vars_transformed = [f"{var}_log" for var in media_vars]
    else:
        # Linear model - use original variables
        target_transformed = target
        media_vars_transformed = media_vars

    # Apply adstock transformations if parameters provided
    adstock_params = config.get("adstock_params", {})
    if adstock_params:
        for var in media_vars:
            var_to_transform = f"{var}_log" if model_type == "log-log" else var

            if var in adstock_params:
                params = adstock_params[var]
                decay_rate = params.get("decay_rate", 0.7)
                max_lag = params.get("max_lag", 4)

                # Apply adstock transformation
                adstocked_var = f"{var_to_transform}_adstocked"
                df[adstocked_var] = apply_adstock(
                    df[var_to_transform].values,
                    decay_rate=decay_rate,
                    lag_weight=0.3,  # Fixed parameter
                    max_lag=max_lag
                )

                # Replace the original transformed variable with adstocked version in the list
                index = media_vars_transformed.index(var_to_transform)
                media_vars_transformed[index] = adstocked_var

    # Combine features
    features = media_vars_transformed + control_vars

    return df, features, target_transformed, model_type


def train_mmm_model(file_id, config):
    """
    Train a Marketing Mix Model using the specified configuration

    Args:
        file_id: ID of the uploaded file
        config: Model configuration dictionary

    Returns:
        dict: Trained model information
    """
    try:
        # Load data
        df = load_data(file_id)

        # Preprocess data
        processed_df, features, target, model_type = preprocess_mmm_data(df, config)

        # Generate model ID
        model_id = generate_id("model")

        # Train model based on type
        if model_type == "log-log":
            # Use the log-log implementation
            results = run_log_log_model_with_data(processed_df, target, features, df)
            model_results = results["log_log"]
        else:
            # Use the linear implementation (MarketingMixModel class)
            mmm = MarketingMixModel()
            mmm.load_data_from_dataframe(processed_df)
            mmm.preprocess_data(target=target, media_cols=features)
            model_results = mmm.fit_model()

            # Calculate elasticities
            elasticities = mmm.calculate_elasticities()

            # Get predictions
            import statsmodels.api as sm
            X = sm.add_constant(processed_df[features])
            predictions = model_results.predict(X)

            model_results = {
                "model": model_results,
                "elasticities": elasticities,
                "predictions": predictions
            }

        # Run diagnostics
        diagnostics = run_comprehensive_diagnostics(
            processed_df, target, features, model_type=model_type
        )

        # Create model information
        model_info = {
            "model_id": model_id,
            "file_id": file_id,
            "config": config,
            "features": features,
            "target": target,
            "model_type": model_type,
            "training_time": datetime.now().isoformat(),
            "results": model_results,
            "diagnostics": diagnostics
        }

        # Save model information to disk
        model_path = MODEL_DIR / f"{model_id}.json"
        with open(model_path, 'w') as f:
            # Convert model_results to serializable format
            serialized_info = {
                "model_id": model_id,
                "file_id": file_id,
                "config": config,
                "features": features,
                "target": target,
                "model_type": model_type,
                "training_time": datetime.now().isoformat(),
                "performance": {
                    "r_squared": float(model_results["model"].rsquared),
                    "adj_r_squared": float(model_results["model"].rsquared_adj),
                    "aic": float(model_results["model"].aic),
                    "bic": float(model_results["model"].bic)
                }
            }
            json.dump(serialized_info, f)

        # Store in memory
        trained_models[model_id] = model_info

        # Return performance metrics
        performance = {
            "r_squared": float(model_results["model"].rsquared),
            "adj_r_squared": float(model_results["model"].rsquared_adj),
            "aic": float(model_results["model"].aic),
            "bic": float(model_results["model"].bic)
        }

        return {
            "model_id": model_id,
            "performance": performance
        }

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise


def run_log_log_model_with_data(df, target, features, original_df):
    """
    Run log-log model with existing DataFrame instead of loading from file

    Args:
        df: Preprocessed DataFrame
        target: Target column name
        features: Feature column names
        original_df: Original DataFrame (without transformations)

    Returns:
        dict: Model results
    """
    # Prepare X and y for log-log model
    import statsmodels.api as sm
    X = sm.add_constant(df[features])
    y = df[target]

    # Fit log-log model
    log_log_model = sm.OLS(y, X).fit()

    # Calculate elasticities
    elasticities = {}
    for feature in features:
        # For log-log model, elasticity is the coefficient
        if feature in log_log_model.params:
            elasticities[feature.replace('_log', '').replace('_adstocked', '')] = log_log_model.params[feature]

    # Make predictions (log scale)
    log_log_predictions_log = log_log_model.predict(X)

    # Back-transform predictions to original scale
    if target.endswith('_log'):
        original_target = target.replace('_log', '')
        log_log_predictions = np.expm1(log_log_predictions_log)

        # Calculate metrics in original scale
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        rmse = np.sqrt(mean_squared_error(original_df[original_target], log_log_predictions))
        mape = mean_absolute_percentage_error(original_df[original_target], log_log_predictions) * 100
    else:
        log_log_predictions = log_log_predictions_log

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        rmse = np.sqrt(mean_squared_error(original_df[target], log_log_predictions))
        mape = mean_absolute_percentage_error(original_df[target], log_log_predictions) * 100

    return {
        "log_log": {
            "model": log_log_model,
            "elasticities": elasticities,
            "predictions": log_log_predictions,
            "metrics": {
                "rmse": rmse,
                "mape": mape
            }
        }
    }


def get_model_results(model_id):
    """
    Get detailed results for a trained model

    Args:
        model_id: ID of the trained model

    Returns:
        dict: Model results including performance, elasticities, etc.
    """
    if model_id not in trained_models:
        raise ValueError(f"Model with ID {model_id} not found")

    model_info = trained_models[model_id]
    model_results = model_info["results"]

    # Extract original data
    file_id = model_info["file_id"]
    df = load_data(file_id)

    # Extract target (removing _log suffix if present)
    target = model_info["target"]
    original_target = target.replace('_log', '')

    # Extract date column from config
    date_col = model_info["config"].get("date_variable")

    # Create actual vs predicted data
    actual_vs_predicted = []

    if date_col in df.columns:
        dates = df[date_col].tolist()
        actual = df[original_target].tolist()

        # Format predictions to match the response structure
        model_type = model_info["model_type"]
        if model_type == "log-log":
            predicted = model_results["predictions"].tolist()
        else:
            # For linear model
            predicted = model_results["predictions"].tolist()

        # Create actual vs predicted list
        for i in range(len(dates)):
            actual_vs_predicted.append({
                "date": dates[i],
                "actual": actual[i],
                "predicted": predicted[i]
            })

    # Extract coefficients
    coefficients = {}
    for feature, value in model_results["model"].params.items():
        if feature != "const":
            # Remove _log and _adstocked suffixes for clarity
            original_feature = feature.replace('_log', '').replace('_adstocked', '')
            coefficients[original_feature] = float(value)

    # Include intercept as a separate item
    if "const" in model_results["model"].params:
        coefficients["intercept"] = float(model_results["model"].params["const"])

    # Format elasticities
    elasticities = {}
    if "elasticities" in model_results:
        for feature, value in model_results["elasticities"].items():
            # Clean feature name
            clean_feature = feature.replace('_log', '').replace('_adstocked', '')
            elasticities[clean_feature] = float(value)

    # Create performance metrics
    performance = {
        "r_squared": float(model_results["model"].rsquared),
        "adj_r_squared": float(model_results["model"].rsquared_adj),
        "aic": float(model_results["model"].aic),
        "bic": float(model_results["model"].bic)
    }

    # Add rmse and mape if available
    if "metrics" in model_results:
        performance["rmse"] = float(model_results["metrics"]["rmse"])
        performance["mape"] = float(model_results["metrics"]["mape"])

    # Create result dictionary
    result = {
        "model_id": model_id,
        "performance": performance,
        "elasticities": elasticities,
        "coefficients": coefficients,
        "actual_vs_predicted": actual_vs_predicted[:100]  # Limit to 100 points for response size
    }

    return result


def get_model_visualizations(model_id, viz_type=None):
    """
    Get visualization data for a trained model

    Args:
        model_id: ID of the trained model
        viz_type: Type of visualization to return (None for all)

    Returns:
        dict: Visualization data
    """
    if model_id not in trained_models:
        raise ValueError(f"Model with ID {model_id} not found")

    model_info = trained_models[model_id]
    model_results = model_info["results"]

    # Extract file and config info
    file_id = model_info["file_id"]
    df = load_data(file_id)
    config = model_info["config"]

    # Extract target and media variables
    target = model_info["target"]
    original_target = target.replace('_log', '')
    date_col = config.get("date_variable")
    media_vars = config.get("media_variables", [])

    # Prepare visualization data
    visualizations = {}

    # 1. Actual vs. Predicted visualization
    if viz_type is None or viz_type == "actual_vs_predicted":
        if date_col in df.columns:
            dates = df[date_col].astype(str).tolist()
            actual = df[original_target].tolist()

            # Get predictions
            model_type = model_info["model_type"]
            if model_type == "log-log":
                predicted = model_results["predictions"].tolist()
            else:
                predicted = model_results["predictions"].tolist()

            visualizations["actual_vs_predicted"] = {
                "x": dates,
                "y_actual": actual,
                "y_predicted": predicted
            }

    # 2. Response Curves visualization
    if viz_type is None or viz_type == "response_curves":
        response_curves = {}

        for var in media_vars:
            # Get current average spend
            current_spend = df[var].mean()

            # Create range of values from 0 to 2x current spend
            spend_range = np.linspace(0, current_spend * 2, 20)

            # Calculate response for each spend level
            response = []

            # Use model to predict response at each spend level
            for spend in spend_range:
                # Create a copy of the average values
                X_copy = df.copy()
                X_copy[var] = spend

                # Preprocess (simplification - in production this should use the exact same preprocessing)
                model_type = model_info["model_type"]
                if model_type == "log-log":
                    X_copy[f"{var}_log"] = np.log1p(X_copy[var])
                    var_to_use = f"{var}_log"

                    # Check if adstocking was applied
                    if f"{var_to_use}_adstocked" in model_info["features"]:
                        var_to_use = f"{var_to_use}_adstocked"

                        # Apply adstock (simplified - should match original preprocessing)
                        if var in config.get("adstock_params", {}):
                            params = config["adstock_params"][var]
                            decay_rate = params.get("decay_rate", 0.7)
                            max_lag = params.get("max_lag", 4)

                            X_copy[var_to_use] = apply_adstock(
                                X_copy[f"{var}_log"].values,
                                decay_rate=decay_rate,
                                lag_weight=0.3,
                                max_lag=max_lag
                            )
                else:
                    var_to_use = var

                    # Check if adstocking was applied
                    if f"{var}_adstocked" in model_info["features"]:
                        var_to_use = f"{var}_adstocked"

                        # Apply adstock (simplified)
                        if var in config.get("adstock_params", {}):
                            params = config["adstock_params"][var]
                            decay_rate = params.get("decay_rate", 0.7)
                            max_lag = params.get("max_lag", 4)

                            X_copy[var_to_use] = apply_adstock(
                                X_copy[var].values,
                                decay_rate=decay_rate,
                                lag_weight=0.3,
                                max_lag=max_lag
                            )

                # Predict with this value
                # (This is simplified - in production, use full preprocessing pipeline)
                import statsmodels.api as sm
                X_pred = sm.add_constant(X_copy[model_info["features"]])

                # Get average prediction
                pred = model_results["model"].predict(X_pred).mean()

                # Transform back to original scale if log model
                if model_type == "log-log" and target.endswith('_log'):
                    pred = np.expm1(pred)

                response.append(float(pred))

            # Store response curve
            response_curves[var] = {
                "spend": spend_range.tolist(),
                "response": response
            }

        visualizations["response_curves"] = response_curves

    # 3. Contributions visualization
    if viz_type is None or viz_type == "contributions":
        # Calculate contributions of each variable
        contributions = {}

        # Get coefficients
        coeffs = model_results["model"].params

        # Calculate contribution for media variables
        labels = []
        values = []

        for var in media_vars:
            # Find corresponding transformed variable
            model_type = model_info["model_type"]
            if model_type == "log-log":
                var_transformed = f"{var}_log"

                # Check if adstocking was applied
                if f"{var_transformed}_adstocked" in model_info["features"]:
                    var_transformed = f"{var_transformed}_adstocked"
            else:
                var_transformed = var

                # Check if adstocking was applied
                if f"{var}_adstocked" in model_info["features"]:
                    var_transformed = f"{var}_adstocked"

            # Calculate contribution if variable is in the model
            if var_transformed in coeffs:
                contrib = float(coeffs[var_transformed] * df[var_transformed].mean())

                # Add to lists
                labels.append(var)
                values.append(contrib)

        # Add control variables if available
        control_vars = config.get("control_variables", [])
        for var in control_vars:
            if var in coeffs:
                contrib = float(coeffs[var] * df[var].mean())

                # Add to lists
                labels.append(var)
                values.append(contrib)

        # Add intercept (baseline)
        if "const" in coeffs:
            labels.append("Baseline")
            values.append(float(coeffs["const"]))

        # Create contributions dictionary
        visualizations["contributions"] = {
            "labels": labels,
            "values": values
        }

    return visualizations


def get_model_diagnostics(model_id):
    """
    Get diagnostic results for a trained model

    Args:
        model_id: ID of the trained model

    Returns:
        dict: Diagnostic results
    """
    if model_id not in trained_models:
        raise ValueError(f"Model with ID {model_id} not found")

    model_info = trained_models[model_id]

    # Extract diagnostics information
    # Note: The full diagnostics results are complex and contain non-serializable objects
    # For the API, we'll extract just the key metrics

    diagnostics = model_info.get("diagnostics", {})

    # Extract residual analysis results
    residual_analysis = {}
    if "residual_analysis" in diagnostics:
        ra = diagnostics["residual_analysis"]

        # Extract normality test
        if "normality_test" in ra:
            residual_analysis["normality_test"] = {
                "statistic": float(ra["normality_test"].get("statistic", 0)),
                "p_value": float(ra["normality_test"].get("p_value", 1)),
                "is_normal": ra["normality_test"].get("is_normal", True)
            }

        # Extract heteroskedasticity test
        if "heteroskedasticity_test" in ra:
            residual_analysis["heteroskedasticity_test"] = {
                "statistic": float(ra["heteroskedasticity_test"].get("lm_statistic", 0)),
                "p_value": float(ra["heteroskedasticity_test"].get("lm_p_value", 1)),
                "has_heteroskedasticity": ra["heteroskedasticity_test"].get("has_heteroskedasticity", False)
            }

        # Extract autocorrelation test
        if "autocorrelation_test" in ra:
            residual_analysis["autocorrelation_test"] = {
                "durbin_watson": float(ra["autocorrelation_test"].get("durbin_watson", 2)),
                "has_autocorrelation": ra["autocorrelation_test"].get("has_autocorrelation", False)
            }

    # Extract stability assessment
    stability_assessment = {}
    if "jackknife_analysis" in diagnostics and "coefficient_statistics" in diagnostics["jackknife_analysis"]:
        coef_stats = diagnostics["jackknife_analysis"]["coefficient_statistics"]

        coefficient_stability = {}
        for feature, stats in coef_stats.items():
            if "cv" in stats:
                cv = float(stats["cv"]) if stats["cv"] is not None else None
                is_stable = cv < 0.3 if cv is not None else None

                # Clean feature name
                clean_feature = feature.replace('_log', '').replace('_adstocked', '')

                coefficient_stability[clean_feature] = {
                    "cv": cv,
                    "is_stable": is_stable
                }

        stability_assessment["coefficient_stability"] = coefficient_stability

    # Extract model quality assessment
    model_quality = {}
    if "diagnostic_summary" in diagnostics and "model_quality" in diagnostics["diagnostic_summary"]:
        mq = diagnostics["diagnostic_summary"]["model_quality"]

        model_quality["issues"] = mq.get("issues", [])
        model_quality["recommendations"] = mq.get("recommendations", [])

    # Combine all diagnostics
    diagnostic_results = {
        "residual_analysis": residual_analysis,
        "stability_assessment": stability_assessment,
        "model_quality": model_quality
    }

    return diagnostic_results


def optimize_budget(model_id, optimization_config):
    """
    Optimize budget allocation based on model results

    Args:
        model_id: ID of the trained model
        optimization_config: Budget optimization configuration

    Returns:
        dict: Optimization results
    """
    if model_id not in trained_models:
        raise ValueError(f"Model with ID {model_id} not found")

    model_info = trained_models[model_id]

    # Extract model results and elasticities
    model_results = model_info["results"]

    # Get elasticities
    if "elasticities" not in model_results:
        raise ValueError("Model does not have elasticities calculated")

    elasticities = model_results["elasticities"]

    # Extract configuration
    file_id = model_info["file_id"]
    df = load_data(file_id)
    media_vars = model_info["config"].get("media_variables", [])

    # Calculate current allocation
    current_allocation = {}
    for var in media_vars:
        current_allocation[var] = float(df[var].mean())

    # Calculate total current budget
    total_current_budget = sum(current_allocation.values())

    # Get optimization parameters
    total_budget = optimization_config.get("total_budget", total_current_budget)
    constraints = optimization_config.get("constraints", {})
    scenarios = optimization_config.get("scenarios", [])

    # Prepare constraints for optimizer
    min_spend = {}
    max_spend = {}

    for var, constraint in constraints.items():
        if "min" in constraint:
            min_spend[var] = constraint["min"]
        else:
            # Default minimum is 50% of current spend
            min_spend[var] = current_allocation.get(var, 0) * 0.5

        if "max" in constraint:
            max_spend[var] = constraint["max"]
        else:
            # Default maximum is 200% of current spend
            max_spend[var] = current_allocation.get(var, 0) * 2.0

    # Run optimization
    try:
        from mmm.optimization import optimize_budget_allocation

        # Clean elasticities to match expected format
        clean_elasticities = {}
        for var in media_vars:
            # Remove suffixes from elasticities
            clean_var = var.replace('_log', '').replace('_adstocked', '')
            if clean_var in elasticities:
                clean_elasticities[var] = elasticities[clean_var]
            else:
                # Default to small elasticity if not found
                clean_elasticities[var] = 0.01

        # Run main optimization
        optimization_results = optimize_budget_allocation(
            clean_elasticities,
            current_spend=current_allocation,
            current_revenue=1000000,  # Placeholder - we'll scale the results
            total_budget=total_budget,
            min_spend=min_spend,
            max_spend=max_spend
        )

        # Extract optimized allocation
        optimized_allocation = optimization_results["optimized_allocation"]
        expected_lift = optimization_results["percent_lift"]
        roi_improvement = optimization_results["roi_improvement"]

        # Run scenario optimizations if requested
        scenario_results = []

        for scenario in scenarios:
            name = scenario.get("name", "Scenario")
            budget_multiplier = scenario.get("budget_multiplier", 1.0)

            # Calculate scenario budget
            scenario_budget = total_current_budget * budget_multiplier

            # Run optimization for this scenario
            scenario_optimization = optimize_budget_allocation(
                clean_elasticities,
                current_spend=current_allocation,
                current_revenue=1000000,
                total_budget=scenario_budget,
                min_spend=min_spend,
                max_spend=max_spend
            )

            # Extract results
            scenario_allocation = scenario_optimization["optimized_allocation"]
            scenario_lift = scenario_optimization["percent_lift"]

            # Add to results
            scenario_results.append({
                "name": name,
                "budget": float(scenario_budget),
                "allocation": {var: float(allocation) for var, allocation in scenario_allocation.items()},
                "expected_lift": float(scenario_lift)
            })

        # Format output
        optimization_output = {
            "current_allocation": {var: float(allocation) for var, allocation in current_allocation.items()},
            "current_total": float(total_current_budget),
            "optimized_allocation": {var: float(allocation) for var, allocation in optimized_allocation.items()},
            "optimized_total": float(total_budget),
            "expected_lift": float(expected_lift),
            "roi_improvement": float(roi_improvement),
            "scenarios": scenario_results
        }

        return optimization_output

    except Exception as e:
        logger.error(f"Error in budget optimization: {str(e)}")
        raise