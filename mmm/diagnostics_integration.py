"""
Integration of model diagnostics with the existing Marketing Mix Model framework.

This module provides a bridge between the comprehensive diagnostics module and
the existing MMM implementation, making it easy to diagnose both linear and
log-log models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

# Import both MMM implementations
from examples.basic_mmm import run_log_log_model
from examples.improved_mmm import run_complete_mmm_workflow

# Import diagnostics module
from mmm.model_diagnostics import (
    run_comprehensive_diagnostics,
    generate_diagnostic_report,
    analyze_residuals,
    plot_coefficient_stability,
    rolling_window_analysis,
    leave_one_out_analysis,
    jackknife_resampling,
    monte_carlo_simulation,
    model_specification_comparison,
    outlier_impact_analysis,
    time_based_validation,
    prediction_confidence_intervals
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def diagnose_mmm_models(data_path, base_model_path=None, focus=None):
    """
    Run diagnostics on both linear and log-log MMM models.

    Args:
        data_path: Path to the data file
        base_model_path: Path to save base models (if None, don't save)
        focus: List of specific diagnostics to focus on (if None, run all)

    Returns:
        Dictionary with diagnostic results for both models
    """
    logger.info(f"Starting MMM diagnostics for {data_path}")

    # Load data
    df = pd.read_csv(data_path)

    # Determine features and target
    target = None
    features = []

    # Auto-detect target and features based on common naming patterns
    for col in df.columns:
        if col.lower() in ['sales', 'revenue', 'conversions', 'units']:
            target = col
        elif any(kw in col.lower() for kw in ['spend', 'cost', 'budget', 'tv', 'digital', 'search', 'social']):
            features.append(col)

    # If target not found, use last column as target by convention
    if target is None:
        target = df.columns[-1]
        logger.warning(f"Target column not detected. Using {target} as target.")

    # If features not found, use all columns except target
    if not features:
        features = [col for col in df.columns if col != target]
        logger.warning(f"Feature columns not detected. Using all non-target columns as features.")

    logger.info(f"Detected target: {target}")
    logger.info(f"Detected features: {features}")

    # Run both model types
    try:
        # Run log-log model first
        logger.info("Running log-log model...")
        loglog_results = run_log_log_model(data_path)

        # Run linear model
        logger.info("Running linear model (via run_complete_mmm_workflow)...")
        linear_results = run_complete_mmm_workflow(data_path)
    except Exception as e:
        logger.error(f"Error running base models: {str(e)}")
        loglog_results = None
        linear_results = None

    # Save base models if requested
    if base_model_path is not None:
        try:
            if not os.path.exists(base_model_path):
                os.makedirs(base_model_path)

            # Save diagnostic plots
            if loglog_results is not None:
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(loglog_results['log_log']['predictions'])),
                         loglog_results['log_log']['predictions'], 'g-', label='Log-Log Model')
                plt.plot(range(len(loglog_results['baseline']['predictions'])),
                         loglog_results['baseline']['predictions'], 'r-', label='Linear Model')
                plt.title('Model Comparison')
                plt.legend()
                plt.savefig(os.path.join(base_model_path, 'model_comparison.png'))
        except Exception as e:
            logger.warning(f"Error saving base models: {str(e)}")

    # Extract models for diagnostics
    models_for_diagnostics = {}

    # Extract log-log model
    if loglog_results is not None and 'log_log' in loglog_results and 'model' in loglog_results['log_log']:
        models_for_diagnostics['log_log'] = {
            'model': loglog_results['log_log']['model'],
            'elasticities': loglog_results['log_log']['elasticities'],
            'predictions': loglog_results['log_log']['predictions'],
            'metrics': loglog_results['log_log']['metrics']
        }

    # Extract linear model
    if linear_results is not None and 'model' in linear_results:
        models_for_diagnostics['linear'] = {
            'model': linear_results['model'],
            'elasticities': linear_results.get('elasticities', {}),
            'validation': linear_results.get('validation', {})
        }

    # Run diagnostics
    diagnostics_results = {}

    for model_type, model_info in models_for_diagnostics.items():
        logger.info(f"Running diagnostics for {model_type} model...")

        # Determine appropriate diagnostics based on focus
        if focus is not None:
            diagnostics_to_run = focus
        else:
            # Run all diagnostics by default
            diagnostics_to_run = [
                'residuals',
                'coefficient_stability',
                'rolling_window',
                'leave_one_out',
                'jackknife',
                'monte_carlo',
                'specification_comparison',
                'outliers',
                'time_validation',
                'prediction_intervals'
            ]

        # Initialize results for this model
        model_diagnostics = {}

        # Prepare data based on model type
        if model_type == 'log_log':
            # Transform data for log-log model
            df_transformed = df.copy()
            df_transformed[f"{target}_log"] = np.log1p(df[target])
            target_transformed = f"{target}_log"

            transformed_features = []
            for feature in features:
                df_transformed[f"{feature}_log"] = np.log1p(df[feature])
                transformed_features.append(f"{feature}_log")

            model = model_info['model']
        else:
            # Use original data for linear model
            df_transformed = df
            target_transformed = target
            transformed_features = features
            model = model_info['model']

        # Get X and y for diagnostics
        X = pd.DataFrame()
        for feature in transformed_features:
            if feature in df_transformed.columns:
                X[feature] = df_transformed[feature]

        # Add constant if needed
        if 'const' in model.params.index and 'const' not in X.columns:
            X = pd.DataFrame(sm.add_constant(X))

        y = df_transformed[target_transformed]

        # Run selected diagnostics
        if 'residuals' in diagnostics_to_run:
            try:
                model_diagnostics['residual_analysis'] = analyze_residuals(model, X, y, plot=True)
            except Exception as e:
                logger.warning(f"Error in residual analysis for {model_type} model: {str(e)}")

        if 'coefficient_stability' in diagnostics_to_run:
            try:
                coef_fig = plot_coefficient_stability(model, X, y)
                plt.savefig(f'{model_type}_coefficient_stability.png')
                model_diagnostics['coefficient_stability'] = {'plot': f'{model_type}_coefficient_stability.png'}
            except Exception as e:
                logger.warning(f"Error in coefficient stability for {model_type} model: {str(e)}")

        if 'rolling_window' in diagnostics_to_run:
            try:
                rw_diagnostics = rolling_window_analysis(
                    df_transformed, target_transformed, transformed_features,
                    model_type=model_type)
                model_diagnostics['rolling_window'] = rw_diagnostics
            except Exception as e:
                logger.warning(f"Error in rolling window analysis for {model_type} model: {str(e)}")

        if 'leave_one_out' in diagnostics_to_run:
            try:
                loo_diagnostics = leave_one_out_analysis(
                    df_transformed, target_transformed, transformed_features,
                    model_type=model_type)
                model_diagnostics['leave_one_out'] = loo_diagnostics
            except Exception as e:
                logger.warning(f"Error in leave-one-out analysis for {model_type} model: {str(e)}")

        if 'jackknife' in diagnostics_to_run:
            try:
                jack_diagnostics = jackknife_resampling(
                    df_transformed, target_transformed, transformed_features,
                    model_type=model_type)
                model_diagnostics['jackknife'] = jack_diagnostics
            except Exception as e:
                logger.warning(f"Error in jackknife analysis for {model_type} model: {str(e)}")

        if 'monte_carlo' in diagnostics_to_run:
            try:
                mc_diagnostics = monte_carlo_simulation(
                    df_transformed, target_transformed, transformed_features,
                    model_type=model_type)
                model_diagnostics['monte_carlo'] = mc_diagnostics
            except Exception as e:
                logger.warning(f"Error in Monte Carlo simulation for {model_type} model: {str(e)}")

        if 'specification_comparison' in diagnostics_to_run:
            try:
                spec_diagnostics = model_specification_comparison(
                    df, target, features)
                model_diagnostics['specification_comparison'] = spec_diagnostics
            except Exception as e:
                logger.warning(f"Error in model specification comparison for {model_type} model: {str(e)}")

        if 'outliers' in diagnostics_to_run:
            try:
                outlier_diagnostics = outlier_impact_analysis(
                    df, target, features, model_type=model_type)
                model_diagnostics['outliers'] = outlier_diagnostics
            except Exception as e:
                logger.warning(f"Error in outlier impact analysis for {model_type} model: {str(e)}")

        if 'time_validation' in diagnostics_to_run:
            try:
                time_col = None
                for col in df.columns:
                    if col.lower() in ['date', 'time', 'week', 'month', 'year']:
                        time_col = col
                        break

                if time_col:
                    time_diagnostics = time_based_validation(
                        df, target, features, time_col=time_col,
                        model_type=model_type)
                else:
                    time_diagnostics = time_based_validation(
                        df, target, features, model_type=model_type)

                model_diagnostics['time_validation'] = time_diagnostics
            except Exception as e:
                logger.warning(f"Error in time-based validation for {model_type} model: {str(e)}")

        if 'prediction_intervals' in diagnostics_to_run:
            try:
                pi_diagnostics = prediction_confidence_intervals(
                    model, X, y, model_type=model_type)
                model_diagnostics['prediction_intervals'] = pi_diagnostics
            except Exception as e:
                logger.warning(f"Error in prediction intervals for {model_type} model: {str(e)}")

        # Run comprehensive diagnostics if all diagnostics are selected
        if len(diagnostics_to_run) == 10:  # All diagnostics are selected
            try:
                comprehensive_results = run_comprehensive_diagnostics(
                    df, target, features, model_type=model_type)

                # Generate report
                report_path = generate_diagnostic_report(comprehensive_results, model_type)
                model_diagnostics['report_path'] = report_path
            except Exception as e:
                logger.warning(f"Error in comprehensive diagnostics for {model_type} model: {str(e)}")

        # Store results for this model
        diagnostics_results[model_type] = model_diagnostics

    # Create comparison summary
    logger.info("Creating model comparison summary...")
    comparison_summary = create_model_comparison_summary(diagnostics_results)
    diagnostics_results['comparison_summary'] = comparison_summary

    logger.info("Diagnostics completed")
    return diagnostics_results


def create_model_comparison_summary(diagnostics_results):
    """
    Create a summary comparing linear and log-log model diagnostics.

    Args:
        diagnostics_results: Dictionary with diagnostics for both models

    Returns:
        Dictionary with comparative summary
    """
    summary = {
        'model_quality': {},
        'stability': {},
        'elasticity_reliability': {},
        'recommendation': ''
    }

    # Check if we have both model types
    model_types = list(diagnostics_results.keys())
    if len(model_types) < 2 or 'linear' not in model_types or 'log_log' not in model_types:
        summary[
            'recommendation'] = "Insufficient data to compare models. Run diagnostics on both linear and log-log models."
        return summary

    # Count issues in each model
    linear_issues = {
        'model_quality': 0,
        'stability': 0,
        'elasticity_reliability': 0
    }

    loglog_issues = {
        'model_quality': 0,
        'stability': 0,
        'elasticity_reliability': 0
    }

    # Check residual analysis
    for model_type in ['linear', 'log_log']:
        if 'residual_analysis' in diagnostics_results[model_type]:
            res_analysis = diagnostics_results[model_type]['residual_analysis']
            if 'summary' in res_analysis and 'residual_issues' in res_analysis['summary']:
                if model_type == 'linear':
                    linear_issues['model_quality'] += len(res_analysis['summary']['residual_issues'])
                else:
                    loglog_issues['model_quality'] += len(res_analysis['summary']['residual_issues'])

    # Check stability metrics
    for model_type in ['linear', 'log_log']:
        # From jackknife analysis
        if 'jackknife' in diagnostics_results[model_type] and 'coefficient_statistics' in \
                diagnostics_results[model_type]['jackknife']:
            jack_stats = diagnostics_results[model_type]['jackknife']['coefficient_statistics']
            stability_issues = 0

            for feature, stats in jack_stats.items():
                if 'cv' in stats and stats['cv'] is not None and stats['cv'] > 0.5:  # >50% CV indicates instability
                    stability_issues += 1

            if model_type == 'linear':
                linear_issues['stability'] += stability_issues
            else:
                loglog_issues['stability'] += stability_issues

        # From specification comparison (elasticity consistency)
        if 'specification_comparison' in diagnostics_results[model_type] and 'elasticity_statistics' in \
                diagnostics_results[model_type]['specification_comparison']:
            spec_stats = diagnostics_results[model_type]['specification_comparison']['elasticity_statistics']
            elasticity_issues = 0

            for feature, stats in spec_stats.items():
                if 'cv' in stats and stats['cv'] is not None and stats['cv'] > 0.3:  # >30% CV indicates inconsistency
                    elasticity_issues += 1

            if model_type == 'linear':
                linear_issues['elasticity_reliability'] += elasticity_issues
            else:
                loglog_issues['elasticity_reliability'] += elasticity_issues

    # Compare models based on issues count
    for dimension in ['model_quality', 'stability', 'elasticity_reliability']:
        if linear_issues[dimension] < loglog_issues[dimension]:
            summary[dimension] = {
                'preferred_model': 'linear',
                'linear_issues': linear_issues[dimension],
                'loglog_issues': loglog_issues[dimension]
            }
        else:
            summary[dimension] = {
                'preferred_model': 'log_log',
                'linear_issues': linear_issues[dimension],
                'loglog_issues': loglog_issues[dimension]
            }

    # Make overall recommendation
    dimension_weights = {
        'model_quality': 0.3,
        'stability': 0.3,
        'elasticity_reliability': 0.4
    }

    linear_score = sum(
        dimension_weights[dim] * (1 - (linear_issues[dim] / (linear_issues[dim] + loglog_issues[dim] + 0.0001)))
        for dim in dimension_weights)
    loglog_score = sum(
        dimension_weights[dim] * (1 - (loglog_issues[dim] / (linear_issues[dim] + loglog_issues[dim] + 0.0001)))
        for dim in dimension_weights)

    if linear_score > loglog_score:
        summary[
            'recommendation'] = "Based on diagnostics, the linear model performs better overall. However, if calculating elasticities is the primary goal, consider using the log-log model which typically provides more stable elasticity estimates."
    else:
        summary[
            'recommendation'] = "Based on diagnostics, the log-log model performs better overall. It typically provides more stable elasticity estimates which is beneficial for marketing mix modeling."

    # Add model selection confidence
    score_diff = abs(linear_score - loglog_score)
    if score_diff > 0.2:
        confidence = "high"
    elif score_diff > 0.1:
        confidence = "moderate"
    else:
        confidence = "low"

    summary['selection_confidence'] = confidence
    summary['model_scores'] = {
        'linear': linear_score,
        'log_log': loglog_score
    }

    return summary


def generate_comprehensive_analysis_report(diagnostics_results, output_path="mmm_diagnostics_report.html"):
    """
    Generate a comprehensive HTML report from the diagnostic results.

    Args:
        diagnostics_results: Dictionary with diagnostics for both models
        output_path: Path to save the HTML report

    Returns:
        Path to the generated report
    """
    # Create comprehensive HTML report
    with open(output_path, 'w') as f:
        f.write("""
        <html>
        <head>
            <title>MMM Diagnostics Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                h1, h2, h3 { color: #333; }
                .section { margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
                .issue { color: #d9534f; }
                .recommendation { color: #5cb85c; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; }
                th { background-color: #f2f2f2; }
                .plot-container { margin: 20px 0; }
                .model-comparison { display: flex; justify-content: space-between; }
                .model-card { width: 48%; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .metrics-table { margin-top: 20px; }
                .better { background-color: #dff0d8; }
                .worse { background-color: #f2dede; }
            </style>
        </head>
        <body>
        """)

        # Header
        f.write("<h1>Marketing Mix Model Diagnostics Report</h1>")

        # Model Comparison Summary
        if 'comparison_summary' in diagnostics_results:
            comparison = diagnostics_results['comparison_summary']

            f.write("<div class='section'>")
            f.write("<h2>Model Comparison Summary</h2>")

            # Create table with dimension comparison
            f.write("<table class='metrics-table'>")
            f.write(
                "<tr><th>Dimension</th><th>Linear Model Issues</th><th>Log-Log Model Issues</th><th>Preferred Model</th></tr>")

            for dimension in ['model_quality', 'stability', 'elasticity_reliability']:
                if dimension in comparison:
                    dim_data = comparison[dimension]
                    preferred = dim_data.get('preferred_model', 'Unknown')

                    f.write("<tr>")
                    f.write(f"<td>{dimension.replace('_', ' ').title()}</td>")

                    # Linear model issues
                    linear_class = 'better' if preferred == 'linear' else 'worse'
                    f.write(f"<td class='{linear_class}'>{dim_data.get('linear_issues', 'N/A')}</td>")

                    # Log-log model issues
                    loglog_class = 'better' if preferred == 'log_log' else 'worse'
                    f.write(f"<td class='{loglog_class}'>{dim_data.get('loglog_issues', 'N/A')}</td>")

                    # Preferred model
                    f.write(f"<td>{preferred.title()}</td>")

                    f.write("</tr>")

            f.write("</table>")

            # Add model scores
            if 'model_scores' in comparison:
                f.write("<h3>Overall Model Scores</h3>")
                linear_score = comparison['model_scores'].get('linear', 0)
                loglog_score = comparison['model_scores'].get('log_log', 0)

                f.write("<table>")
                f.write("<tr><th>Linear Model</th><th>Log-Log Model</th><th>Selection Confidence</th></tr>")
                f.write(
                    f"<tr><td>{linear_score:.2f}</td><td>{loglog_score:.2f}</td><td>{comparison.get('selection_confidence', 'Unknown').title()}</td></tr>")
                f.write("</table>")

            # Add recommendation
            if 'recommendation' in comparison:
                f.write("<h3>Recommendation</h3>")
                f.write(f"<p>{comparison['recommendation']}</p>")

            f.write("</div>")

        # Model Cards
        f.write("<div class='section'>")
        f.write("<h2>Model Details</h2>")
        f.write("<div class='model-comparison'>")

        # Loop through model types
        for model_type in ['linear', 'log_log']:
            if model_type in diagnostics_results:
                model_diagnostics = diagnostics_results[model_type]

                f.write(f"<div class='model-card'>")
                f.write(f"<h3>{model_type.title()} Model</h3>")

                # Add residual analysis summary
                if 'residual_analysis' in model_diagnostics and 'summary' in model_diagnostics['residual_analysis']:
                    res_summary = model_diagnostics['residual_analysis']['summary']

                    f.write("<h4>Residual Analysis</h4>")
                    f.write(f"<p>Mean Residual: {res_summary.get('mean_residual', 'N/A')}</p>")

                    if 'residual_issues' in res_summary and res_summary['residual_issues']:
                        f.write("<p>Issues:</p><ul>")
                        for issue in res_summary['residual_issues']:
                            f.write(f"<li class='issue'>{issue}</li>")
                        f.write("</ul>")
                    else:
                        f.write("<p>No residual issues detected.</p>")

                # Add elasticity stability
                if 'monte_carlo' in model_diagnostics and 'elasticity_statistics' in model_diagnostics['monte_carlo']:
                    mc_stats = model_diagnostics['monte_carlo']['elasticity_statistics']

                    f.write("<h4>Elasticity Stability</h4>")
                    f.write("<table>")
                    f.write("<tr><th>Feature</th><th>Mean</th><th>CV (%)</th></tr>")

                    for feature, stats in mc_stats.items():
                        if 'cv' in stats and stats['cv'] is not None:
                            cv_class = 'worse' if stats['cv'] > 0.3 else 'better'
                            f.write(
                                f"<tr><td>{feature}</td><td>{stats.get('mean', 'N/A'):.4f}</td><td class='{cv_class}'>{stats.get('cv', 'N/A') * 100:.1f}%</td></tr>")

                    f.write("</table>")

                # Add key diagnostic plots
                if 'residual_analysis' in model_diagnostics and 'plots' in model_diagnostics['residual_analysis']:
                    plots = model_diagnostics['residual_analysis']['plots']

                    f.write("<h4>Key Diagnostic Plots</h4>")
                    f.write("<div class='plot-container'>")

                    if 'residual_analysis' in plots:
                        f.write(
                            f"<img src='{plots['residual_analysis']}' alt='Residual Analysis' style='max-width:100%; margin-bottom:15px;'>")

                    f.write("</div>")

                f.write("</div>")

        f.write("</div>")  # Close model-comparison div
        f.write("</div>")  # Close section div

        # Detailed Diagnostics
        f.write("<div class='section'>")
        f.write("<h2>Detailed Diagnostics</h2>")

        # Provide links to individual diagnostic reports if available
        links_available = False

        for model_type in ['linear', 'log_log']:
            if model_type in diagnostics_results and 'report_path' in diagnostics_results[model_type]:
                links_available = True
                report_path = diagnostics_results[model_type]['report_path']
                f.write(
                    f"<p><a href='{report_path}' target='_blank'>{model_type.title()} Model Detailed Report</a></p>")

        if not links_available:
            f.write(
                "<p>Detailed individual reports not available. Run with full diagnostics to generate detailed reports.</p>")

        f.write("</div>")

        # Close HTML
        f.write("</body></html>")

    return output_path


if __name__ == "__main__":
    import sys

    # Check if data path provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "mmm_data.csv"  # Default data path

    # Run diagnostics
    diagnostics_results = diagnose_mmm_models(data_path)

    # Generate report
    report_path = generate_comprehensive_analysis_report(diagnostics_results)

    print(f"Diagnostics completed. Report saved to {report_path}")
