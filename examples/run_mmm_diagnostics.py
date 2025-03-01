#!/usr/bin/env python
"""
Run Marketing Mix Model Diagnostics

This script demonstrates how to use the enhanced model diagnostics capabilities
to validate and compare linear and log-log MMM specifications.

Usage:
  python run_mmm_diagnostics.py [data_path] [--focus diagnostic1,diagnostic2]

Example:
  python run_mmm_diagnostics.py mmm_data.csv --focus residuals,monte_carlo,time_validation
"""

import argparse
import logging
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

# Import diagnostics modules
from mmm.diagnostics_integration import diagnose_mmm_models, generate_comprehensive_analysis_report
from mmm.model_diagnostics import (
    analyze_residuals,
    monte_carlo_simulation,
    model_specification_comparison,
    run_comprehensive_diagnostics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run MMM diagnostics')

    parser.add_argument('data_path', nargs='?', default='C:\_econometricModel\data\mmm_data.csv',
                        help='Path to the data file (default: C:\_econometricModel\data\mmm_data.csv)')

    parser.add_argument('--focus', type=str, default=None,
                        help='Comma-separated list of diagnostics to focus on')

    parser.add_argument('--output', type=str, default='mmm_diagnostics_report.html',
                        help='Output path for HTML report')

    parser.add_argument('--quick', action='store_true',
                        help='Run only essential diagnostics for faster results')

    parser.add_argument('--models', type=str, default='both',
                        choices=['linear', 'log_log', 'both'],
                        help='Which model type(s) to diagnose')

    return parser.parse_args()


def run_mmm_diagnostics(args):
    """Run MMM diagnostics based on command-line arguments"""
    logger.info(f"Starting MMM diagnostics with {args.data_path}")
    start_time = time.time()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse focus diagnostics if provided
    focus = None
    if args.focus:
        focus = [diag.strip() for diag in args.focus.split(',')]
        logger.info(f"Focusing on specific diagnostics: {focus}")

    # If quick mode is enabled, run only essential diagnostics
    if args.quick:
        focus = ['residuals', 'monte_carlo', 'specification_comparison']
        logger.info(f"Quick mode enabled. Running only essential diagnostics: {focus}")

    # Run diagnostics integration function
    try:
        diagnostics_results = diagnose_mmm_models(
            args.data_path,
            base_model_path=os.path.dirname(args.output),
            focus=focus
        )

        # Generate comprehensive report
        report_path = generate_comprehensive_analysis_report(
            diagnostics_results,
            output_path=args.output
        )

        logger.info(f"Diagnostics completed in {time.time() - start_time:.1f} seconds")
        logger.info(f"Report saved to {report_path}")

        # Print main findings to console
        print_main_findings(diagnostics_results, args.output)

        return diagnostics_results, report_path

    except Exception as e:
        logger.error(f"Error running diagnostics: {str(e)}")
        raise


def run_direct_diagnostics(args):
    """Run direct diagnostics on data without using the integration module"""
    logger.info(f"Starting direct diagnostics on {args.data_path}")
    start_time = time.time()

    # Load data
    df = pd.read_csv(args.data_path)

    # Auto-detect target and features
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

    # Run diagnostics based on model type
    results = {}

    if args.models in ['linear', 'both']:
        logger.info("Running diagnostics for linear model...")
        linear_results = run_comprehensive_diagnostics(
            df, target, features, model_type='linear-linear')
        results['linear'] = linear_results

    if args.models in ['log_log', 'both']:
        logger.info("Running diagnostics for log-log model...")
        loglog_results = run_comprehensive_diagnostics(
            df, target, features, model_type='log-log')
        results['log_log'] = loglog_results

    logger.info(f"Direct diagnostics completed in {time.time() - start_time:.1f} seconds")

    # Print key findings
    print_diagnostics_summary(results)

    return results


def print_diagnostics_summary(results):
    """Print a summary of diagnostic findings to the console"""
    print("\n" + "=" * 80)
    print(" MODEL DIAGNOSTICS SUMMARY ")
    print("=" * 80)

    for model_type, model_results in results.items():
        if 'diagnostic_summary' in model_results:
            summary = model_results['diagnostic_summary']
            print(f"\n{model_type.upper()} MODEL:")
            print(f"Overall Confidence: {summary['overall_assessment']['confidence_level']} " +
                  f"({summary['overall_assessment']['confidence_score']}%)")

            # Print issues by category
            for category in ['model_quality', 'stability', 'elasticity_reliability']:
                if category in summary and 'issues' in summary[category]:
                    issues = summary[category]['issues']
                    if issues:
                        print(f"\n{category.replace('_', ' ').title()} Issues:")
                        for issue in issues:
                            print(f"  - {issue}")

            # Print top recommendations
            if any('recommendations' in summary[cat] for cat in
                   ['model_quality', 'stability', 'elasticity_reliability']):
                print("\nTop Recommendations:")
                for category in ['model_quality', 'stability', 'elasticity_reliability']:
                    if category in summary and 'recommendations' in summary[category]:
                        for i, rec in enumerate(summary[category]['recommendations']):
                            if i < 2:  # Limit to top 2 recommendations per category
                                print(f"  - {rec}")

            print("\n" + "-" * 80)


def print_main_findings(diagnostics_results, output_path):
    """Print the main findings from the diagnostics to the console"""
    print("\n" + "=" * 80)
    print(" MARKETING MIX MODEL DIAGNOSTICS - MAIN FINDINGS ")
    print("=" * 80)

    if 'comparison_summary' in diagnostics_results:
        comparison = diagnostics_results['comparison_summary']

        # Print overall recommendation
        if 'recommendation' in comparison:
            print("\nRECOMMENDATION:")
            print(comparison['recommendation'])

        # Print model comparison table
        print("\nMODEL COMPARISON:")
        print(f"{'Dimension':<25} | {'Linear Model':<15} | {'Log-Log Model':<15} | {'Preferred':<10}")
        print("-" * 75)

        for dimension in ['model_quality', 'stability', 'elasticity_reliability']:
            if dimension in comparison:
                dim_data = comparison[dimension]
                dim_name = dimension.replace('_', ' ').title()
                linear_issues = dim_data.get('linear_issues', 'N/A')
                loglog_issues = dim_data.get('loglog_issues', 'N/A')
                preferred = dim_data.get('preferred_model', 'Unknown').title()

                print(f"{dim_name:<25} | {linear_issues:<15} | {loglog_issues:<15} | {preferred:<10}")

        # Print model scores
        if 'model_scores' in comparison:
            linear_score = comparison['model_scores'].get('linear', 0)
            loglog_score = comparison['model_scores'].get('log_log', 0)
            confidence = comparison.get('selection_confidence', 'Unknown').title()

            print("\nOVERALL SCORES:")
            print(f"Linear Model: {linear_score:.2f}")
            print(f"Log-Log Model: {loglog_score:.2f}")
            print(f"Selection Confidence: {confidence}")

    # Print elasticity comparison if available
    elasticity_data = {}

    for model_type in ['linear', 'log_log']:
        if model_type in diagnostics_results:
            model_diag = diagnostics_results[model_type]

            # Try to get elasticities from Monte Carlo simulation
            if 'monte_carlo' in model_diag and 'elasticity_statistics' in model_diag['monte_carlo']:
                for feature, stats in model_diag['monte_carlo']['elasticity_statistics'].items():
                    if feature not in elasticity_data:
                        elasticity_data[feature] = {}
                    elasticity_data[feature][model_type] = stats.get('mean', 'N/A')

    if elasticity_data:
        print("\nELASTICITY COMPARISON:")
        print(f"{'Feature':<20} | {'Linear Model':<15} | {'Log-Log Model':<15} | {'% Difference':<15}")
        print("-" * 75)

        for feature, values in elasticity_data.items():
            linear_val = values.get('linear', 'N/A')
            loglog_val = values.get('log_log', 'N/A')

            if linear_val != 'N/A' and loglog_val != 'N/A':
                pct_diff = (loglog_val - linear_val) / linear_val * 100 if linear_val != 0 else float('inf')
                pct_diff_str = f"{pct_diff:+.1f}%" if pct_diff != float('inf') else "N/A"
            else:
                pct_diff_str = "N/A"

            # Format values as strings
            linear_str = f"{linear_val:.4f}" if linear_val != 'N/A' else 'N/A'
            loglog_str = f"{loglog_val:.4f}" if loglog_val != 'N/A' else 'N/A'

            print(f"{feature:<20} | {linear_str:<15} | {loglog_str:<15} | {pct_diff_str:<15}")

    print("\n" + "=" * 80)
    print(f"For complete details, view the HTML report at: {os.path.abspath(output_path)}")
    print("=" * 80 + "\n")


def main():
    """Main execution function"""
    args = parse_args()

    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return 1

    # Determine which approach to use
    # For most use cases, the integration approach is preferred as it handles both models
    # But for more direct control, users can select the direct approach
    if args.models == 'both':
        results, report_path = run_mmm_diagnostics(args)
    else:
        # Run direct diagnostics when focusing on a specific model type
        results = run_direct_diagnostics(args)
        report_path = None

    if report_path:
        logger.info(f"Diagnostics completed. Report saved to {report_path}")

        # Try to open the report in a browser if possible
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(report_path))
        except:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
