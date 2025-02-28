"""
Main entry point for the Marketing Mix Model package.
"""

import logging
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

from mmm.core import MarketingMixModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Marketing Mix Model Analysis')

    parser.add_argument('--data', type=str, default=None,
                        help='Path to the data file')
    parser.add_argument('--target', type=str, default='sales',
                        help='Target variable column')
    parser.add_argument('--date', type=str, default='date',
                        help='Date column')
    parser.add_argument('--adstock', action='store_true',
                        help='Apply adstock transformations')
    parser.add_argument('--cv', action='store_true',
                        help='Perform cross-validation')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform budget optimization')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save the model')
    parser.add_argument('--report', type=str, default=None,
                        help='Path to save the report')

    return parser.parse_args()


def main():
    """Main function to run the Marketing Mix Model analysis."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Initialize the MMM
        mmm = MarketingMixModel()

        # Set default data path if not provided
        if args.data is None:
            current_dir = Path.cwd()
            data_dir = current_dir / 'data'
            args.data = str(data_dir / 'synthetic_advertising_data_v2.csv')

        # Load and preprocess data
        logger.info("Loading data...")
        mmm.load_data(args.data)

        logger.info("Preprocessing data...")
        mmm.preprocess_data(
            target=args.target,
            date_col=args.date
        )

        # Apply adstock transformations if requested
        if args.adstock:
            logger.info("Applying adstock transformations...")
            mmm.apply_adstock_to_all_media()

        # Fit the model
        logger.info("Fitting the MMM model...")
        model_results = mmm.fit_model()
        print("\nModel Summary:")
        print(model_results.summary())

        # Cross-validate if requested
        if args.cv:
            logger.info("Performing cross-validation...")
            cv_results = mmm.cross_validate()
            mmm.plot_cv_results()

        # Calculate elasticities
        logger.info("Calculating elasticities...")
        elasticities = mmm.calculate_elasticities()
        mmm.plot_elasticities()

        # Calculate ROI
        logger.info("Calculating ROI metrics...")
        roi_metrics = mmm.calculate_roi()

        # Optimize budget if requested
        if args.optimize:
            logger.info("Optimizing budget allocation...")
            optimized_budget = mmm.optimize_budget()

        # Generate summary report
        logger.info("Generating summary report...")
        report = mmm.generate_summary_report()
        print("\nSummary Report:")
        print(report)

        # Save report if requested
        if args.report:
            with open(args.report, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {args.report}")

        # Save model if requested
        if args.save:
            mmm.save_model(args.save)
            logger.info(f"Model saved to {args.save}")

        # Keep plots open
        plt.show()

        logger.info("MMM analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()