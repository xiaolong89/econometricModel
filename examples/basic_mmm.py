"""
Basic example of Marketing Mix Model usage.
"""

import logging
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from mmm.core import MarketingMixModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Basic MMM example using synthetic data."""
    try:
        # Initialize the MMM
        mmm = MarketingMixModel()

        # Set paths relative to project structure
        current_dir = Path(__file__).parent.parent
        data_path = str(current_dir / 'data' / 'synthetic_advertising_data_v2.csv')

        # Load and preprocess data
        logger.info("Loading data...")
        mmm.load_data(data_path)

        logger.info("Preprocessing data...")
        mmm.preprocess_data(
            target='sales',
            date_col='week',
            media_cols=[
                'ad_spend_linear_tv',
                'ad_spend_digital',
                'ad_spend_search',
                'ad_spend_social',
                'ad_spend_programmatic'
            ],
            control_cols=['gdp', 'inflation', 'consumer_confidence']
        )

        # Apply adstock transformations
        logger.info("Applying adstock transformations...")
        mmm.apply_adstock_to_all_media()

        # Fit the model
        logger.info("Fitting the MMM model...")
        model_results = mmm.fit_model()
        print("\nModel Summary:")
        print(model_results.summary())

        # Cross-validate the model
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

        # Plot channel contributions
        logger.info("Plotting channel contributions...")
        mmm.plot_channel_contributions()

        # Optimize budget
        logger.info("Optimizing budget allocation...")
        optimized_budget = mmm.optimize_budget()

        # Generate summary report
        logger.info("Generating summary report...")
        report = mmm.generate_summary_report()
        print("\nSummary Report:")
        print(report)

        # Save the model
        model_dir = current_dir
        mmm.save_model(str(model_dir / 'mmm_model.pkl'))

        logger.info("MMM analysis completed successfully!")

        # Keep plots open
        plt.show()

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
