import pandas as pd
import logging
import sys  # Import sys for command-line arguments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your MMM modules
from mmm.core import MarketingMixModel

def generate_mmm_results_data(mmm):
    """Generates the mmm_results data in the desired format from the MMM object."""
    results_data = []

    # Model Performance
    results_data.append({
        'Result Type': 'Model Performance',
        'Metric': 'R-squared',
        'Channel': '',
        'Value': mmm.results.rsquared if mmm.results else None,
        'Units': '',
        'Description': 'R-squared value of the model'
    })
    results_data.append({
        'Result Type': 'Model Performance',
        'Metric': 'Adjusted R-squared',
        'Channel': '',
        'Value': mmm.results.rsquared_adj if mmm.results else None,
        'Units': '',
        'Description': 'Adjusted R-squared value of the model'
    })

    # Elasticities
    elasticities = mmm.elasticities
    for channel, elasticity in elasticities.items():
        results_data.append({
            'Result Type': 'Elasticity',
            'Metric': 'Elasticity',
            'Channel': channel,
            'Value': elasticity,
            'Units': '',
            'Description': f'Elasticity for {channel}'
        })

    # ROI Metrics
    roi_metrics = mmm.roi_metrics
    for channel, roi_metrics in roi_metrics.items():
        results_data.append({
            'Result Type': 'ROI',
            'Metric': 'Coefficient',
            'Channel': channel,
            'Value': roi_metrics['coefficient'],
            'Units': '',
            'Description': f'Coefficient for {channel}'
        })
        results_data.append({
            'Result Type': 'ROI',
            'Metric': 'Total Spend',
            'Channel': channel,
            'Value': roi_metrics['total_spend'],
            'Units': '$',
            'Description': f'Total spend for {channel}'
        })
        results_data.append({
            'Result Type': 'ROI',
            'Metric': 'Total Effect',
            'Channel': channel,
            'Value': roi_metrics['total_effect'],
            'Units': '',
            'Description': f'Total effect for {channel}'
        })
        results_data.append({
            'Result Type': 'ROI',
            'Metric': 'ROI',
            'Channel': channel,
            'Value': roi_metrics['roi'],
            'Units': '',
            'Description': f'ROI for {channel}'
        })
        results_data.append({
            'Result Type': 'ROI',
            'Metric': 'ROAS',
            'Channel': channel,
            'Value': roi_metrics['roas'],
            'Units': '',
            'Description': f'ROAS for {channel}'
        })
        results_data.append({
            'Result Type': 'ROI',
            'Metric': 'CPA',
            'Channel': channel,
            'Value': roi_metrics['cpa'],
            'Units': '$',
            'Description': f'CPA for {channel}'
        })

    return pd.DataFrame(results_data)

def run_mmm(csv_file_path, output_file_path="mmm_results.csv"):
    """
    Runs the Marketing Mix Model on the given CSV file and saves the results to a CSV.

    Args:
        csv_file_path: Path to the client's CSV file.
        output_file_path: Path to save the results CSV file.
    """

    try:
        # 1. Load the data
        logging.info(f"Loading data from {csv_file_path}...")
        mmm = MarketingMixModel()  # Create an instance of the MMM class
        data = mmm.load_data(csv_file_path)

        # 2. Preprocess the data
        logging.info("Preprocessing the data...")
        preprocessed_data = mmm.preprocess_data()

        # 3. Apply adstock transformations
        logging.info("Applying adstock transformations...")
        adstocked_data = mmm.apply_adstock_to_all_media()

        # 4. Fit the model
        logging.info("Fitting the model...")
        model_results = mmm.fit_model()

        # 5. Calculate elasticities
        logging.info("Calculating elasticities...")
        elasticities = mmm.calculate_elasticities()

        # 6. Calculate ROI
        logging.info("Calculating ROI...")
        roi = mmm.calculate_roi()

        # Store elasticities and roi_metrics into the MMM object
        mmm.elasticities = elasticities
        mmm.roi_metrics = roi

        # 7. Generate MMM Results Data
        logging.info("Generating MMM Results Data...")
        mmm_results_df = generate_mmm_results_data(mmm)

        # 8. Save MMM Results to CSV
        logging.info(f"Saving MMM Results to {output_file_path}...")
        mmm_results_df.to_csv(output_file_path, index=False)

        logging.info("MMM process completed successfully.")

        return mmm  # Return the MMM object for further analysis

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Get the CSV file path from the command line arguments
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
    else:
        print("Please provide the input CSV file path as a command line argument.")
        sys.exit(1)

    # Get the output file path (optional)
    output_file_path = "mmm_results.csv"  # Default output file path
    if len(sys.argv) > 2:
        output_file_path = sys.argv[2]

    # Run the MMM
    run_mmm(csv_file_path, output_file_path)
