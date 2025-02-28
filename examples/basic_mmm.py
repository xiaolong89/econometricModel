"""
Basic Marketing Mix Model Implementation for mmm_data.csv
This script implements a simple MMM without adstock transformations
to establish a baseline model with the high-quality dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Step 1: Load the Data
    logger.info("Loading mmm_data.csv...")
    data_path = Path('C:\_econometricModel\data\mmm_data.csv')  # Update this path if needed
    df = pd.read_csv(data_path)

    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Step 2: Simple Preprocessing
    # For this basic model, we'll use the raw data without transformations

    # Define variables
    target = 'Sales'
    media_cols = ['TV_Spend', 'Digital_Spend', 'Search_Spend', 'Social_Spend']

    # Create X and y
    X = df[media_cols].copy()
    y = df[target].copy()

    # Step 3: Fit the Model
    logger.info("Fitting basic OLS model...")
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # Step 4: Report Results
    print("\n" + "=" * 50)
    print("BASIC MMM RESULTS")
    print("=" * 50)

    print(f"\nR-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")

    # Print coefficients and significance
    print("\nCoefficients:")
    for var in model.params.index:
        coef = model.params[var]
        p_val = model.pvalues[var]

        # Add stars for significance
        stars = ""
        if p_val < 0.05: stars = "*"
        if p_val < 0.01: stars = "**"
        if p_val < 0.001: stars = "***"

        print(f"{var:15s}: {coef:10.6f} (p={p_val:.4f}) {stars}")

    # Step 5: Calculate Elasticities
    print("\nElasticities:")

    elasticities = {}
    for var in media_cols:
        coef = model.params[var]
        avg_x = X[var].mean()
        avg_y = y.mean()

        # Elasticity formula: coefficient * (mean_x / mean_y)
        elasticity = coef * (avg_x / avg_y)
        elasticities[var] = elasticity

        print(f"{var:15s}: {elasticity:.4f}")

    # Step 6: Visualize Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y, model.fittedvalues, alpha=0.7)

    # Add perfect prediction line
    min_val = min(y.min(), model.fittedvalues.min())
    max_val = max(y.max(), model.fittedvalues.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs. Predicted Sales')
    plt.grid(True, alpha=0.3)

    plot_path = 'actual_vs_predicted.png'
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")

    # Step 7: Print Media ROI
    # Simple ROI calculation: coefficient (revenue per $ spent)
    print("\nMedia ROI (Revenue per $ Spent):")
    for var in media_cols:
        roi = model.params[var]
        print(f"{var:15s}: ${roi:.2f}")

    # Step 8: Simple Budget Allocation
    print("\nRecommended Budget Allocation:")
    total_budget = X[media_cols].sum().sum()

    # Allocate budget proportional to elasticities
    total_elasticity = sum([e for e in elasticities.values() if e > 0])

    for var in media_cols:
        if elasticities[var] > 0:
            allocation = (elasticities[var] / total_elasticity) * 100
            print(f"{var:15s}: {allocation:.2f}%")
        else:
            print(f"{var:15s}: 0.00% (negative or zero elasticity)")

    logger.info("Basic MMM analysis completed successfully")
    print("\nAnalysis completed. You now have a working baseline model.")


if __name__ == "__main__":
    main()
