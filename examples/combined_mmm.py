"""
Combined MMM implementation with proper diminishing returns and adstock transformations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add the apply_adstock function and analyze_mmm_with_log_and_adstock function here
# (Copy from the 'Combined Approach' artifact I provided)

# Then add a main execution block:
if __name__ == "__main__":
    # Path to your data file
    data_path = 'mmm_data.csv'

    # Load data
    data = pd.read_csv(data_path)
    print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")

    # Run analysis
    results = analyze_mmm_with_log_and_adstock(data)

    # Print the summary of the best model
    best_model_type = max(results['test_metrics'],
                          key=lambda x: results['test_metrics'][x]['r2'])

    print(f"\nBest model based on test R²: {best_model_type}")

    # Keep figures open
    plt.show()
