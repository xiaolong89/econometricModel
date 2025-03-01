"""
Complete Marketing Mix Modeling workflow runner.
This script ties together modeling and optimization steps.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Import from your implementation files
from combined_mmm import analyze_mmm_with_log_and_adstock
from budget_optimization import create_budget_optimization_report

# Copy the run_complete_mmm_workflow function from my 'Complete MMM Workflow Implementation' artifact

if __name__ == "__main__":
    # Path to your data file
    data_path = 'mmm_data.csv'

    # Run the workflow
    results = run_complete_mmm_workflow(data_path)
    print("\nWorkflow completed successfully!")

    # Keep figures open
    plt.show()
