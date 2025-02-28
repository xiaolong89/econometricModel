import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import minimize
import sys
import os

# Add parent directory to path to allow imports from mmm directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from other modules
from mmm.preprocessing import preprocess_data, apply_adstock
from mmm.modeling import build_mmm, validate_model, apply_constraints
from mmm.optimization import calculate_elasticities, calculate_roi, optimize_budget, extract_current_allocation
from mmm.optimization import simple_budget_allocation

def run_complete_mmm_workflow(data_path):
    """Run complete MMM workflow from data loading to optimization"""
    # 1. Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Original DataFrame columns: {df.columns.tolist()}")

    # 2. Rename columns to match expected format
    print("Renaming columns to match expected format...")
    column_mapping = {
        'ad_spend_linear_tv': 'tv_spend',
        'ad_spend_digital': 'display_spend',
        'ad_spend_search': 'search_spend',
        'ad_spend_social': 'social_spend',
        'ad_spend_programmatic': 'email_spend',  # Using as a substitute for email
        'sales': 'revenue'
    }

    # Only rename columns that exist in the dataset
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    print(f"Renamed columns: {df.columns.tolist()}")

    # 3. Preprocess data
    print("Preprocessing data...")
    processed_data = preprocess_data(df)
    print(f"Processed DataFrame columns: {processed_data.columns.tolist()}")

    # 4. Check if log_revenue exists, create it if not
    if 'log_revenue' not in processed_data.columns and 'revenue' in processed_data.columns:
        print("Creating log_revenue column...")
        processed_data['log_revenue'] = np.log(processed_data['revenue'])

    # Verify log_revenue exists
    print(f"Final columns before modeling: {processed_data.columns.tolist()}")
    if 'log_revenue' not in processed_data.columns:
        print("ERROR: log_revenue column still missing!")
        if 'revenue' in processed_data.columns:
            print("Creating log_revenue column (second attempt)...")
            processed_data['log_revenue'] = np.log(processed_data['revenue'])
        else:
            raise ValueError("No 'revenue' column found in the data. Unable to proceed with modeling.")

    # 5. Build model
    print("Building MMM...")
    media_vars = ['traditional_media', 'digital_paid', 'social_media', 'owned_media']

    # Verify that media variables exist in the processed data
    missing_vars = [var for var in media_vars if var not in processed_data.columns]
    if missing_vars:
        print(f"WARNING: The following media variables are missing: {missing_vars}")
        print("Available columns:", processed_data.columns.tolist())
        # Use only available media variables
        media_vars = [var for var in media_vars if var in processed_data.columns]
        if not media_vars:
            raise ValueError("No media variables found in the processed data. Unable to build model.")

    model, vif_data = build_mmm(processed_data)

    print("\nVIF Values:")
    print(vif_data.sort_values('VIF', ascending=False))

    # 6. Apply constraints if needed
    print("\nChecking for negative coefficients...")
    constrained_model = apply_constraints(model, processed_data, media_vars)

    # 7. Validate model
    print("\nValidating model...")
    validation_results = validate_model(constrained_model, processed_data)

    print(f"\nModel Performance:")
    print(f"R²: {validation_results['r_squared']:.4f}")
    print(f"Adjusted R²: {validation_results['adj_r_squared']:.4f}")
    print(f"MAPE: {validation_results['mape']:.2f}%")

    # 8. Calculate elasticities and ROI
    print("\nCalculating elasticities and ROI...")
    elasticities = calculate_elasticities(constrained_model, processed_data, media_vars)
    roi = calculate_roi(constrained_model, processed_data, media_vars)

    print("\nElasticities:")
    for channel, value in elasticities.items():
        print(f"{channel}: {value:.4f}")

    print("\nROI Estimates:")
    for channel, value in roi.items():
        print(f"{channel}: {value:.2f}")

    # 9. Optimize budget
    print("\nOptimizing budget allocation...")
    current_allocation = extract_current_allocation(processed_data, media_vars)
    current_budget = sum(current_allocation.values())

    # Handle optimization with all-zero elasticities
    if sum(elasticities.values()) == 0:
        print("All elasticities are zero. Using equal allocation.")
        optimal_allocation = {channel: current_budget / len(elasticities) for channel in elasticities}
        optimization_results = pd.DataFrame({
            'Current Allocation': current_allocation,
            'Current %': [spend / current_budget * 100 for spend in current_allocation.values()],
            'Optimal Allocation': optimal_allocation,
            'Optimal %': [spend / current_budget * 100 for spend in optimal_allocation.values()],
            'Change %': [(optimal_allocation[ch] - current_allocation[ch]) / current_allocation[ch] * 100
                         if current_allocation[ch] > 0 else 0 for ch in current_allocation]
        })
        expected_lift = 0
    else:
        # Use simple_budget_allocation instead of optimize_budget
        optimal_allocation = simple_budget_allocation(elasticities, current_budget)

        # Get a consistent order of channels to use as index
        channels = list(current_allocation.keys())

        # Create DataFrame with explicit index
        optimization_results = pd.DataFrame({
            'Current Allocation': [current_allocation[ch] for ch in channels],
            'Current %': [current_allocation[ch] / current_budget * 100 for ch in channels],
            'Optimal Allocation': [optimal_allocation[ch] for ch in channels],
            'Optimal %': [optimal_allocation[ch] / current_budget * 100 for ch in channels],
            'Change %': [(optimal_allocation[ch] - current_allocation[ch]) / current_allocation[ch] * 100
                         if current_allocation[ch] > 0 else 0 for ch in channels]
        }, index=channels)

        # Calculate expected lift
        current_effect = sum(elasticities[ch] * current_allocation[ch] for ch in elasticities)
        optimal_effect = sum(elasticities[ch] * optimal_allocation[ch] for ch in elasticities)
        expected_lift = (optimal_effect - current_effect) / current_effect * 100 if current_effect > 0 else 0

    print("\nBudget Optimization Results:")
    print(optimization_results)
    print(f"\nExpected Revenue Lift: {expected_lift:.2f}%")

    return {
        'model': constrained_model,
        'validation': validation_results,
        'elasticities': elasticities,
        'roi': roi,
        'optimization': optimization_results,
        'expected_lift': expected_lift
    }


# Execute the complete workflow
if __name__ == "__main__":
    try:
        results = run_complete_mmm_workflow('C:\_econometricModel\data\synthetic_advertising_data_v2.csv')
        print("\nMMM workflow completed successfully!")

        # Save a visualization of the model fit
        plt.figure(figsize=(12, 6))
        plt.plot(results['validation']['actual'], label='Actual')
        plt.plot(results['validation']['predictions'], label='Predicted')
        plt.title(
            f'Model Fit - R²: {results["validation"]["r_squared"]:.4f}, MAPE: {results["validation"]["mape"]:.2f}%')
        plt.legend()
        plt.grid(True)
        plt.savefig('model_fit.png')
        plt.close()

        # Create budget allocation visualization
        optimization = results['optimization']
        plt.figure(figsize=(10, 6))
        plt.bar(optimization.index, optimization['Current %'],
                color='lightblue', label='Current')
        plt.bar(optimization.index, optimization['Optimal %'],
                color='orange', alpha=0.7, label='Optimal')
        plt.title('Budget Allocation Comparison')
        plt.ylabel('Percent of Total Budget')
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig('budget_optimization.png')
        plt.close()

        print("Visualizations saved as 'model_fit.png' and 'budget_optimization.png'")

    except Exception as e:
        print(f"Error in MMM workflow: {str(e)}")
        import traceback

        traceback.print_exc()
