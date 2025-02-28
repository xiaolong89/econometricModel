"""
Budget Optimization Script for Marketing Mix Model
Uses results from improved_mmm.py to optimize marketing budget allocation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from improved_mmm import run_complete_mmm_workflow
from optimization import optimize_budget, extract_current_allocation

# Run the complete MMM workflow to get results
print("Running MMM to get elasticities...")
data_path = 'data/synthetic_advertising_data_v2.csv'  # Update path if needed
results = run_complete_mmm_workflow(data_path)

# Extract key results
model = results['model']
elasticities = results['elasticities']
media_vars = list(elasticities.keys())

# Load data again for current allocation
print("\nLoading data for allocation analysis...")
df = pd.read_csv(data_path)

# Preprocess data to get the same structure used in modeling
from preprocessing import preprocess_data
processed_data = preprocess_data(df)

# Extract current allocation
current_allocation = extract_current_allocation(processed_data, media_vars)
current_budget = sum(current_allocation.values())

print("\nCurrent Budget Allocation:")
for channel, amount in current_allocation.items():
    print(f"{channel}: ${amount:.2f} ({amount/current_budget*100:.1f}%)")

# Run the optimization
print("\nOptimizing budget allocation...")
optimization_results, expected_lift = optimize_budget(elasticities, current_budget, current_allocation)

print("\nOptimization Results:")
print(optimization_results)
print(f"\nExpected Revenue Lift: {expected_lift:.2f}%")

# Visualize current vs optimal allocation
plt.figure(figsize=(12, 6))

# Bar chart for allocation comparison
plt.subplot(1, 2, 1)
plt.bar(optimization_results.index, optimization_results['Current %'],
        color='lightblue', label='Current')
plt.bar(optimization_results.index, optimization_results['Optimal %'],
        color='orange', alpha=0.7, label='Optimal')
plt.title('Budget Allocation Comparison')
plt.ylabel('Percent of Total Budget')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, axis='y')

# Pie chart for optimal allocation
plt.subplot(1, 2, 2)
plt.pie(optimization_results['Optimal %'],
        labels=optimization_results.index,
        autopct='%1.1f%%',
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Optimal Budget Allocation')

plt.tight_layout()
plt.savefig('budget_optimization_result.png')
plt.show()

# Run scenarios with different total budgets
print("\nRunning budget scenarios...")
scenarios = [0.8, 1.0, 1.2]  # 80%, 100%, and 120% of current budget

scenario_results = {}
for multiplier in scenarios:
    scenario_budget = current_budget * multiplier
    scenario_allocation, scenario_lift = optimize_budget(
        elasticities, scenario_budget, current_allocation)
    scenario_results[f"{int(multiplier*100)}%"] = {
        'allocation': scenario_allocation,
        'lift': scenario_lift,
        'budget': scenario_budget
    }

# Print scenario results
print("\nBudget Scenario Results:")
for scenario, result in scenario_results.items():
    print(f"\n{scenario} Budget (${result['budget']:.2f}):")
    print(f"Expected Revenue Lift: {result['lift']:.2f}%")
    print("Channel Allocation:")
    for channel, optimal in zip(result['allocation'].index, result['allocation']['Optimal %']):
        print(f"  {channel}: {optimal:.1f}%")

print("\nAnalysis complete! Budget optimization recommendations ready for review.")