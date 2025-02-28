"""
Budget optimization functions for Marketing Mix Models.
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def simple_budget_allocation(elasticities, total_budget):
    """
    Simple budget allocation based on elasticities.

    Args:
        elasticities: Dictionary mapping channels to elasticities
        total_budget: Total budget to allocate

    Returns:
        Dictionary mapping channels to allocated budget
    """
    # Filter out negative or zero elasticities
    positive_elasticities = {k: v for k, v in elasticities.items() if v > 0}

    if not positive_elasticities:
        logger.warning("No positive elasticities found. Using equal allocation.")
        equal_budget = total_budget / len(elasticities)
        return {channel: equal_budget for channel in elasticities.keys()}

    # Calculate weights based on elasticities
    total_elasticity = sum(positive_elasticities.values())
    weights = {k: v / total_elasticity for k, v in positive_elasticities.items()}

    # Allocate budget
    allocation = {channel: weight * total_budget for channel, weight in weights.items()}

    # Handle channels with non-positive elasticities
    for channel in elasticities:
        if channel not in allocation:
            allocation[channel] = 0

    return allocation


def optimize_budget_hill_function(coefficients, current_spend, total_budget, diminishing_returns_params,
                                  bound_limits=None):
    """
    Optimize budget using Hill function for diminishing returns.

    Args:
        coefficients: Model coefficients for each channel
        current_spend: Current spending for each channel
        total_budget: Total budget to allocate
        diminishing_returns_params: Dictionary with 'shape' and 'scale' params for each channel
        bound_limits: Dictionary with min/max spend for each channel

    Returns:
        Dictionary with optimized budget allocation
    """
    channels = list(coefficients.keys())

    # Set default bounds if not provided
    if bound_limits is None:
        bound_limits = {}
        for channel in channels:
            # Default: between 50% and 200% of current spend
            bound_limits[channel] = {
                'min': max(0.5 * current_spend[channel], 1000),  # Minimum spend
                'max': 2.0 * current_spend[channel]  # Maximum spend
            }

    # Define Hill function
    def hill_response(spend, shape, scale):
        """Calculate response using Hill function"""
        return spend ** shape / (scale ** shape + spend ** shape)

    # Define objective function (negative because we maximize)
    def objective(x):
        # x is a vector of spend values
        spend_dict = {channel: spend for channel, spend in zip(channels, x)}

        # Calculate total response
        response = 0
        for i, channel in enumerate(channels):
            response += coefficients[channel] * hill_response(
                spend_dict[channel],
                diminishing_returns_params[channel]['shape'],
                diminishing_returns_params[channel]['scale']
            )

        return -response

    # Define constraint (total budget)
    def budget_constraint(x):
        return total_budget - sum(x)

    # Define bounds
    bounds = [(bound_limits[channel]['min'], bound_limits[channel]['max'])
              for channel in channels]

    # Set up constraint
    constraints = [{'type': 'eq', 'fun': budget_constraint}]

    # Initial guess (current spend)
    x0 = [current_spend[channel] for channel in channels]

    # Run optimization
    result = opt.minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        logger.warning(f"Optimization failed: {result.message}")

    # Parse results
    optimized_budget = {channel: spend for channel, spend in zip(channels, result.x)}

    return optimized_budget


def optimize_budget_with_constraints(elasticities, current_spend, total_budget, min_budget=None, max_budget=None):
    """
    Optimize budget allocation with minimum and maximum constraints.

    Args:
        elasticities: Dictionary mapping channels to elasticities
        current_spend: Current spending for each channel
        total_budget: Total budget to allocate
        min_budget: Dictionary with minimum budget per channel
        max_budget: Dictionary with maximum budget per channel

    Returns:
        Dictionary with optimized budget allocation
    """
    channels = list(elasticities.keys())

    # Set default min/max if not provided
    if min_budget is None:
        min_budget = {channel: 0 for channel in channels}

    if max_budget is None:
        max_budget = {channel: total_budget for channel in channels}

    # Define objective function (negative because we maximize)
    def objective(x):
        # x is a vector of spend values
        spend_dict = {channel: spend for channel, spend in zip(channels, x)}

        # Calculate total response (assuming linear response with elasticity)
        response = 0
        for i, channel in enumerate(channels):
            # Only consider positive elasticities
            if elasticities[channel] > 0:
                response += spend_dict[channel] * elasticities[channel]

        return -response

    # Define constraint (total budget)
    def budget_constraint(x):
        return total_budget - sum(x)

    # Define bounds
    bounds = [(min_budget[channel], max_budget[channel]) for channel in channels]

    # Set up constraint
    constraints = [{'type': 'eq', 'fun': budget_constraint}]

    # Initial guess (current spend)
    x0 = [current_spend[channel] for channel in channels]

    # Run optimization
    result = opt.minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        logger.warning(f"Optimization failed: {result.message}")

    # Parse results
    optimized_budget = {channel: spend for channel, spend in zip(channels, result.x)}

    return optimized_budget


# Add this to mmm/optimization.py
def optimize_budget(elasticities, current_budget, current_allocation):
    """Simple budget optimization based on elasticities"""
    # Calculate optimal allocation
    total_elasticity = sum(elasticities.values())
    optimal_allocation = {}

    for channel, elasticity in elasticities.items():
        optimal_allocation[channel] = (elasticity / total_elasticity) * current_budget

    # Calculate expected lift
    current_effect = sum(elasticities[channel] * spend for channel, spend in current_allocation.items())
    optimal_effect = sum(elasticities[channel] * spend for channel, spend in optimal_allocation.items())

    expected_lift = 0
    if current_effect > 0:  # Avoid division by zero
        expected_lift = (optimal_effect - current_effect) / current_effect * 100

    # Prepare results
    results = pd.DataFrame({
        'Current Allocation': current_allocation,
        'Current %': [spend / current_budget * 100 for spend in current_allocation.values()],
        'Optimal Allocation': optimal_allocation,
        'Optimal %': [spend / current_budget * 100 for spend in optimal_allocation.values()],
        'Change %': [(optimal_allocation[ch] - current_allocation[ch]) / current_allocation[ch] * 100
                     for ch in current_allocation]
    })

    return results, expected_lift


# New functions as per implementation guide

def calculate_revenue_impact(spend_changes, elasticities, current_spend, current_revenue):
    """
    Calculate expected revenue impact from spend changes using elasticities.

    Args:
        spend_changes: Dictionary of {channel: new_spend}
        elasticities: Dictionary of {channel: elasticity}
        current_spend: Dictionary of {channel: current_spend}
        current_revenue: Current baseline revenue

    Returns:
        Expected new revenue
    """
    # Implementation with logarithmic response functions
    revenue_multiplier = 1.0
    for channel, new_spend in spend_changes.items():
        if channel in elasticities and channel in current_spend:
            # Add small constant to avoid division by zero
            current = max(current_spend[channel], 0.01)
            new = max(new_spend, 0.01)

            # Calculate multiplicative effect using elasticity formula
            # rev_new/rev_old = (spend_new/spend_old)^elasticity
            if elasticities[channel] > 0:  # Safeguard against zero/negative elasticities
                effect = (new / current) ** elasticities[channel]
                revenue_multiplier *= effect

    return current_revenue * revenue_multiplier


def objective_function(spend_allocation, channels, elasticities, current_spend,
                       current_revenue, diminishing_returns=False):
    """
    Objective function to maximize in optimization (negative because we minimize).

    Args:
        spend_allocation: Array of spend values (from optimizer)
        channels: List of channel names (to map array indices to channels)
        elasticities, current_spend, current_revenue: As above
        diminishing_returns: Whether to apply diminishing returns adjustments

    Returns:
        Negative expected revenue (for minimization)
    """
    # Convert optimization array to spend dictionary
    spend_changes = {channel: spend for channel, spend in zip(channels, spend_allocation)}

    # Apply diminishing returns adjustments if needed
    if diminishing_returns:
        # Implement logic for diminishing returns
        pass

    # Calculate expected revenue
    expected_revenue = calculate_revenue_impact(
        spend_changes, elasticities, current_spend, current_revenue
    )

    # Return negative as scipy.optimize minimizes by default
    return -expected_revenue


def total_budget_constraint(spend_allocation, total_budget):
    """Constraint function ensuring total spend equals budget."""
    return total_budget - sum(spend_allocation)


def channel_minimum_constraints(spend_allocation, channels, min_spend):
    """
    Generate constraint functions for minimum spend by channel.

    Args:
        spend_allocation: Array of spend values
        channels: List of channel names
        min_spend: Dictionary of {channel: minimum_spend}

    Returns:
        List of constraint dictionaries for scipy.optimize
    """
    constraints = []
    for i, channel in enumerate(channels):
        if channel in min_spend:
            # Create constraint: spend[i] - min_spend[channel] >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=i, min_val=min_spend[channel]: x[idx] - min_val
            })
    return constraints


def optimize_budget_allocation(elasticities, current_spend, current_revenue,
                               total_budget, min_spend=None, max_spend=None):
    """
    Find optimal budget allocation across channels.

    Args:
        elasticities: Dictionary of {channel: elasticity}
        current_spend: Dictionary of {channel: current_spend}
        current_revenue: Current baseline revenue
        total_budget: Total budget constraint
        min_spend: Dictionary of {channel: min_spend} or None
        max_spend: Dictionary of {channel: max_spend} or None

    Returns:
        Dictionary with optimized allocation and expected impact
    """
    # Setup
    channels = list(elasticities.keys())
    initial_allocation = [current_spend.get(channel, total_budget / len(channels))
                          for channel in channels]

    # Set up constraints
    constraints = [{'type': 'eq', 'fun': lambda x: total_budget - sum(x)}]

    # Add min/max constraints
    if min_spend:
        min_constraints = channel_minimum_constraints(
            initial_allocation, channels, min_spend
        )
        constraints.extend(min_constraints)

    if max_spend:
        for i, channel in enumerate(channels):
            if channel in max_spend:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=i, max_val=max_spend[channel]: max_val - x[idx]
                })

    # Bounds to prevent negative values
    bounds = [(0, None) for _ in channels]

    # Run optimization
    result = opt.minimize(
        lambda x: objective_function(x, channels, elasticities, current_spend, current_revenue),
        initial_allocation,
        method='SLSQP',  # Sequential Least Squares Programming (handles constraints well)
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-6, 'maxiter': 1000}
    )

    # Process results
    optimized_allocation = {channel: spend for channel, spend
                            in zip(channels, result.x)}

    # Calculate expected revenue
    expected_revenue = -result.fun  # Negative of minimized objective
    revenue_lift = expected_revenue - current_revenue
    percent_lift = (revenue_lift / current_revenue) * 100

    # ROI calculations
    current_total_spend = sum(current_spend.values())
    current_roi = current_revenue / current_total_spend
    optimized_roi = expected_revenue / total_budget
    roi_improvement = optimized_roi - current_roi

    # Organize results
    optimization_results = {
        'success': result.success,
        'status': result.message,
        'iterations': result.nit,
        'optimized_allocation': optimized_allocation,
        'expected_revenue': expected_revenue,
        'revenue_lift': revenue_lift,
        'percent_lift': percent_lift,
        'current_roi': current_roi,
        'optimized_roi': optimized_roi,
        'roi_improvement': roi_improvement
    }

    return optimization_results


def run_budget_scenarios(elasticities, current_spend, current_revenue,
                         budget_scenarios, min_spend_pct=0.5):
    """
    Run optimization for multiple budget scenarios.

    Args:
        elasticities, current_spend, current_revenue: As above
        budget_scenarios: List of total budget values to test
        min_spend_pct: Minimum spend as percentage of current (default 50%)

    Returns:
        Dictionary of scenario results
    """
    # Calculate minimum spend constraints (default: 50% of current)
    min_spend = {channel: current_spend[channel] * min_spend_pct
                 for channel in current_spend}

    # Run scenarios
    scenario_results = {}
    for budget in budget_scenarios:
        scenario_name = f"Budget_{budget}"
        scenario_results[scenario_name] = optimize_budget_allocation(
            elasticities, current_spend, current_revenue,
            total_budget=budget, min_spend=min_spend
        )

    return scenario_results


def plot_allocation_comparison(current_spend, optimized_allocation,
                               revenue_lift, percent_lift):
    """
    Create bar chart comparing current vs. optimized allocation.

    Args:
        current_spend: Dictionary of current spend by channel
        optimized_allocation: Dictionary of optimized spend by channel
        revenue_lift: Expected revenue lift
        percent_lift: Percent revenue lift

    Returns:
        Matplotlib figure
    """
    channels = list(current_spend.keys())
    current_values = [current_spend[ch] for ch in channels]
    optimized_values = [optimized_allocation[ch] for ch in channels]

    # Calculate changes
    changes = [(optimized_values[i] - current_values[i]) / current_values[i] * 100
               for i in range(len(channels))]

    # Setup plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Spend comparison
    x = np.arange(len(channels))
    width = 0.35

    ax1.bar(x - width / 2, current_values, width, label='Current')
    ax1.bar(x + width / 2, optimized_values, width, label='Optimized')

    ax1.set_title('Budget Allocation Comparison')
    ax1.set_ylabel('Spend')
    ax1.set_xticks(x)
    ax1.set_xticklabels(channels, rotation=45, ha='right')
    ax1.legend()

    # Plot 2: Percent changes
    colors = ['green' if c > 0 else 'red' for c in changes]
    ax2.bar(channels, changes, color=colors)
    ax2.set_title(f'Spend Changes (Expected Revenue Lift: {percent_lift:.1f}%)')
    ax2.set_ylabel('% Change')
    ax2.set_xticklabels(channels, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig('budget_optimization_results.png')

    return fig


def apply_budget_optimization(mmm_results, current_spend, total_budget=None,
                              min_percent=0.5, scenarios=None):
    """
    Apply budget optimization to MMM results.

    Args:
        mmm_results: Output from MMM containing elasticities
        current_spend: Dictionary of current spend by channel
        total_budget: Total budget (default: sum of current spend)
        min_percent: Minimum percent of current spend to maintain
        scenarios: List of budget scenarios to test (optional)

    Returns:
        Optimization results
    """
    # Extract elasticities
    elasticities = mmm_results['elasticities']

    # Get current revenue
    current_revenue = mmm_results.get('baseline_revenue', 1000000)  # Default if not provided

    # Set default budget to current total if not specified
    if total_budget is None:
        total_budget = sum(current_spend.values())

    # Define minimum spend constraints
    min_spend = {channel: spend * min_percent for channel, spend in current_spend.items()}

    # Run main optimization
    results = optimize_budget_allocation(
        elasticities, current_spend, current_revenue,
        total_budget, min_spend=min_spend
    )

    # Run scenarios if requested
    if scenarios:
        scenario_results = run_budget_scenarios(
            elasticities, current_spend, current_revenue,
            budget_scenarios=scenarios, min_spend_pct=min_percent
        )
        results['scenarios'] = scenario_results

    # Create visualization
    plot_allocation_comparison(
        current_spend, results['optimized_allocation'],
        results['revenue_lift'], results['percent_lift']
    )

    return results


def s_shaped_response(spend, current_spend, elasticity, saturation_point):
    """
    S-shaped response curve with saturation effects.

    Args:
        spend: New spend level
        current_spend: Current spend level
        elasticity: Channel elasticity
        saturation_point: Point at which diminishing returns become significant

    Returns:
        Response multiplier
    """
    relative_spend = spend / current_spend
    if relative_spend <= 1:
        # Below current spend: Use elasticity
        return relative_spend ** elasticity
    else:
        # Above current spend: Diminishing returns
        saturation = 1 - np.exp(-(relative_spend - 1) / saturation_point)
        return (1 + saturation * (relative_spend - 1) ** 0.5) ** elasticity


def calculate_revenue_with_interactions(spend_changes, elasticities,
                                        interaction_effects, current_spend,
                                        current_revenue):
    """
    Calculate revenue impact with interaction effects between channels.

    Args:
        spend_changes: Dictionary of {channel: new_spend}
        elasticities: Dictionary of {channel: elasticity}
        interaction_effects: Dictionary of {(channel1, channel2): effect}
        current_spend: Dictionary of {channel: current_spend}
        current_revenue: Current baseline revenue

    Returns:
        Expected new revenue
    """
    # Base calculation without interactions
    base_revenue = calculate_revenue_impact(
        spend_changes, elasticities, current_spend, current_revenue
    )

    # Calculate interaction effects
    interaction_multiplier = 1.0
    for (channel1, channel2), effect in interaction_effects.items():
        if (channel1 in spend_changes and channel2 in spend_changes and
                channel1 in current_spend and channel2 in current_spend):

            # Calculate relative spend changes
            rel_change1 = spend_changes[channel1] / current_spend[channel1]
            rel_change2 = spend_changes[channel2] / current_spend[channel2]

            # Apply interaction effect (multiplicative)
            if rel_change1 > 1 and rel_change2 > 1:  # Only apply for increases
                # Simple multiplicative interaction model
                interaction_term = (rel_change1 * rel_change2) ** effect
                interaction_multiplier *= interaction_term

    return base_revenue * interaction_multiplier


def calculate_elasticities(model, data, media_vars):
    """Calculate elasticities for media variables"""
    elasticities = {}

    # For log-log model, coefficients are elasticities
    for var in media_vars:
        if var in model.params:
            elasticities[var] = model.params[var]

    return elasticities


def calculate_roi(model, data, media_vars):
    """Calculate ROI for each media channel"""
    roi = {}

    # Extract mean values
    mean_revenue = data['revenue'].mean()

    for var in media_vars:
        # Get original channel name (before adstock)
        original_channel = var.replace('_adstock', '') if '_adstock' in var else var

        # Skip if channel not in model or data
        if var not in model.params or original_channel not in data.columns:
            continue

        # Calculate effect
        # For log-linear model: dY/dX = β * Y
        effect = model.params[var] * mean_revenue

        # Calculate ROI = effect per dollar spent
        mean_spend = data[original_channel].mean()
        if mean_spend > 0:
            roi[var] = effect / mean_spend

    return roi


def optimize_budget(elasticities, current_budget, current_allocation):
    """Simple budget optimization based on elasticities"""
    # Calculate optimal allocation
    total_elasticity = sum(elasticities.values())
    optimal_allocation = {}

    for channel, elasticity in elasticities.items():
        optimal_allocation[channel] = (elasticity / total_elasticity) * current_budget

    # Calculate expected lift
    current_effect = sum(elasticity * spend for channel, (elasticity, spend) in
                         zip(elasticities.items(), current_allocation.values()))

    optimal_effect = sum(elasticity * spend for channel, (elasticity, spend) in
                         zip(elasticities.items(), optimal_allocation.values()))

    expected_lift = (optimal_effect - current_effect) / current_effect * 100

    # Prepare results
    results = pd.DataFrame({
        'Current Allocation': current_allocation,
        'Current %': [spend / current_budget * 100 for spend in current_allocation.values()],
        'Optimal Allocation': optimal_allocation,
        'Optimal %': [spend / current_budget * 100 for spend in optimal_allocation.values()],
        'Change %': [(optimal_allocation[ch] - current_allocation[ch]) / current_allocation[ch] * 100
                     for ch in current_allocation]
    })

    return results, expected_lift


def extract_current_allocation(data, media_vars):
    """Extract current budget allocation from data"""
    # Get original channel names (before aggregation and adstock)
    original_channels = {
        'traditional_media': ['tv_spend'],
        'digital_paid': ['search_spend', 'display_spend'],
        'social_media': ['social_spend'],
        'owned_media': ['email_spend']
    }

    # Calculate mean spend for each aggregate channel
    current_allocation = {}
    for agg_channel, channels in original_channels.items():
        if agg_channel in media_vars:
            current_allocation[agg_channel] = sum(data[ch].mean() for ch in channels
                                                  if ch in data.columns)

    return current_allocation


# Only run this code when executing this file directly
if __name__ == "__main__":
    # Test code
    try:
        import pandas as pd
        from mmm.preprocessing import preprocess_data
        from mmm.modeling import build_mmm, apply_constraints

        # Load test data
        df = pd.read_csv('data/synthetic_advertising_data_v2.csv')
        processed_data = preprocess_data(df)

        # Build and constrain model
        model, _ = build_mmm(processed_data)
        media_vars = ['traditional_media', 'digital_paid', 'social_media', 'owned_media']
        constrained_model = apply_constraints(model, processed_data, media_vars)

        # Calculate elasticities and ROI
        elasticities = calculate_elasticities(constrained_model, processed_data, media_vars)
        roi = calculate_roi(constrained_model, processed_data, media_vars)

        print("\nElasticities:")
        for channel, value in elasticities.items():
            print(f"{channel}: {value:.4f}")

        print("\nROI Estimates:")
        for channel, value in roi.items():
            print(f"{channel}: {value:.2f}")

        # Optimize budget allocation
        current_allocation = extract_current_allocation(processed_data, media_vars)
        current_budget = sum(current_allocation.values())

        optimization_results, expected_lift = optimize_budget(elasticities, current_budget, current_allocation)

        print("\nBudget Optimization Results:")
        print(optimization_results)
        print(f"\nExpected Revenue Lift: {expected_lift:.2f}%")

        # Visualize allocation changes
        plt.figure(figsize=(10, 6))
        plt.bar(optimization_results.index, optimization_results['Current %'],
                color='lightblue', label='Current')
        plt.bar(optimization_results.index, optimization_results['Optimal %'],
                color='orange', alpha=0.7, label='Optimal')
        plt.title('Budget Allocation Comparison')
        plt.ylabel('Percent of Total Budget')
        plt.legend()
        plt.grid(True, axis='y')
        plt.show()
    except Exception as e:
        print(f"Error in test code: {e}")
