# Marketing Mix Model (MMM) Framework

## Project Overview
This Marketing Mix Model (MMM) framework provides a comprehensive solution for analyzing marketing effectiveness and optimizing budget allocation across channels. It enables marketers and data scientists to quantify the impact of marketing activities on business outcomes, measure return on investment (ROI), and make data-driven decisions about future marketing investments.

The framework implements state-of-the-art techniques for handling common challenges in marketing measurement, including:
- Media carryover effects through adstock transformations
- Diminishing returns modeling
- Multicollinearity mitigation using PCA and constrained regression
- Channel attribution and elasticity calculation
- Budget optimization with various constraints

## Key Features
- **Advanced Preprocessing**: Stationarity checks, feature engineering, and transformation options
- **Flexible Adstock Implementations**: Multiple carryover effect patterns (geometric, Weibull, delayed)
- **Robust Modeling Approaches**: 
  - Linear and log-log model specifications
  - PCA for multicollinearity 
  - Constrained optimization for valid elasticities
- **Comprehensive Visualization**: Response curves, media contributions, and performance metrics
- **Budget Optimization**: Multiple approaches from simple allocation to advanced response-based optimization
- **Detailed Diagnostics**: Model quality assessment, multicollinearity detection, and decomposition tools

## Installation

### Prerequisites
- Python 3.8+
- Dependencies listed in requirements.txt

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/yourusername/marketing-mix-model.git
cd marketing-mix-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies
The framework relies on the following key libraries:
- pandas, numpy for data manipulation
- statsmodels for regression analysis
- scikit-learn for machine learning components
- scipy for advanced optimization
- matplotlib, seaborn for visualization

## Quick Start Guide

### Basic MMM Analysis
```python
from mmm.core import MarketingMixModel
from mmm.preprocessing import preprocess_data
from mmm.adstock import apply_adstock

# Load data
data = pd.read_csv('your_data.csv')

# Initialize model
mmm = MarketingMixModel()
mmm.load_data_from_dataframe(data)

# Preprocess data
mmm.preprocess_data(
    target='revenue',
    date_col='date',
    media_cols=['tv_spend', 'digital_display_spend', 'search_spend', 'social_media_spend'],
    control_cols=['price_index', 'competitor_price_index']
)

# Apply adstock transformations
mmm.apply_adstock_to_all_media()

# Fit model
results = mmm.fit_model()

# Calculate elasticities
elasticities = mmm.calculate_elasticities()

# Optimize budget
optimized_budget = mmm.optimize_budget()

# Generate report
report = mmm.generate_summary_report()
print(report)
```

### Advanced Usage with Log Transformations and Improved Modeling
```python
from examples.basic_mmm import run_log_log_model

# Run log-log model comparison
results = run_log_log_model('your_data.csv')

# Access results
baseline_elasticities = results['baseline']['elasticities']
log_log_elasticities = results['log_log']['elasticities']

# Compare model performance
baseline_metrics = results['baseline']['metrics']
log_log_metrics = results['log_log']['metrics']
```

### Complete Workflow with Optimizations
```python
from examples.improved_mmm import run_complete_mmm_workflow

# Run complete MMM workflow from data loading to optimization
results = run_complete_mmm_workflow('your_data.csv')

# Access components of the results
model = results['model']
validation = results['validation']
elasticities = results['elasticities']
roi = results['roi']
optimization = results['optimization']
expected_lift = results['expected_lift']
```

## Configuration Options

### Adstock Parameters
Adstock transformations can be customized with parameters like:
- `decay_rate`: Controls how quickly the marketing effect diminishes over time
- `lag_weight`: Weight applied to lagged effects
- `max_lag`: Maximum number of time periods to consider for carryover

Example:
```python
decay_rates = {
    'tv_spend': 0.85,  # TV has longer effect
    'digital_display_spend': 0.7,
    'search_spend': 0.3,  # Search has more immediate effect
    'social_media_spend': 0.6
}

mmm.apply_adstock_to_all_media(
    media_cols=['tv_spend', 'digital_display_spend', 'search_spend', 'social_media_spend'],
    decay_rates=decay_rates
)
```

### Modeling Approaches
The framework supports multiple modeling approaches:
- Standard OLS regression
- Ridge regression for regularization
- Lasso regression for feature selection
- PCA-based regression for multicollinearity
- Constrained optimization for valid coefficients

### Optimization Constraints
Budget optimization can be customized with:
- Minimum spend constraints
- Maximum spend constraints
- Total budget constraints
- Channel-specific bounds

## Project Structure

```
marketing-mix-model/
├── mmm/                      # Core implementation
│   ├── __init__.py           # Package initialization
│   ├── core.py               # Main MMM implementation
│   ├── preprocessing.py      # Data preprocessing utilities
│   ├── adstock.py            # Adstock transformation functions
│   ├── modeling.py           # Advanced modeling approaches
│   ├── optimization.py       # Budget optimization algorithms
│   ├── utils.py              # Helper functions
│   └── visualization.py      # Visualization tools
├── examples/                 # Example implementations
│   ├── basic_mmm.py          # Basic implementation with linear vs log-log
│   ├── improved_mmm.py       # Enhanced implementation with constraints
│   ├── optimized_mmm.py      # Full optimization workflow
│   ├── adstock_grid_search.py # Grid search for optimal adstock parameters
│   ├── channel_interactions.py # Analysis of channel interaction effects
│   ├── seasonality.py        # Time-based effects and seasonality
│   ├── budget_optimization.py # Budget allocation optimization
│   └── combined_mmm.py       # Combined approach with all enhancements
├── data/                     # Data directory (add your data here)
├── docs/                     # Documentation
├── tests/                    # Unit tests
└── requirements.txt          # Dependencies
```

## Contribution Guidelines
Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License Information
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- This framework is built on statistical techniques and methodologies from marketing science literature
- Special thanks to the open source libraries that make this possible: statsmodels, scikit-learn, pandas, and numpy

## Citation
If you use this framework in your research or business applications, please cite:

```
Marketing Mix Model Framework (2025)
https://github.com/yourusername/marketing-mix-model
```