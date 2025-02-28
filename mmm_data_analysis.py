"""
MMM Data Evaluation Script
Analyzes synthetic marketing mix datasets to determine suitability for modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from pathlib import Path

# Set up the plots
plt.style.use('ggplot')
sns.set(style="whitegrid")


def analyze_mmm_dataset(file_path):
    """Analyze a marketing mix dataset and assess its suitability for modeling"""
    print(f"\n{'=' * 50}")
    print(f"ANALYZING DATASET: {Path(file_path).name}")
    print(f"{'=' * 50}")

    # Load the data
    df = pd.read_csv(file_path)

    # Basic dataset information
    print(f"\n1. DATASET OVERVIEW")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Columns: {', '.join(df.columns)}")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n   Missing Values:")
        print(missing[missing > 0])
    else:
        print("\n   No missing values")

    # Basic descriptive statistics
    print("\n2. DESCRIPTIVE STATISTICS")
    print(df.describe().T)

    # Identify likely target variable (usually 'Sales' or similar)
    target_col = None
    for col in ['Sales', 'sales', 'Revenue', 'revenue']:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        # If no obvious target column, attempt to identify it by naming pattern
        for col in df.columns:
            if 'sale' in col.lower() or 'revenue' in col.lower():
                target_col = col
                break

    # If still no target column, use the last column as a guess
    if target_col is None:
        target_col = df.columns[-1]
        print(f"\n   No obvious sales/revenue column found. Using {target_col} as target.")
    else:
        print(f"\n   Target variable identified: {target_col}")

    # Identify likely marketing variables
    marketing_cols = []
    for col in df.columns:
        if any(term in col.lower() for term in ['spend', 'tv', 'radio', 'digital', 'search', 'social', 'ad']):
            marketing_cols.append(col)

    print(f"   Marketing variables identified: {', '.join(marketing_cols)}")

    # Correlation analysis
    print("\n3. CORRELATION ANALYSIS")
    corr = df.corr()

    # Print correlations with target
    print(f"\n   Correlations with {target_col}:")
    target_corrs = corr[target_col].sort_values(ascending=False)
    for col, val in target_corrs.items():
        if col != target_col:
            print(f"   - {col}: {val:.4f}")

    # Check for multicollinearity between marketing variables
    if len(marketing_cols) > 1:
        print("\n   Marketing Variables Correlation Matrix:")
        marketing_corr = corr.loc[marketing_cols, marketing_cols]
        print(marketing_corr)

        # Identify highly correlated pairs
        high_corr_threshold = 0.7
        high_corr_pairs = []

        for i, col1 in enumerate(marketing_cols):
            for col2 in marketing_cols[i + 1:]:
                corr_val = abs(marketing_corr.loc[col1, col2])
                if corr_val > high_corr_threshold:
                    high_corr_pairs.append((col1, col2, corr_val))

        if high_corr_pairs:
            print("\n   High correlation pairs (potential multicollinearity):")
            for col1, col2, val in high_corr_pairs:
                print(f"   - {col1} & {col2}: {val:.4f}")
        else:
            print("\n   No severe multicollinearity detected between marketing variables")

    # Create scatter plots for marketing variables vs target
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(marketing_cols):
        plt.subplot(2, (len(marketing_cols) + 1) // 2, i + 1)
        plt.scatter(df[col], df[target_col], alpha=0.6)

        # Add trend line
        z = np.polyfit(df[col], df[target_col], 1)
        p = np.poly1d(z)
        plt.plot(df[col], p(df[col]), "r--", alpha=0.8)

        # Add correlation coefficient
        corr_val = np.corrcoef(df[col], df[target_col])[0, 1]
        plt.title(f"{col} vs {target_col}\nCorrelation: {corr_val:.4f}")
        plt.tight_layout()

    plt.savefig(f"{Path(file_path).stem}_scatter_plots.png")
    print(f"\n   Scatter plots saved as {Path(file_path).stem}_scatter_plots.png")

    # Time series analysis if time column exists
    time_col = None
    for col in ['date', 'Date', 'week', 'Week', 'time', 'Time']:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        print(f"\n4. TIME SERIES ANALYSIS")
        # If it's a string date, convert to datetime
        if df[time_col].dtype == 'object':
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                # If conversion fails, it might be a simple index
                pass

        # Sort by time column
        df = df.sort_values(time_col)

        # Plot target variable over time
        plt.figure(figsize=(15, 6))
        plt.plot(df[time_col], df[target_col], 'b-', marker='o', alpha=0.7)
        plt.title(f"{target_col} Over Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{Path(file_path).stem}_time_series.png")
        print(f"   Time series plot saved as {Path(file_path).stem}_time_series.png")

        # Check for time trends in target
        if df[time_col].dtype.name != 'object':  # If it's not just a string
            trend_corr = np.corrcoef(np.arange(len(df)), df[target_col])[0, 1]
            print(f"   Time trend correlation with {target_col}: {trend_corr:.4f}")
    else:
        print("\n4. TIME SERIES ANALYSIS: No time column found")
        # Create a simple index instead
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(df)), df[target_col], 'b-', marker='o', alpha=0.7)
        plt.title(f"{target_col} Values")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{Path(file_path).stem}_values.png")
        print(f"   Values plot saved as {Path(file_path).stem}_values.png")

    # Simple regression model
    print("\n5. SIMPLE REGRESSION MODEL")

    # Create X and y
    X = df[marketing_cols]
    X = sm.add_constant(X)  # Add constant term
    y = df[target_col]

    # Fit model
    model = sm.OLS(y, X).fit()

    # Print summary statistics
    print(f"   R-squared: {model.rsquared:.4f}")
    print(f"   Adjusted R-squared: {model.rsquared_adj:.4f}")

    # Print coefficients
    print("\n   Coefficients:")
    for var, coef in zip(model.model.exog_names, model.params):
        t_stat = coef / model.bse[model.model.exog_names.index(var)]
        p_value = model.pvalues[model.model.exog_names.index(var)]
        stars = ""
        if p_value < 0.05:
            stars = "*"
        if p_value < 0.01:
            stars = "**"
        if p_value < 0.001:
            stars = "***"

        print(f"   - {var}: {coef:.6f} (t={t_stat:.2f}, p={p_value:.4f}) {stars}")

    # Check distribution of residuals
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.hist(model.resid, bins=20, alpha=0.7)
    plt.title("Residuals Distribution")

    plt.subplot(1, 2, 2)
    plt.scatter(model.fittedvalues, model.resid, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(f"{Path(file_path).stem}_residuals.png")
    print(f"   Residuals plots saved as {Path(file_path).stem}_residuals.png")

    # Overall dataset assessment
    print("\n6. DATASET ASSESSMENT")

    # Check if marketing variables are predictive
    significant_vars = sum(model.pvalues[1:] < 0.05)  # Exclude constant

    if model.rsquared < 0.1:
        print("   ❌ R-squared is very low (<0.1)")
    elif model.rsquared < 0.3:
        print("   ⚠️ R-squared is moderate (0.1-0.3)")
    else:
        print("   ✅ R-squared is reasonable (>0.3)")

    if significant_vars == 0:
        print("   ❌ No marketing variables are statistically significant")
    elif significant_vars < len(marketing_cols) / 2:
        print(f"   ⚠️ Only {significant_vars} of {len(marketing_cols)} marketing variables are significant")
    else:
        print(f"   ✅ {significant_vars} of {len(marketing_cols)} marketing variables are significant")

    # Check for reasonable correlations
    marketing_target_corrs = [abs(corr[target_col][col]) for col in marketing_cols]
    avg_corr = np.mean(marketing_target_corrs)

    if avg_corr < 0.1:
        print(f"   ❌ Average correlation with {target_col} is very weak ({avg_corr:.4f})")
    elif avg_corr < 0.3:
        print(f"   ⚠️ Average correlation with {target_col} is moderate ({avg_corr:.4f})")
    else:
        print(f"   ✅ Average correlation with {target_col} is reasonable ({avg_corr:.4f})")

    if high_corr_pairs and len(high_corr_pairs) > len(marketing_cols) / 3:
        print(f"   ⚠️ High multicollinearity found in {len(high_corr_pairs)} variable pairs")

    # Overall suitability
    suitability_score = 0
    if model.rsquared >= 0.1: suitability_score += 1
    if model.rsquared >= 0.3: suitability_score += 1
    if significant_vars > 0: suitability_score += 1
    if significant_vars >= len(marketing_cols) / 2: suitability_score += 1
    if avg_corr >= 0.1: suitability_score += 1
    if avg_corr >= 0.3: suitability_score += 1

    if suitability_score <= 1:
        print("\n   OVERALL ASSESSMENT: Poor - Not suitable for MMM")
    elif suitability_score <= 3:
        print("\n   OVERALL ASSESSMENT: Marginal - May be challenging for MMM")
    elif suitability_score <= 5:
        print("\n   OVERALL ASSESSMENT: Adequate - Suitable with caveats")
    else:
        print("\n   OVERALL ASSESSMENT: Good - Well suited for MMM")

    return {
        'file_name': Path(file_path).name,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'significant_vars': significant_vars,
        'total_vars': len(marketing_cols),
        'avg_correlation': avg_corr,
        'multicollinearity_pairs': len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0,
        'suitability_score': suitability_score
    }


def main():
    """Analyze all three MMM datasets and compare results"""
    # Analyze each dataset
    results = []

    # Define dataset paths - update these to match your directory structure
    datasets = [
        "data/mmm_data.csv",
        "data/synthetic_advertising_data_v2.csv",
        "data/synthetic_dataset_noisy.csv"
    ]

    for dataset in datasets:
        try:
            result = analyze_mmm_dataset(dataset)
            results.append(result)
        except Exception as e:
            print(f"\nError analyzing {dataset}: {str(e)}")

    # Compare results
    print("\n\n" + "=" * 80)
    print("DATASET COMPARISON")
    print("=" * 80)

    comparison_df = pd.DataFrame(results)
    print(comparison_df[['file_name', 'r_squared', 'adj_r_squared', 'significant_vars',
                         'total_vars', 'avg_correlation', 'suitability_score']])

    # Identify best dataset
    if len(results) > 0:
        best_dataset = max(results, key=lambda x: x['suitability_score'])
        print(f"\nMost suitable dataset: {best_dataset['file_name']}")
        print(f"Suitability score: {best_dataset['suitability_score']}/6")

        if best_dataset['suitability_score'] <= 1:
            print("\nWARNING: Even the best dataset has significant issues.")
            print("Consider generating better synthetic data or using real data.")


if __name__ == "__main__":
    main()
