# In a Python script or shell
from mmm.model_diagnostics import analyze_residuals
import pandas as pd
import statsmodels.api as sm

# Load your data
df = pd.read_csv('/data/mmm_data.csv')

# Fit a simple model
X = sm.add_constant(df[['TV_Spend', 'Digital_Spend']])
y = df['Sales']
model = sm.OLS(y, X).fit()

# Run residual analysis
results = analyze_residuals(model, X, y)

# Check results
print(results['summary'])
