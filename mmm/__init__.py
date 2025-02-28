"""
Marketing Mix Model (MMM) package for analyzing marketing effectiveness.
"""

from mmm.core import MarketingMixModel
from mmm.adstock import apply_adstock, geometric_adstock, weibull_adstock
from mmm.modeling import fit_ridge_model, fit_lasso_model
from mmm.optimization import optimize_budget

__version__ = '0.1.0'