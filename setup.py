"""
Setup script for the Marketing Mix Model package.
"""

from setuptools import setup, find_packages

setup(
    name="mmm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],
    author="Corbin Bridge",
    author_email="cbridge89@gmail.com",
    description="Marketing Mix Model package for analyzing marketing effectiveness",
    keywords="marketing, analytics, statistics, model",
    python_requires=">=3.7",
)