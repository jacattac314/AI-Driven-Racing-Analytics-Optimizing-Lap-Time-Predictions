from setuptools import setup, find_packages

setup(
    name="f1-lap-time-forecasting",
    version="0.1.0",
    description="AI-Driven Racing Analytics: Optimizing Lap Time Predictions",
    author="F1 Analytics Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "tensorflow>=2.11.0",
        "statsmodels>=0.13.0",
        "pmdarima>=2.0.0",
        "shap>=0.41.0",
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ]
    },
)
