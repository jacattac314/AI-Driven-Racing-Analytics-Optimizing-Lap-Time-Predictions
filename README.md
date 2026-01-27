# F1 Lap Time Forecasting: AI-Driven Racing Analytics

Complete end-to-end machine learning pipeline for Formula 1 lap time prediction using ensemble, regression, time-series, and deep learning models.

## Overview

This project implements a comprehensive ML pipeline to predict F1 lap times using:
- **Data**: Public F1 data from Ergast API (2000-2024)
- **Models**: XGBoost, Random Forest, SVR, KNN, Ridge, ARIMA, Neural Networks, Stacking Ensemble
- **Features**: Driver stats, track characteristics, weather, temporal patterns, historical performance
- **Optimization**: Cross-validation, GridSearchCV, RFECV feature selection
- **Metrics**: MAE (primary), RMSE, R², MAPE

## Features

✅ **Complete Data Pipeline**
- Automated data acquisition from Ergast F1 API
- Data validation and quality checks
- Time-series aware data cleaning and preprocessing

✅ **Comprehensive Feature Engineering**
- Driver features (career stats, recent form, championship standing)
- Track features (characteristics, historical performance)
- Temporal features (lag, rolling windows, delta calculations)
- Historical features (constructor performance, season trends)
- Weather and tire strategy features

✅ **8 Model Types**
- **Ensemble**: XGBoost, Random Forest, Stacking
- **Regression**: SVR, KNN, Ridge
- **Time-Series**: ARIMA
- **Deep Learning**: Neural Networks (TensorFlow/Keras)

✅ **Training Infrastructure**
- Time-series cross-validation
- Hyperparameter tuning (GridSearchCV)
- Feature selection (RFECV)
- Early stopping and learning rate scheduling

✅ **Evaluation & Interpretability**
- Comprehensive metrics (MAE, RMSE, R², MAPE)
- Per-track and per-driver performance analysis
- SHAP values for model interpretation
- Feature importance analysis

✅ **Automation**
- CLI scripts for each pipeline step
- End-to-end pipeline runner
- Jupyter notebooks for exploration

## Project Structure

```
AI-Driven-Racing-Analytics-Optimizing-Lap-Time-Predictions/
├── config/
│   ├── config.yaml                 # Main configuration
│   └── model_configs/              # Model-specific configs
├── data/
│   ├── raw/                        # Raw Ergast API data
│   ├── processed/                  # Cleaned data
│   ├── features/                   # Engineered features
│   └── splits/                     # Train/val/test splits
├── src/
│   ├── data/                       # Data pipeline
│   │   ├── data_collector.py       # Ergast API client
│   │   ├── data_validator.py       # Data validation
│   │   ├── data_cleaner.py         # Data cleaning
│   │   └── data_splitter.py        # Time-series splitting
│   ├── features/                   # Feature engineering
│   │   ├── feature_engineer.py     # Main orchestrator
│   │   ├── driver_features.py      # Driver-specific features
│   │   ├── track_features.py       # Track characteristics
│   │   ├── temporal_features.py    # Time-series features
│   │   ├── historical_features.py  # Historical performance
│   │   └── feature_selector.py     # RFECV selection
│   ├── models/                     # Model implementations
│   │   ├── base_model.py           # Base model class
│   │   ├── ensemble/               # XGBoost, RandomForest, Stacking
│   │   ├── regression/             # SVR, KNN, Ridge
│   │   ├── timeseries/             # ARIMA
│   │   └── neural/                 # Neural networks
│   ├── training/                   # Training infrastructure
│   │   ├── trainer.py              # Training orchestrator
│   │   ├── cross_validator.py      # CV strategies
│   │   └── hyperparameter_tuner.py # GridSearchCV
│   ├── evaluation/                 # Evaluation tools
│   │   ├── metrics.py              # Metrics calculation
│   │   ├── evaluator.py            # Model comparison
│   │   └── interpretability.py     # SHAP analysis
│   └── utils/                      # Utilities
│       ├── logging_utils.py
│       ├── io_utils.py
│       └── visualization.py
├── scripts/                        # CLI scripts
│   ├── fetch_data.py               # Data acquisition
│   ├── preprocess_data.py          # Preprocessing
│   ├── engineer_features.py        # Feature engineering
│   ├── train_all_models.py         # Training
│   ├── evaluate_models.py          # Evaluation
│   └── run_pipeline.py             # End-to-end pipeline
├── notebooks/                      # Jupyter notebooks
├── models/                         # Saved models
├── results/                        # Evaluation results
│   ├── metrics/                    # Performance metrics
│   ├── plots/                      # Visualizations
│   └── reports/                    # Analysis reports
├── tests/                          # Unit tests
├── requirements.txt                # Dependencies
├── setup.py                        # Package setup
├── Makefile                        # Automation commands
└── README.md                       # This file
```

## Installation

### Requirements
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-Driven-Racing-Analytics-Optimizing-Lap-Time-Predictions.git
cd AI-Driven-Racing-Analytics-Optimizing-Lap-Time-Predictions

# Install dependencies
pip install -r requirements.txt

# Or use make
make setup
```

## Quick Start

### Option 1: Run Complete Pipeline

```bash
# Run end-to-end pipeline
python scripts/run_pipeline.py

# Or use make
make all
```

### Option 2: Run Individual Steps

```bash
# 1. Fetch data from Ergast API
python scripts/fetch_data.py

# 2. Clean and preprocess data
python scripts/preprocess_data.py

# 3. Engineer features
python scripts/engineer_features.py

# 4. Train all models
python scripts/train_all_models.py

# 5. Evaluate models
python scripts/evaluate_models.py
```

### Option 3: Use Make Commands

```bash
make fetch       # Fetch data
make preprocess  # Preprocess data
make features    # Engineer features
make train       # Train models
make evaluate    # Evaluate models
make all         # Run complete pipeline
```

## Configuration

Edit `config/config.yaml` to customize:

- **Data**: API settings, seasons to fetch
- **Features**: Enable/disable feature groups, rolling windows, lag features
- **Training**: Cross-validation folds, train/val/test splits
- **Models**: Enable/disable models, hyperparameters, tuning grids

Example:

```yaml
data:
  seasons:
    start: 2000
    end: 2024

features:
  enabled_features:
    driver_features: true
    track_features: true
    temporal_features: true

models:
  xgboost:
    enabled: true
    params:
      learning_rate: 0.1
      max_depth: 6
      n_estimators: 100
```

## Usage Examples

### Train Specific Models

```python
from src.training.trainer import ModelTrainer
from src.utils.io_utils import load_config

config = load_config('config/config.yaml')
trainer = ModelTrainer(config)

# Train only XGBoost and Random Forest
models = trainer.train_all_models(
    X_train, y_train, X_val, y_val,
    models_to_train=['xgboost', 'random_forest']
)
```

### Feature Engineering

```python
from src.features.feature_engineer import FeatureEngineer

feature_engineer = FeatureEngineer(config)
df_features = feature_engineer.create_all_features(df)
```

### Model Evaluation

```python
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator()
comparison_df = evaluator.evaluate_all_models(models, X_test, y_test)
print(comparison_df)
```

## Data

Data is sourced from the [Ergast F1 API](http://ergast.com/mrd/):
- **Races**: Race schedule, circuits, dates
- **Results**: Race finishing positions, points, lap times
- **Qualifying**: Qualifying session times
- **Pit Stops**: Pit stop timing and duration
- **Lap Times**: Lap-by-lap timing (sampled)

**Time Range**: 2000-2024 (25 seasons)

**Data Split**:
- Train: 2000-2019 (20 seasons)
- Validation: 2020-2021 (2 seasons)
- Test: 2022-2024 (3 seasons)

## Models

### Ensemble Models
- **XGBoost**: Gradient boosting with early stopping
- **Random Forest**: Bagging with out-of-bag error
- **Stacking**: Meta-ensemble of XGBoost, RF, SVR, Ridge

### Regression Models
- **SVR**: Support Vector Regression with RBF kernel
- **KNN**: K-Nearest Neighbors with distance weighting
- **Ridge**: L2 regularized linear regression

### Time-Series Models
- **ARIMA**: Auto-ARIMA with seasonal components

### Deep Learning
- **Neural Network**: Dense layers with dropout and batch normalization

## Performance

Expected performance on test set:
- **MAE**: < 2.0 seconds
- **RMSE**: < 3.0 seconds
- **R²**: > 0.85

Best performing models are typically:
1. XGBoost
2. Stacking Ensemble
3. Random Forest

## Interpretability

### Feature Importance
```python
from src.evaluation.interpretability import ModelInterpreter

interpreter = ModelInterpreter()
importance_df = interpreter.get_feature_importance(model, feature_names, 'xgboost')
```

### SHAP Values
```python
shap_values = interpreter.calculate_shap_values(model, X_test, 'xgboost')
interpreter.plot_shap_summary('xgboost')
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Development

### Code Style
```bash
# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/
```

### Adding New Models

1. Create model class in `src/models/`
2. Inherit from `BaseModel`
3. Implement `train()` and `predict()` methods
4. Add to `ModelTrainer.get_model_instance()`
5. Add configuration to `config/config.yaml`

## Results

Results are saved in `results/`:
- **metrics/**: JSON files with performance metrics
- **plots/**: Visualizations (predictions vs actual, residuals, comparison)
- **reports/**: Statistical analysis reports

## Documentation

- **[README.md](README.md)** - Complete technical documentation and setup guide
- **[SHOWCASE.md](SHOWCASE.md)** - Project results, visualizations, and achievements
- **[MIRO.md](MIRO.md)** - Visual planning guide for architecture diagrams and workflow mapping

## Troubleshooting

**API Rate Limiting**: Adjust `rate_limit` in config.yaml

**Memory Issues**: Reduce data size or use sampling

**Model Training Failures**: Check logs in `logs/` directory

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License

## Citation

If you use this project, please cite:

```bibtex
@software{f1_lap_time_forecasting,
  title={F1 Lap Time Forecasting: AI-Driven Racing Analytics},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/AI-Driven-Racing-Analytics-Optimizing-Lap-Time-Predictions}
}
```

## Acknowledgments

- [Ergast F1 API](http://ergast.com/mrd/) for providing F1 data
- scikit-learn, XGBoost, TensorFlow communities
- Formula 1 and motorsport analytics community

## Contact

For questions or support, please open an issue on GitHub.
