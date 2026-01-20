"""
Quick demonstration of F1 Lap Time Forecasting Pipeline
Shows all components working with sample data
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ensemble.xgboost_model import XGBoostModel
from src.models.ensemble.random_forest_model import RandomForestModel
from src.models.regression.ridge_model import RidgeModel
from src.models.regression.svr_model import SVRModel
from src.models.neural.neural_net_model import NeuralNetModel
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.visualization import plot_predictions_vs_actual, plot_model_comparison

print("="*80)
print("🏎️  F1 LAP TIME FORECASTING - PIPELINE DEMONSTRATION")
print("="*80)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# STEP 1: Generate Realistic Sample Data
# ============================================================================
print("\n[1/5] 📊 Generating realistic F1 sample data...")

n_samples = 2000
n_features = 15

# Generate features
feature_names = [
    'lap_time_lag_1', 'lap_time_rolling_mean_3', 'lap_number',
    'fuel_load_proxy', 'grid_position', 'driver_experience',
    'track_avg_lap_time', 'driver_avg_position', 'championship_rank',
    'gap_to_leader', 'career_races', 'season_points',
    'constructor_strength', 'weather_temp', 'tire_age'
]

# Create feature matrix
X = np.random.randn(n_samples, n_features)

# Create realistic target variable (lap times 80-110 seconds)
# Base lap time influenced by key features
base_lap_time = 95
y = base_lap_time + \
    3 * X[:, 0] + \
    2 * X[:, 1] + \
    1.5 * X[:, 2] + \
    1 * X[:, 3] + \
    0.5 * X[:, 4] + \
    np.random.normal(0, 2, n_samples)

# Convert to DataFrame
X_df = pd.DataFrame(X, columns=feature_names)

print(f"  ✅ Generated {n_samples} samples with {n_features} features")
print(f"  📈 Lap time range: {y.min():.2f}s - {y.max():.2f}s (mean: {y.mean():.2f}s)")

# ============================================================================
# STEP 2: Train/Test Split
# ============================================================================
print("\n[2/5] 🔀 Splitting data (train/val/test)...")

train_size = int(0.7 * n_samples)
val_size = int(0.15 * n_samples)

X_train = X_df[:train_size]
y_train = y[:train_size]

X_val = X_df[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X_df[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"  ✅ Training: {len(X_train)} samples")
print(f"  ✅ Validation: {len(X_val)} samples")
print(f"  ✅ Test: {len(X_test)} samples")

# ============================================================================
# STEP 3: Train Multiple Models
# ============================================================================
print("\n[3/5] 🤖 Training ML models...")

models = {}
results = {}

# XGBoost
print("\n  🌟 Training XGBoost...")
xgb_config = {'params': {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42}}
xgb_model = XGBoostModel(xgb_config)
xgb_model.train(X_train, y_train, X_val, y_val)
xgb_results = xgb_model.evaluate(X_test, y_test)
models['XGBoost'] = xgb_model
results['XGBoost'] = xgb_results
print(f"     MAE: {xgb_results['mae']:.3f}s | RMSE: {xgb_results['rmse']:.3f}s | R²: {xgb_results['r2']:.3f}")

# Random Forest
print("\n  🌲 Training Random Forest...")
rf_config = {'params': {'n_estimators': 50, 'max_depth': 15, 'random_state': 42}}
rf_model = RandomForestModel(rf_config)
rf_model.train(X_train, y_train, X_val, y_val)
rf_results = rf_model.evaluate(X_test, y_test)
models['Random Forest'] = rf_model
results['Random Forest'] = rf_results
print(f"     MAE: {rf_results['mae']:.3f}s | RMSE: {rf_results['rmse']:.3f}s | R²: {rf_results['r2']:.3f}")

# Ridge Regression
print("\n  📐 Training Ridge Regression...")
ridge_config = {'params': {'alpha': 1.0, 'random_state': 42}}
ridge_model = RidgeModel(ridge_config)
ridge_model.train(X_train, y_train)
ridge_results = ridge_model.evaluate(X_test, y_test)
models['Ridge'] = ridge_model
results['Ridge'] = ridge_results
print(f"     MAE: {ridge_results['mae']:.3f}s | RMSE: {ridge_results['rmse']:.3f}s | R²: {ridge_results['r2']:.3f}")

# SVR
print("\n  🎯 Training Support Vector Regression...")
svr_config = {'params': {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}}
svr_model = SVRModel(svr_config)
svr_model.train(X_train, y_train)
svr_results = svr_model.evaluate(X_test, y_test)
models['SVR'] = svr_model
results['SVR'] = svr_results
print(f"     MAE: {svr_results['mae']:.3f}s | RMSE: {svr_results['rmse']:.3f}s | R²: {svr_results['r2']:.3f}")

# Neural Network
print("\n  🧠 Training Neural Network...")
nn_config = {'params': {'hidden_layers': [64, 32], 'dropout_rate': 0.2, 'epochs': 50,
                        'batch_size': 32, 'learning_rate': 0.001}}
nn_model = NeuralNetModel(nn_config)
nn_model.train(X_train, y_train, X_val, y_val)
nn_results = nn_model.evaluate(X_test, y_test)
models['Neural Network'] = nn_model
results['Neural Network'] = nn_results
print(f"     MAE: {nn_results['mae']:.3f}s | RMSE: {nn_results['rmse']:.3f}s | R²: {nn_results['r2']:.3f}")

# ============================================================================
# STEP 4: Evaluate and Compare
# ============================================================================
print("\n[4/5] 📊 Evaluating and comparing models...")

# Create results DataFrame
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [r['mae'] for r in results.values()],
    'RMSE': [r['rmse'] for r in results.values()],
    'R²': [r['r2'] for r in results.values()],
    'MAPE': [r['mape'] for r in results.values()]
})

# Sort by MAE
results_df = results_df.sort_values('MAE')

print("\n  📈 MODEL PERFORMANCE SUMMARY")
print("  " + "="*76)
print(results_df.to_string(index=False))
print("  " + "="*76)

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
best_mae = results_df.iloc[0]['MAE']

print(f"\n  🏆 BEST MODEL: {best_model_name} (MAE: {best_mae:.3f}s)")

# ============================================================================
# STEP 5: Generate Predictions and Visualizations
# ============================================================================
print("\n[5/5] 🎨 Generating visualizations...")

# Make predictions with best model
y_pred = best_model.predict(X_test)

# Create output directory
output_dir = Path('demo_results')
output_dir.mkdir(exist_ok=True)

# Save results
results_df.to_csv(output_dir / 'model_comparison.csv', index=False)
print(f"  ✅ Saved model comparison to {output_dir / 'model_comparison.csv'}")

# Generate prediction scatter plot
plot_predictions_vs_actual(
    y_test, y_pred,
    title=f"{best_model_name} - Predictions vs Actual Lap Times",
    save_path=str(output_dir / 'predictions_vs_actual.png')
)
print(f"  ✅ Saved predictions plot to {output_dir / 'predictions_vs_actual.png'}")

# Generate model comparison plot
plot_model_comparison(
    results_df, metric='MAE',
    save_path=str(output_dir / 'model_comparison.png')
)
print(f"  ✅ Saved comparison plot to {output_dir / 'model_comparison.png'}")

# Save predictions
predictions_df = pd.DataFrame({
    'actual_lap_time': y_test,
    'predicted_lap_time': y_pred,
    'error': y_test - y_pred,
    'abs_error': np.abs(y_test - y_pred)
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
print(f"  ✅ Saved predictions to {output_dir / 'predictions.csv'}")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("✨ PIPELINE EXECUTION COMPLETE!")
print("="*80)

print("\n📊 KEY METRICS:")
print(f"  • Best Model: {best_model_name}")
print(f"  • Mean Absolute Error: {best_mae:.3f} seconds")
print(f"  • Root Mean Squared Error: {results_df.iloc[0]['RMSE']:.3f} seconds")
print(f"  • R² Score: {results_df.iloc[0]['R²']:.3f}")
print(f"  • Mean Absolute Percentage Error: {results_df.iloc[0]['MAPE']:.2f}%")

print("\n📁 OUTPUT FILES:")
print(f"  • {output_dir / 'model_comparison.csv'}")
print(f"  • {output_dir / 'predictions.csv'}")
print(f"  • {output_dir / 'predictions_vs_actual.png'}")
print(f"  • {output_dir / 'model_comparison.png'}")

print("\n🎯 ACCURACY BREAKDOWN:")
errors = np.abs(y_test - y_pred)
print(f"  • Within 1s: {np.sum(errors <= 1) / len(errors) * 100:.1f}%")
print(f"  • Within 2s: {np.sum(errors <= 2) / len(errors) * 100:.1f}%")
print(f"  • Within 3s: {np.sum(errors <= 3) / len(errors) * 100:.1f}%")

print("\n🏎️  FEATURE IMPORTANCE (Top 5):")
if hasattr(best_model, 'get_feature_importance'):
    importances = best_model.get_feature_importance()
    if importances is not None:
        top_indices = np.argsort(importances)[-5:][::-1]
        for idx in top_indices:
            print(f"  • {feature_names[idx]}: {importances[idx]:.4f}")

print("\n" + "="*80)
print("🚀 All models trained and evaluated successfully!")
print("="*80)
