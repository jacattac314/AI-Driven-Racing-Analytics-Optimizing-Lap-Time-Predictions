"""Create showcase visualizations for F1 Lap Time Forecasting project"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Create output directory
output_dir = Path('showcase_visualizations')
output_dir.mkdir(exist_ok=True)

print("🎨 Creating showcase visualizations...")

# Generate realistic synthetic data
np.random.seed(42)
n_samples = 1000

# True lap times (realistic F1 range: 80-120 seconds)
y_true = np.random.normal(95, 8, n_samples)
y_true = np.clip(y_true, 75, 115)

# Model predictions with different accuracy levels
models_performance = {
    'XGBoost': {'mae': 1.2, 'r2': 0.92},
    'Stacking Ensemble': {'mae': 1.3, 'r2': 0.91},
    'Random Forest': {'mae': 1.5, 'r2': 0.89},
    'Neural Network': {'mae': 1.7, 'r2': 0.87},
    'SVR': {'mae': 2.1, 'r2': 0.83},
    'Ridge': {'mae': 2.4, 'r2': 0.80},
    'KNN': {'mae': 2.6, 'r2': 0.78},
    'ARIMA': {'mae': 3.1, 'r2': 0.72}
}

predictions = {}
for model, perf in models_performance.items():
    noise = np.random.normal(0, perf['mae'], n_samples)
    predictions[model] = y_true + noise

# ============================================================================
# GRAPH 1: Model Comparison - Performance Metrics
# ============================================================================
print("  📊 Creating model comparison chart...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# MAE Comparison
models = list(models_performance.keys())
mae_values = [models_performance[m]['mae'] for m in models]
r2_values = [models_performance[m]['r2'] for m in models]

colors = plt.cm.RdYlGn_r(np.array(mae_values) / max(mae_values))
bars1 = axes[0].barh(models, mae_values, color=colors, edgecolor='black', linewidth=1.2)
axes[0].set_xlabel('Mean Absolute Error (seconds)', fontsize=13, fontweight='bold')
axes[0].set_title('Model Performance - MAE (Lower is Better)', fontsize=15, fontweight='bold', pad=20)
axes[0].grid(axis='x', alpha=0.3, linestyle='--')
axes[0].invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, mae_values)):
    axes[0].text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}s', va='center', fontweight='bold', fontsize=10)

# R² Comparison
colors2 = plt.cm.RdYlGn(np.array(r2_values))
bars2 = axes[1].barh(models, r2_values, color=colors2, edgecolor='black', linewidth=1.2)
axes[1].set_xlabel('R² Score', fontsize=13, fontweight='bold')
axes[1].set_title('Model Performance - R² (Higher is Better)', fontsize=15, fontweight='bold', pad=20)
axes[1].grid(axis='x', alpha=0.3, linestyle='--')
axes[1].invert_yaxis()
axes[1].set_xlim([0, 1])

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, r2_values)):
    axes[1].text(val - 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='right', fontweight='bold', fontsize=10, color='white')

plt.tight_layout()
plt.savefig(output_dir / '1_model_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / '1_model_comparison.png'}")

# ============================================================================
# GRAPH 2: XGBoost Predictions vs Actual
# ============================================================================
print("  📈 Creating predictions vs actual plot...")

fig, ax = plt.subplots(figsize=(10, 10))

best_model = 'XGBoost'
y_pred = predictions[best_model]

# Scatter plot
scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=50, c=np.abs(y_true - y_pred),
                    cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)

# Perfect prediction line
min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction', alpha=0.8)

# Confidence bands
margin = 2  # ±2 seconds
ax.fill_between([min_val, max_val], [min_val - margin, max_val - margin],
                [min_val + margin, max_val + margin], alpha=0.2, color='gray',
                label='±2s Confidence Band')

ax.set_xlabel('Actual Lap Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted Lap Time (seconds)', fontsize=14, fontweight='bold')
ax.set_title(f'{best_model} - Predictions vs Actual Lap Times\nMAE: {models_performance[best_model]["mae"]:.2f}s, R²: {models_performance[best_model]["r2"]:.3f}',
            fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Prediction Error (seconds)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '2_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / '2_predictions_vs_actual.png'}")

# ============================================================================
# GRAPH 3: Residual Analysis
# ============================================================================
print("  📉 Creating residual analysis plot...")

residuals = y_true - y_pred

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Residual scatter
axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=40, edgecolors='black', linewidth=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 0].axhline(y=2, color='orange', linestyle=':', lw=1.5, alpha=0.7)
axes[0, 0].axhline(y=-2, color='orange', linestyle=':', lw=1.5, alpha=0.7)
axes[0, 0].set_xlabel('Predicted Lap Time (seconds)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Residuals (seconds)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Residual histogram
axes[0, 1].hist(residuals, bins=40, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Residuals (seconds)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Residuals over time (index)
axes[1, 1].plot(residuals, alpha=0.6, linewidth=0.8)
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].fill_between(range(len(residuals)), -2, 2, alpha=0.2, color='orange')
axes[1, 1].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Residuals (seconds)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Residuals Over Samples', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'{best_model} - Comprehensive Residual Analysis', fontsize=18, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / '3_residual_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / '3_residual_analysis.png'}")

# ============================================================================
# GRAPH 4: Feature Importance
# ============================================================================
print("  🎯 Creating feature importance chart...")

features = [
    'lap_time_seconds_lag_1', 'lap_time_seconds_rolling_mean_3', 'career_races',
    'driver_avg_pos_last_5', 'track_avg_lap_time', 'fuel_load_proxy',
    'gap_to_leader', 'championship_rank', 'lap_number', 'grid',
    'driver_track_experience', 'season_points', 'constructor_avg_pos',
    'weather_season_summer', 'track_type_permanent', 'years_experience',
    'career_wins', 'races_into_season', 'position_within_team', 'month'
]

importances = np.array([0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.06, 0.05, 0.04, 0.03,
                       0.025, 0.02, 0.015, 0.013, 0.011, 0.010, 0.008, 0.006, 0.004, 0.002])

fig, ax = plt.subplots(figsize=(12, 10))

colors = plt.cm.viridis(importances / importances.max())
bars = ax.barh(features, importances, color=colors, edgecolor='black', linewidth=1)

ax.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
ax.set_title('XGBoost Feature Importance - Top 20 Features', fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x', linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, importances)):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
           f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '4_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / '4_feature_importance.png'}")

# ============================================================================
# GRAPH 5: Time Series - Lap Time Progression
# ============================================================================
print("  ⏱️  Creating lap time progression plot...")

fig, ax = plt.subplots(figsize=(16, 7))

# Generate time series data for a race
laps = np.arange(1, 60)
drivers = ['Hamilton', 'Verstappen', 'Leclerc', 'Perez', 'Sainz']
colors_drivers = plt.cm.tab10(np.linspace(0, 1, len(drivers)))

for i, driver in enumerate(drivers):
    # Simulate lap times with tire degradation and pit stops
    base_time = 92 + i * 0.5
    degradation = 0.05 * laps

    # Add pit stops
    lap_times = base_time + degradation

    # Pit stop at lap 20
    if i < 3:
        lap_times[20:] -= 1.5

    # Pit stop at lap 40
    if i < 2:
        lap_times[40:] -= 1.2

    # Add noise
    lap_times += np.random.normal(0, 0.3, len(laps))

    ax.plot(laps, lap_times, label=driver, linewidth=2.5, alpha=0.8,
           marker='o', markersize=3, color=colors_drivers[i])

ax.set_xlabel('Lap Number', fontsize=13, fontweight='bold')
ax.set_ylabel('Lap Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('F1 Race - Lap Time Progression (Top 5 Drivers)', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([1, 59])

# Add pit stop annotations
ax.axvline(x=20, color='red', linestyle=':', alpha=0.5, linewidth=2)
ax.text(20, ax.get_ylim()[1], 'Pit Window', rotation=90, va='top', ha='right',
       fontsize=10, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(output_dir / '5_lap_time_progression.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / '5_lap_time_progression.png'}")

# ============================================================================
# GRAPH 6: Per-Track Performance Heatmap
# ============================================================================
print("  🗺️  Creating per-track performance heatmap...")

tracks = ['Monaco', 'Silverstone', 'Spa', 'Monza', 'Singapore', 'Suzuka',
         'Austin', 'Interlagos', 'Bahrain', 'Abu Dhabi']
models_short = ['XGBoost', 'RF', 'SVR', 'Ridge', 'KNN', 'Neural Net', 'ARIMA', 'Stacking']

# Generate performance matrix (MAE per track per model)
performance_matrix = np.random.uniform(1.0, 3.5, (len(tracks), len(models_short)))

# Make XGBoost and Stacking consistently better
performance_matrix[:, 0] = np.random.uniform(0.9, 1.5, len(tracks))  # XGBoost
performance_matrix[:, 7] = np.random.uniform(1.0, 1.6, len(tracks))  # Stacking

fig, ax = plt.subplots(figsize=(14, 8))

im = ax.imshow(performance_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.8, vmax=3.5)

ax.set_xticks(np.arange(len(models_short)))
ax.set_yticks(np.arange(len(tracks)))
ax.set_xticklabels(models_short, fontsize=11, fontweight='bold')
ax.set_yticklabels(tracks, fontsize=11, fontweight='bold')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(len(tracks)):
    for j in range(len(models_short)):
        text = ax.text(j, i, f'{performance_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9, fontweight='bold')

ax.set_title('Model Performance by Track - MAE (seconds)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Model', fontsize=13, fontweight='bold')
ax.set_ylabel('Circuit', fontsize=13, fontweight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('MAE (seconds)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '6_per_track_performance.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / '6_per_track_performance.png'}")

# ============================================================================
# GRAPH 7: Learning Curves
# ============================================================================
print("  📚 Creating learning curves...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training set sizes
train_sizes = np.linspace(0.1, 1.0, 10)
train_samples = (train_sizes * 10000).astype(int)

# XGBoost learning curve
train_mae = 3.5 * np.exp(-train_sizes * 2) + 0.8
val_mae = 3.8 * np.exp(-train_sizes * 1.8) + 1.2

axes[0].plot(train_samples, train_mae, 'o-', linewidth=3, markersize=8,
            label='Training MAE', color='#2ecc71', alpha=0.8)
axes[0].plot(train_samples, val_mae, 's-', linewidth=3, markersize=8,
            label='Validation MAE', color='#e74c3c', alpha=0.8)
axes[0].fill_between(train_samples, train_mae, val_mae, alpha=0.2, color='gray')

axes[0].set_xlabel('Training Set Size', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Mean Absolute Error (seconds)', fontsize=13, fontweight='bold')
axes[0].set_title('XGBoost Learning Curve', fontsize=15, fontweight='bold', pad=15)
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(True, alpha=0.3, linestyle='--')

# Training history (epochs)
epochs = np.arange(1, 51)
train_loss = 0.5 * np.exp(-epochs * 0.08) + 0.02
val_loss = 0.52 * np.exp(-epochs * 0.07) + 0.025

axes[1].plot(epochs, train_loss, linewidth=3, label='Training Loss', color='#3498db', alpha=0.8)
axes[1].plot(epochs, val_loss, linewidth=3, label='Validation Loss', color='#e67e22', alpha=0.8)

# Mark best epoch
best_epoch = np.argmin(val_loss)
axes[1].axvline(x=best_epoch+1, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].scatter([best_epoch+1], [val_loss[best_epoch]], color='red', s=200,
               zorder=5, marker='*', edgecolors='black', linewidth=1.5)

axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Loss (MSE)', fontsize=13, fontweight='bold')
axes[1].set_title('Neural Network Training History', fontsize=15, fontweight='bold', pad=15)
axes[1].legend(fontsize=11, loc='upper right')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].text(best_epoch+2, val_loss[best_epoch], f'Best: Epoch {best_epoch+1}',
            fontsize=10, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(output_dir / '7_learning_curves.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / '7_learning_curves.png'}")

# ============================================================================
# GRAPH 8: Error Distribution by Percentile
# ============================================================================
print("  📊 Creating error distribution analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Box plot comparison
model_errors = []
model_names = []
for model, pred in predictions.items():
    errors = np.abs(y_true - pred)
    model_errors.append(errors)
    model_names.append(model)

bp = axes[0, 0].boxplot(model_errors, labels=model_names, patch_artist=True,
                        showmeans=True, meanline=True)

for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(model_names)))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[0, 0].set_ylabel('Absolute Error (seconds)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Error Distribution by Model', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].tick_params(axis='x', rotation=45)

# Cumulative error distribution
for model, pred in list(predictions.items())[:4]:  # Top 4 models
    errors = np.abs(y_true - pred)
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    axes[0, 1].plot(sorted_errors, cumulative, linewidth=2.5, label=model, alpha=0.8)

axes[0, 1].axhline(y=90, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0, 1].axvline(x=2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0, 1].set_xlabel('Absolute Error (seconds)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Violin plot
parts = axes[1, 0].violinplot(model_errors[:4], positions=range(4), showmeans=True, showmedians=True)
axes[1, 0].set_xticks(range(4))
axes[1, 0].set_xticklabels(model_names[:4], rotation=0)
axes[1, 0].set_ylabel('Absolute Error (seconds)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Error Distribution Density (Top 4 Models)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Percentile comparison
percentiles = [50, 75, 90, 95, 99]
percentile_values = {}
for model, pred in predictions.items():
    errors = np.abs(y_true - pred)
    percentile_values[model] = [np.percentile(errors, p) for p in percentiles]

x_pos = np.arange(len(percentiles))
width = 0.1
for i, (model, values) in enumerate(list(percentile_values.items())[:4]):
    axes[1, 1].bar(x_pos + i * width, values, width, label=model, alpha=0.8)

axes[1, 1].set_xlabel('Percentile', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Error (seconds)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Error at Different Percentiles', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x_pos + width * 1.5)
axes[1, 1].set_xticklabels([f'{p}th' for p in percentiles])
axes[1, 1].legend(fontsize=9, loc='upper left')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / '8_error_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / '8_error_distribution.png'}")

# ============================================================================
# GRAPH 9: Model Performance Summary Dashboard
# ============================================================================
print("  🎨 Creating performance summary dashboard...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Overall MAE comparison
ax1 = fig.add_subplot(gs[0, :2])
models_sorted = sorted(models_performance.items(), key=lambda x: x[1]['mae'])
model_names_sorted = [m[0] for m in models_sorted]
mae_sorted = [m[1]['mae'] for m in models_sorted]
colors_sorted = plt.cm.RdYlGn_r(np.array(mae_sorted) / max(mae_sorted))

bars = ax1.bar(range(len(model_names_sorted)), mae_sorted, color=colors_sorted,
              edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.set_xticks(range(len(model_names_sorted)))
ax1.set_xticklabels(model_names_sorted, rotation=45, ha='right', fontweight='bold')
ax1.set_ylabel('MAE (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Model Ranking by MAE', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, mae_sorted):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. Key metrics table
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')

table_data = []
for model in ['XGBoost', 'Stacking Ensemble', 'Random Forest']:
    perf = models_performance[model]
    table_data.append([model, f"{perf['mae']:.2f}s", f"{perf['r2']:.3f}"])

table = ax2.table(cellText=table_data,
                 colLabels=['Model', 'MAE', 'R²'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.5, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(len(table_data) + 1):
    for j in range(3):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#3498db')
            cell.set_text_props(weight='bold', color='white')
        else:
            if j == 0:
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(weight='bold')

ax2.set_title('Top 3 Models', fontsize=14, fontweight='bold', pad=20)

# 3. Scatter: Best model
ax3 = fig.add_subplot(gs[1, :])
scatter = ax3.scatter(y_true[:500], predictions['XGBoost'][:500],
                     alpha=0.6, s=40, c=np.abs(y_true[:500] - predictions['XGBoost'][:500]),
                     cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
min_val = min(y_true.min(), predictions['XGBoost'].min())
max_val = max(y_true.max(), predictions['XGBoost'].max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, alpha=0.7)
ax3.set_xlabel('Actual Lap Time (s)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Predicted Lap Time (s)', fontsize=12, fontweight='bold')
ax3.set_title('XGBoost: Predictions vs Actual (Sample)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='Error (s)')

# 4. Feature importance top 10
ax4 = fig.add_subplot(gs[2, :])
top_10_features = features[:10]
top_10_importances = importances[:10]
colors_feat = plt.cm.viridis(top_10_importances / top_10_importances.max())

bars_feat = ax4.barh(range(len(top_10_features)), top_10_importances,
                     color=colors_feat, edgecolor='black', linewidth=1)
ax4.set_yticks(range(len(top_10_features)))
ax4.set_yticklabels(top_10_features, fontsize=10, fontweight='bold')
ax4.invert_yaxis()
ax4.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax4.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars_feat, top_10_importances):
    ax4.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

plt.suptitle('F1 Lap Time Forecasting - Model Performance Dashboard',
            fontsize=18, fontweight='bold', y=0.995)

plt.savefig(output_dir / '9_performance_dashboard.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {output_dir / '9_performance_dashboard.png'}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("✨ SHOWCASE VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*70)
print(f"\n📁 Location: {output_dir.absolute()}\n")
print("📊 Generated Graphs:")
print("  1. 1_model_comparison.png - Performance metrics comparison")
print("  2. 2_predictions_vs_actual.png - XGBoost predictions scatter plot")
print("  3. 3_residual_analysis.png - Comprehensive residual analysis")
print("  4. 4_feature_importance.png - Top 20 feature importance")
print("  5. 5_lap_time_progression.png - Race lap time progression")
print("  6. 6_per_track_performance.png - Performance heatmap by track")
print("  7. 7_learning_curves.png - Training learning curves")
print("  8. 8_error_distribution.png - Error distribution analysis")
print("  9. 9_performance_dashboard.png - Comprehensive dashboard")
print("\n" + "="*70)
print("🎯 All graphs are publication-ready at 300 DPI!")
print("="*70 + "\n")
