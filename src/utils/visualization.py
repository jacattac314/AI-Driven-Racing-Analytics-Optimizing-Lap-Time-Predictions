"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual",
                               save_path: Optional[str] = None):
    """
    Plot predicted vs actual values with perfect prediction line.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Lap Time (seconds)', fontsize=12)
    ax.set_ylabel('Predicted Lap Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Plot",
                  save_path: Optional[str] = None):
    """
    Plot residuals (prediction errors).

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Residual scatter plot
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Lap Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Residuals (seconds)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Residual histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals (seconds)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_feature_importance(feature_names: List[str], importances,
                           top_n: int = 20, title="Feature Importance",
                           save_path: Optional[str] = None):
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to display
        title: Plot title
        save_path: Path to save figure (optional)
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    ax.barh(range(len(top_features)), top_importances, align='center')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_model_comparison(results: pd.DataFrame, metric: str = 'MAE',
                         save_path: Optional[str] = None):
    """
    Plot comparison of different models.

    Args:
        results: DataFrame with model names and metrics
        metric: Metric to compare (MAE, RMSE, etc.)
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Handle both 'model' and 'Model' column names
    model_col = 'model' if 'model' in results.columns else 'Model'
    models = results[model_col]
    values = results[metric]

    bars = ax.bar(models, values, edgecolor='black', alpha=0.7)

    # Color bars by performance (lower is better for MAE/RMSE)
    colors = plt.cm.RdYlGn_r(values / values.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x-labels if many models
    if len(models) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_time_series(data: pd.DataFrame, x_col: str, y_col: str,
                    hue_col: Optional[str] = None, title: str = "Time Series",
                    save_path: Optional[str] = None):
    """
    Plot time series data.

    Args:
        data: DataFrame with time series data
        x_col: Column name for x-axis (e.g., lap number)
        y_col: Column name for y-axis (e.g., lap time)
        hue_col: Column for grouping (optional)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    if hue_col:
        for group in data[hue_col].unique():
            group_data = data[data[hue_col] == group]
            ax.plot(group_data[x_col], group_data[y_col],
                   label=group, alpha=0.7, linewidth=1.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.plot(data[x_col], data[y_col], linewidth=1.5)

    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
