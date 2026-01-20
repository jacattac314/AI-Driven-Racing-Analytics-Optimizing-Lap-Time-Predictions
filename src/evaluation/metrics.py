"""Evaluation metrics for lap time prediction"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Dict


class Metrics:
    """Calculate evaluation metrics"""

    @staticmethod
    def calculate_mae(y_true, y_pred) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calculate_rmse(y_true, y_pred) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def calculate_r2(y_true, y_pred) -> float:
        """Calculate R² score."""
        return r2_score(y_true, y_pred)

    @staticmethod
    def calculate_mape(y_true, y_pred) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def calculate_all_metrics(y_true, y_pred) -> Dict[str, float]:
        """
        Calculate all regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'mae': Metrics.calculate_mae(y_true, y_pred),
            'rmse': Metrics.calculate_rmse(y_true, y_pred),
            'r2': Metrics.calculate_r2(y_true, y_pred),
            'mape': Metrics.calculate_mape(y_true, y_pred)
        }

        return metrics

    @staticmethod
    def calculate_per_group_metrics(y_true, y_pred, groups) -> pd.DataFrame:
        """
        Calculate metrics per group (e.g., per track, per driver).

        Args:
            y_true: True values
            y_pred: Predicted values
            groups: Group labels

        Returns:
            DataFrame with metrics per group
        """
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'group': groups
        })

        group_metrics = []

        for group_name in df['group'].unique():
            group_data = df[df['group'] == group_name]
            metrics = Metrics.calculate_all_metrics(
                group_data['y_true'].values,
                group_data['y_pred'].values
            )
            metrics['group'] = group_name
            metrics['count'] = len(group_data)
            group_metrics.append(metrics)

        return pd.DataFrame(group_metrics)

    @staticmethod
    def calculate_quantile_metrics(y_true, y_pred, quantiles=[0.25, 0.5, 0.75]) -> Dict:
        """
        Calculate metrics at different quantiles.

        Args:
            y_true: True values
            y_pred: Predicted values
            quantiles: List of quantiles

        Returns:
            Dictionary with quantile metrics
        """
        errors = np.abs(y_true - y_pred)

        quantile_metrics = {}
        for q in quantiles:
            quantile_metrics[f'error_q{int(q*100)}'] = np.quantile(errors, q)

        return quantile_metrics
