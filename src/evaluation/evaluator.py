"""Model evaluation orchestrator"""

import pandas as pd
from pathlib import Path
from typing import Dict

from src.evaluation.metrics import Metrics
from src.utils.logging_utils import setup_logger
from src.utils.io_utils import save_json, save_dataframe
from src.utils.visualization import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_model_comparison
)


class ModelEvaluator:
    """Evaluate and compare multiple models"""

    def __init__(self):
        self.logger = setup_logger(__name__, 'logs/evaluator.log')
        self.evaluation_results = {}

    def evaluate_model(self, model, model_name: str, X_test, y_test,
                      additional_data: Dict = None) -> Dict:
        """
        Evaluate a single model.

        Args:
            model: Trained model instance
            model_name: Model name
            X_test: Test features
            y_test: Test target
            additional_data: Additional data for grouped metrics (optional)

        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating {model_name}")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = Metrics.calculate_all_metrics(y_test, y_pred)

        # Add predictions to results
        results = {
            'model': model_name,
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test
        }

        # Calculate per-group metrics if additional data provided
        if additional_data:
            if 'track' in additional_data:
                track_metrics = Metrics.calculate_per_group_metrics(
                    y_test, y_pred, additional_data['track']
                )
                results['track_metrics'] = track_metrics

            if 'driver' in additional_data:
                driver_metrics = Metrics.calculate_per_group_metrics(
                    y_test, y_pred, additional_data['driver']
                )
                results['driver_metrics'] = driver_metrics

        # Store results
        self.evaluation_results[model_name] = results

        self.logger.info(
            f"{model_name} - MAE: {metrics['mae']:.4f}, "
            f"RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}"
        )

        return results

    def evaluate_all_models(self, models: Dict, X_test, y_test,
                           additional_data: Dict = None) -> pd.DataFrame:
        """
        Evaluate all models.

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            additional_data: Additional data for grouped metrics (optional)

        Returns:
            DataFrame with comparison results
        """
        self.logger.info(f"Evaluating {len(models)} models")

        comparison_data = []

        for model_name, model in models.items():
            try:
                results = self.evaluate_model(
                    model, model_name, X_test, y_test, additional_data
                )

                # Add to comparison
                metrics = results['metrics']
                comparison_data.append({
                    'model': model_name,
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'R2': metrics['r2'],
                    'MAPE': metrics['mape']
                })

            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('MAE', ascending=True)

        return comparison_df

    def generate_visualizations(self, output_dir: str):
        """
        Generate evaluation visualizations.

        Args:
            output_dir: Output directory for plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating evaluation visualizations")

        for model_name, results in self.evaluation_results.items():
            y_true = results['actual']
            y_pred = results['predictions']

            # Predictions vs Actual
            plot_predictions_vs_actual(
                y_true,
                y_pred,
                title=f"{model_name} - Predictions vs Actual",
                save_path=str(output_path / f"{model_name}_predictions.png")
            )

            # Residuals
            plot_residuals(
                y_true,
                y_pred,
                title=f"{model_name} - Residual Analysis",
                save_path=str(output_path / f"{model_name}_residuals.png")
            )

        self.logger.info(f"Visualizations saved to {output_dir}")

    def generate_comparison_plot(self, comparison_df: pd.DataFrame, output_path: str):
        """
        Generate model comparison plot.

        Args:
            comparison_df: DataFrame with model comparison
            output_path: Path to save plot
        """
        plot_model_comparison(
            comparison_df,
            metric='MAE',
            save_path=output_path
        )

        self.logger.info(f"Comparison plot saved to {output_path}")

    def save_results(self, output_dir: str):
        """
        Save evaluation results.

        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, results in self.evaluation_results.items():
            model_dir = output_path / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics
            metrics_file = model_dir / 'metrics.json'
            save_json(results['metrics'], str(metrics_file))

            # Save predictions
            predictions_df = pd.DataFrame({
                'actual': results['actual'],
                'predicted': results['predictions'],
                'error': results['actual'] - results['predictions']
            })
            save_dataframe(predictions_df, str(model_dir / 'predictions.csv'))

            # Save per-track metrics if available
            if 'track_metrics' in results:
                save_dataframe(
                    results['track_metrics'],
                    str(model_dir / 'track_metrics.csv')
                )

            # Save per-driver metrics if available
            if 'driver_metrics' in results:
                save_dataframe(
                    results['driver_metrics'],
                    str(model_dir / 'driver_metrics.csv')
                )

        self.logger.info(f"Results saved to {output_dir}")

    def get_best_model(self, metric: str = 'mae') -> str:
        """
        Get best performing model.

        Args:
            metric: Metric to use for comparison

        Returns:
            Best model name
        """
        best_model = None
        best_score = float('inf')

        for model_name, results in self.evaluation_results.items():
            score = results['metrics'].get(metric, float('inf'))

            if score < best_score:
                best_score = score
                best_model = model_name

        self.logger.info(f"Best model: {best_model} with {metric}={best_score:.4f}")

        return best_model
