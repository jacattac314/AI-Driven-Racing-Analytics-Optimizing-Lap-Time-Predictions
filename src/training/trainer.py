"""Main training orchestrator for all models"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import time

from src.models.ensemble.xgboost_model import XGBoostModel
from src.models.ensemble.random_forest_model import RandomForestModel
from src.models.ensemble.stacking_ensemble import StackingEnsembleModel
from src.models.regression.svr_model import SVRModel
from src.models.regression.knn_model import KNNModel
from src.models.regression.ridge_model import RidgeModel
from src.models.timeseries.arima_model import ARIMAModel
from src.models.neural.neural_net_model import NeuralNetModel

from src.training.cross_validator import CrossValidator
from src.training.hyperparameter_tuner import HyperparameterTuner
from src.utils.logging_utils import setup_logger
from src.utils.io_utils import save_json


class ModelTrainer:
    """Orchestrate training of all models"""

    def __init__(self, config: dict):
        """
        Initialize model trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(__name__, 'logs/trainer.log')

        self.cross_validator = CrossValidator(config)
        self.hyperparameter_tuner = HyperparameterTuner(config)

        self.models = {}
        self.training_results = {}

    def get_model_instance(self, model_type: str):
        """
        Get model instance based on type.

        Args:
            model_type: Model type name

        Returns:
            Model instance
        """
        model_config = self.config['models'].get(model_type, {})

        model_classes = {
            'xgboost': XGBoostModel,
            'random_forest': RandomForestModel,
            'svr': SVRModel,
            'knn': KNNModel,
            'ridge': RidgeModel,
            'arima': ARIMAModel,
            'neural_net': NeuralNetModel,
            'stacking': StackingEnsembleModel
        }

        model_class = model_classes.get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type}")

        return model_class(model_config)

    def _is_model_enabled(self, model_type: str) -> bool:
        """Check if model is enabled in config."""
        model_config = self.config.get('models', {}).get(model_type, {})
        return model_config.get('enabled', True)

    def _execute_training(self, model, X_train, y_train, X_val, y_val) -> float:
        """Execute training and return training time."""
        start_time = time.time()
        model.train(X_train, y_train, X_val, y_val)
        return time.time() - start_time

    def _evaluate_model(self, model, X_val, y_val) -> dict:
        """Evaluate model and return metrics."""
        if X_val is not None and y_val is not None:
            return model.evaluate(X_val, y_val)
        return {}

    def _record_results(self, model_type: str, model, training_time: float, val_metrics: dict):
        """Store model and training results."""
        self.models[model_type] = model
        self.training_results[model_type] = {
            'training_time': training_time,
            'validation_metrics': val_metrics,
            'model_params': model.params if hasattr(model, 'params') else {}
        }

    def _log_training_success(self, model_type: str, training_time: float, val_metrics: dict):
        """Log successful training with MAE if available."""
        mae = val_metrics.get('mae')
        mae_str = f"{mae:.4f}" if mae is not None else "N/A"
        self.logger.info(
            f"{model_type} training completed in {training_time:.2f}s. "
            f"Val MAE: {mae_str}"
        )

    def train_single_model(self, model_type: str, X_train, y_train,
                          X_val=None, y_val=None, tune_hyperparameters=False):
        """
        Train a single model.

        Args:
            model_type: Model type name
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Trained model instance
        """
        self.logger.info(f"Training {model_type} model")

        if not self._is_model_enabled(model_type):
            self.logger.info(f"{model_type} is disabled in config")
            return None

        model = self.get_model_instance(model_type)

        try:
            training_time = self._execute_training(model, X_train, y_train, X_val, y_val)
            val_metrics = self._evaluate_model(model, X_val, y_val)

            self._record_results(model_type, model, training_time, val_metrics)
            self._log_training_success(model_type, training_time, val_metrics)

            return model

        except Exception as e:
            self.logger.error(f"Failed to train {model_type}: {str(e)}")
            return None

    def train_all_models(self, X_train, y_train, X_val=None, y_val=None,
                        models_to_train: List[str] = None):
        """
        Train all enabled models.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            models_to_train: List of model types to train (optional)

        Returns:
            Dictionary of trained models
        """
        self.logger.info("Starting training for all models")

        # Determine which models to train
        if models_to_train is None:
            models_to_train = [
                'xgboost', 'random_forest', 'svr', 'knn',
                'ridge', 'arima', 'neural_net', 'stacking'
            ]

        # Train each model
        for model_type in models_to_train:
            self.train_single_model(
                model_type,
                X_train,
                y_train,
                X_val,
                y_val
            )

        self.logger.info(f"Training completed for {len(self.models)} models")

        return self.models

    def save_models(self, output_dir: str):
        """
        Save all trained models.

        Args:
            output_dir: Output directory for models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_type, model in self.models.items():
            model_dir = output_path / model_type
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            if model_type == 'neural_net':
                model_file = model_dir / 'model.h5'
            else:
                model_file = model_dir / 'model.pkl'

            model.save_model(str(model_file))
            self.logger.info(f"Saved {model_type} to {model_file}")

        # Save training results
        results_file = output_path / 'training_results.json'
        save_json(self.training_results, str(results_file))
        self.logger.info(f"Saved training results to {results_file}")

    def get_training_summary(self) -> pd.DataFrame:
        """
        Get summary of training results.

        Returns:
            DataFrame with training summary
        """
        summary_data = []

        for model_type, results in self.training_results.items():
            val_metrics = results.get('validation_metrics', {})

            summary_data.append({
                'model': model_type,
                'training_time': results.get('training_time', 0),
                'val_mae': val_metrics.get('mae', None),
                'val_rmse': val_metrics.get('rmse', None),
                'val_r2': val_metrics.get('r2', None),
                'val_mape': val_metrics.get('mape', None)
            })

        summary_df = pd.DataFrame(summary_data)

        if not summary_df.empty:
            summary_df = summary_df.sort_values('val_mae', ascending=True)

        return summary_df
