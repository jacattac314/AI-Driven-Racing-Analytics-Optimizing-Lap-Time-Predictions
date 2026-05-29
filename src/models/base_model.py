"""Abstract base class for all models"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any
import joblib
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for all prediction models"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize base model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training arguments
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        pass

    def evaluate(self, X, y) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        predictions = self.predict(X)

        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        mape = np.mean(np.abs((y - predictions) / y)) * 100

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }

        return metrics

    def save_model(self, filepath: str):
        """
        Save model to disk.

        WARNING: joblib use pickle for serialization. Only save data you trust.

        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str):
        """
        Load model from disk.

        WARNING: joblib uses pickle for serialization and is not secure.
        Only load data you trust. Never load data that could have come
        from an untrusted source, or that could have been tampered with.

        Args:
            filepath: Path to load model from
        """
        self.model = joblib.load(filepath)
        self.is_trained = True

    def get_model(self):
        """Get the underlying model object."""
        return self.model

    def get_feature_importance(self):
        """
        Get feature importance (if supported by model).

        Returns:
            Feature importance array or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        else:
            return None
