"""Ridge Regression model implementation"""

from sklearn.linear_model import Ridge
from src.models.base_model import BaseModel
from src.utils.logging_utils import setup_logger


class RidgeModel(BaseModel):
    """Ridge Regression model"""

    def __init__(self, config: dict = None):
        """
        Initialize Ridge model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.logger = setup_logger(__name__, 'logs/ridge_model.log')

        # Default parameters
        self.params = {
            'alpha': 1.0,
            'random_state': 42
        }

        # Update with config if provided
        if config and 'params' in config:
            self.params.update(config['params'])

    def _create_model(self):
        """Create Ridge model instance."""
        return Ridge(**self.params)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train Ridge model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training arguments
        """
        self.logger.info("Training Ridge model")

        # Train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        self.is_trained = True
        self.logger.info("Ridge training completed")

    def predict(self, X):
        """
        Make predictions using Ridge.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)

    def get_feature_importance(self):
        """Get feature coefficients from Ridge."""
        if self.is_trained:
            import numpy as np
            return np.abs(self.model.coef_)
        return None
