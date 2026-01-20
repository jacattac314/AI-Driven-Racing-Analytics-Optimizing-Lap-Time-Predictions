"""Random Forest model implementation"""

from sklearn.ensemble import RandomForestRegressor
from src.models.base_model import BaseModel
from src.utils.logging_utils import setup_logger


class RandomForestModel(BaseModel):
    """Random Forest regression model"""

    def __init__(self, config: dict = None):
        """
        Initialize Random Forest model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.logger = setup_logger(__name__, 'logs/random_forest_model.log')

        # Default parameters
        self.params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }

        # Update with config if provided
        if config and 'params' in config:
            self.params.update(config['params'])

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, not used for RF)
            y_val: Validation target (optional, not used for RF)
            **kwargs: Additional training arguments
        """
        self.logger.info("Training Random Forest model")

        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)

        self.is_trained = True

        # Log OOB score if available
        if hasattr(self.model, 'oob_score_'):
            self.logger.info(f"OOB Score: {self.model.oob_score_:.4f}")

        self.logger.info("Random Forest training completed")

    def predict(self, X):
        """
        Make predictions using Random Forest.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)

    def get_feature_importance(self):
        """Get feature importance from Random Forest."""
        if self.is_trained:
            return self.model.feature_importances_
        return None
