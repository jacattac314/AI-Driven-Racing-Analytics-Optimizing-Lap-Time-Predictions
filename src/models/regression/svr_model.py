"""Support Vector Regression model implementation"""

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseModel
from src.utils.logging_utils import setup_logger


class SVRModel(BaseModel):
    """Support Vector Regression model"""

    def __init__(self, config: dict = None):
        """
        Initialize SVR model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.logger = setup_logger(__name__, 'logs/svr_model.log')

        # Default parameters
        self.params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        }

        # Update with config if provided
        if config and 'params' in config:
            self.params.update(config['params'])

        # Scaler for feature normalization (important for SVR)
        self.scaler = StandardScaler()

    def _create_model(self):
        """Create SVR model instance."""
        return SVR(**self.params)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train SVR model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training arguments
        """
        self.logger.info("Training SVR model")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)

        self.is_trained = True
        self.logger.info("SVR training completed")

    def predict(self, X):
        """
        Make predictions using SVR.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Scale features
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)
