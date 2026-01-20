"""K-Nearest Neighbors model implementation"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseModel
from src.utils.logging_utils import setup_logger


class KNNModel(BaseModel):
    """K-Nearest Neighbors regression model"""

    def __init__(self, config: dict = None):
        """
        Initialize KNN model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.logger = setup_logger(__name__, 'logs/knn_model.log')

        # Default parameters
        self.params = {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'euclidean',
            'n_jobs': -1
        }

        # Update with config if provided
        if config and 'params' in config:
            self.params.update(config['params'])

        # Scaler for feature normalization (important for KNN)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train KNN model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training arguments
        """
        self.logger.info("Training KNN model")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model = KNeighborsRegressor(**self.params)
        self.model.fit(X_train_scaled, y_train)

        self.is_trained = True
        self.logger.info("KNN training completed")

    def predict(self, X):
        """
        Make predictions using KNN.

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
