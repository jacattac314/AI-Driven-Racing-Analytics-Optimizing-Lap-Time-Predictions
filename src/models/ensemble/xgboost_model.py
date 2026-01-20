"""XGBoost model implementation"""

import xgboost as xgb
from src.models.base_model import BaseModel
from src.utils.logging_utils import setup_logger


class XGBoostModel(BaseModel):
    """XGBoost regression model"""

    def __init__(self, config: dict = None):
        """
        Initialize XGBoost model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.logger = setup_logger(__name__, 'logs/xgboost_model.log')

        # Default parameters
        self.params = {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }

        # Update with config if provided
        if config and 'params' in config:
            self.params.update(config['params'])

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training arguments
        """
        self.logger.info("Training XGBoost model")

        # Early stopping if validation set provided
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X_train, y_train)

        self.is_trained = True
        self.logger.info("XGBoost training completed")

    def predict(self, X):
        """
        Make predictions using XGBoost.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)

    def get_feature_importance(self):
        """Get feature importance from XGBoost."""
        if self.is_trained:
            return self.model.feature_importances_
        return None
