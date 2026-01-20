"""Stacking ensemble model implementation"""

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from src.models.base_model import BaseModel
from src.models.ensemble.xgboost_model import XGBoostModel
from src.models.ensemble.random_forest_model import RandomForestModel
from src.models.regression.svr_model import SVRModel
from src.models.regression.ridge_model import RidgeModel
from src.utils.logging_utils import setup_logger


class StackingEnsembleModel(BaseModel):
    """Stacking ensemble of multiple models"""

    def __init__(self, config: dict = None):
        """
        Initialize Stacking ensemble.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.logger = setup_logger(__name__, 'logs/stacking_ensemble.log')

        # Base models
        self.base_models = []

        # Meta-learner (default: Ridge)
        self.meta_learner = Ridge(alpha=1.0, random_state=42)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train stacking ensemble.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training arguments
        """
        self.logger.info("Training Stacking Ensemble model")

        # Initialize base models
        xgb_config = {'params': {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 6}}
        rf_config = {'params': {'n_estimators': 50, 'max_depth': 15}}
        svr_config = {'params': {'kernel': 'rbf', 'C': 1.0}}
        ridge_config = {'params': {'alpha': 1.0}}

        xgb_model = XGBoostModel(xgb_config)
        rf_model = RandomForestModel(rf_config)
        svr_model = SVRModel(svr_config)
        ridge_model = RidgeModel(ridge_config)

        # Create base estimators list
        base_estimators = [
            ('xgboost', xgb_model.model if xgb_model.model else xgb_model._create_model()),
            ('random_forest', rf_model.model if rf_model.model else rf_model._create_model()),
            ('svr', svr_model.model if svr_model.model else svr_model._create_model()),
            ('ridge', ridge_model.model if ridge_model.model else ridge_model._create_model())
        ]

        # Create stacking regressor
        self.model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=self.meta_learner,
            cv=5,
            n_jobs=-1
        )

        # Train
        self.model.fit(X_train, y_train)

        self.is_trained = True
        self.logger.info("Stacking Ensemble training completed")

    def predict(self, X):
        """
        Make predictions using stacking ensemble.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)
