"""ARIMA time-series model implementation"""

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.models.base_model import BaseModel
from src.utils.logging_utils import setup_logger


class ARIMAModel(BaseModel):
    """ARIMA time-series regression model"""

    def __init__(self, config: dict = None):
        """
        Initialize ARIMA model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.logger = setup_logger(__name__, 'logs/arima_model.log')

        # Default parameters
        self.params = {
            'seasonal': True,
            'm': 10,  # Seasonality period (laps)
            'max_p': 5,
            'max_q': 5,
            'max_d': 2
        }

        # Update with config if provided
        if config and 'params' in config:
            self.params.update(config['params'])

        self.order = None
        self.seasonal_order = None

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train ARIMA model.

        Note: ARIMA is primarily for time-series, but we can use it
        with exogenous variables (ARIMAX/SARIMAX).

        Args:
            X_train: Training features (exogenous variables)
            y_train: Training target (time series)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training arguments
        """
        self.logger.info("Training ARIMA model")

        try:
            # Use auto_arima to find optimal parameters
            self.model = auto_arima(
                y_train,
                exogenous=X_train if X_train.shape[1] > 0 else None,
                seasonal=self.params['seasonal'],
                m=self.params['m'],
                max_p=self.params['max_p'],
                max_q=self.params['max_q'],
                max_d=self.params['max_d'],
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            self.order = self.model.order
            self.seasonal_order = self.model.seasonal_order

            self.is_trained = True
            self.logger.info(f"ARIMA training completed. Order: {self.order}, Seasonal: {self.seasonal_order}")

        except Exception as e:
            self.logger.error(f"ARIMA training failed: {str(e)}")
            # Fallback to simple ARIMA(1,1,1) if auto_arima fails
            self.logger.info("Falling back to ARIMA(1,1,1)")
            self.model = SARIMAX(
                y_train,
                exog=X_train if X_train.shape[1] > 0 else None,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0)
            ).fit(disp=False)
            self.is_trained = True

    def predict(self, X):
        """
        Make predictions using ARIMA.

        Args:
            X: Feature matrix (exogenous variables)

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            n_periods = len(X)
            exog = X if X.shape[1] > 0 else None

            predictions = self.model.predict(n_periods=n_periods, exogenous=exog)
            return predictions

        except Exception as e:
            self.logger.error(f"ARIMA prediction failed: {str(e)}")
            # Return mean as fallback
            return np.full(len(X), np.mean(self.model.fittedvalues))
