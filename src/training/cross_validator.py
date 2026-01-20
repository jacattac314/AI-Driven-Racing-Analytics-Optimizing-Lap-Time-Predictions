"""Cross-validation strategies for time-series data"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from src.utils.logging_utils import setup_logger


class CrossValidator:
    """Time-series aware cross-validation"""

    def __init__(self, config: dict):
        """
        Initialize cross-validator.

        Args:
            config: Configuration dictionary
        """
        self.n_folds = config['training']['cv']['n_folds']
        self.strategy = config['training']['cv']['strategy']
        self.logger = setup_logger(__name__, 'logs/cross_validator.log')

    def time_series_split(self, X, y):
        """
        Perform time-series split (expanding window).

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            TimeSeriesSplit object
        """
        self.logger.info(f"Creating TimeSeriesSplit with {self.n_folds} folds")
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        return tscv

    def cross_validate_model(self, model, X, y, scoring='neg_mean_absolute_error'):
        """
        Perform cross-validation on a model.

        Args:
            model: Model instance with fit/predict methods
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric

        Returns:
            Cross-validation scores
        """
        self.logger.info("Performing cross-validation")

        # Get cross-validation strategy
        cv = self.time_series_split(X, y)

        # Perform cross-validation
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        self.logger.info(
            f"CV Scores: {scores}, Mean: {scores.mean():.4f}, Std: {scores.std():.4f}"
        )

        return scores

    def get_cv_folds(self, X, y):
        """
        Get train/validation indices for each fold.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            List of (train_idx, val_idx) tuples
        """
        cv = self.time_series_split(X, y)
        folds = []

        for fold_num, (train_idx, val_idx) in enumerate(cv.split(X)):
            self.logger.info(
                f"Fold {fold_num + 1}: Train size={len(train_idx)}, Val size={len(val_idx)}"
            )
            folds.append((train_idx, val_idx))

        return folds
