"""Feature selection using RFECV"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from src.utils.logging_utils import setup_logger


class FeatureSelector:
    """Select important features using RFECV"""

    def __init__(self, config: dict = None):
        self.logger = setup_logger(__name__, 'logs/feature_selector.log')

        if config:
            self.min_features = config['feature_selection']['min_features_to_select']
            self.cv_folds = config['feature_selection']['cv_folds']
            self.step = config['feature_selection']['step']
        else:
            self.min_features = 10
            self.cv_folds = 5
            self.step = 1

        self.selected_features = None
        self.feature_importances = None

    def select_features_rfecv(self, X: pd.DataFrame, y: pd.Series,
                             estimator=None) -> list:
        """
        Select features using Recursive Feature Elimination with CV.

        Args:
            X: Feature matrix
            y: Target variable
            estimator: Sklearn estimator (default: RandomForestRegressor)

        Returns:
            List of selected feature names
        """
        self.logger.info(f"Starting RFECV with {len(X.columns)} features")

        if estimator is None:
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            )

        # Perform RFECV
        rfecv = RFECV(
            estimator=estimator,
            step=self.step,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            min_features_to_select=self.min_features,
            n_jobs=-1
        )

        rfecv.fit(X, y)

        # Get selected features
        selected_features = X.columns[rfecv.support_].tolist()
        self.selected_features = selected_features

        self.logger.info(
            f"Selected {len(selected_features)} features using RFECV"
        )
        self.logger.info(f"Optimal number of features: {rfecv.n_features_}")

        return selected_features

    def select_features_by_importance(self, X: pd.DataFrame, y: pd.Series,
                                     top_n: int = None,
                                     threshold: float = 0.01) -> list:
        """
        Select features based on importance from tree-based model.

        Args:
            X: Feature matrix
            y: Target variable
            top_n: Select top N features (optional)
            threshold: Minimum importance threshold

        Returns:
            List of selected feature names
        """
        self.logger.info("Selecting features by importance")

        # Train random forest
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)

        # Get feature importances
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        self.feature_importances = feature_importance_df

        # Select features
        if top_n:
            selected_features = feature_importance_df.head(top_n)['feature'].tolist()
        else:
            selected_features = feature_importance_df[
                feature_importance_df['importance'] >= threshold
            ]['feature'].tolist()

        self.logger.info(f"Selected {len(selected_features)} important features")

        return selected_features

    def remove_correlated_features(self, X: pd.DataFrame,
                                   threshold: float = 0.95) -> list:
        """
        Remove highly correlated features.

        Args:
            X: Feature matrix
            threshold: Correlation threshold

        Returns:
            List of features to keep
        """
        self.logger.info("Removing correlated features")

        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation greater than threshold
        to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        features_to_keep = [col for col in X.columns if col not in to_drop]

        self.logger.info(
            f"Removed {len(to_drop)} correlated features, "
            f"keeping {len(features_to_keep)} features"
        )

        return features_to_keep

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: str = 'rfecv') -> list:
        """
        Select features using specified method.

        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('rfecv', 'importance', 'correlation')

        Returns:
            List of selected feature names
        """
        if method == 'rfecv':
            return self.select_features_rfecv(X, y)
        elif method == 'importance':
            return self.select_features_by_importance(X, y)
        elif method == 'correlation':
            return self.remove_correlated_features(X)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importances from last selection.

        Returns:
            DataFrame with feature importances
        """
        if self.feature_importances is None:
            self.logger.warning("No feature importances available")
            return pd.DataFrame()

        return self.feature_importances
