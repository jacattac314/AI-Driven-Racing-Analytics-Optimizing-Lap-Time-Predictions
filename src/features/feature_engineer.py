"""Main feature engineering orchestrator"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.features.driver_features import DriverFeatureEngineer
from src.features.track_features import TrackFeatureEngineer
from src.features.temporal_features import TemporalFeatureEngineer
from src.features.historical_features import HistoricalFeatureEngineer
from src.features.weather_features import WeatherFeatureEngineer
from src.features.tire_features import TireFeatureEngineer
from src.features.feature_selector import FeatureSelector
from src.utils.logging_utils import setup_logger
from src.utils.io_utils import save_dataframe


class FeatureEngineer:
    """Main feature engineering orchestrator"""

    def __init__(self, config: dict):
        """
        Initialize feature engineer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(__name__, 'logs/feature_engineer.log')

        # Initialize feature engineers
        self.driver_engineer = DriverFeatureEngineer()
        self.track_engineer = TrackFeatureEngineer()
        self.temporal_engineer = TemporalFeatureEngineer(config)
        self.historical_engineer = HistoricalFeatureEngineer(config)
        self.weather_engineer = WeatherFeatureEngineer()
        self.tire_engineer = TireFeatureEngineer()
        self.feature_selector = FeatureSelector(config)

        # Feature toggles
        self.enabled_features = config['features']['enabled_features']

    def create_all_features(self, df: pd.DataFrame,
                           target_col: str = 'lap_time_seconds') -> pd.DataFrame:
        """
        Create all features based on configuration.

        Args:
            df: Input DataFrame
            target_col: Target column name

        Returns:
            DataFrame with all features
        """
        self.logger.info("Starting feature engineering")
        original_rows = len(df)

        # Driver features
        if self.enabled_features.get('driver_features', True):
            self.logger.info("Creating driver features")
            df = self.driver_engineer.create_all_driver_features(df)

        # Track features
        if self.enabled_features.get('track_features', True):
            self.logger.info("Creating track features")
            df = self.track_engineer.create_all_track_features(df)

        # Historical features
        if self.enabled_features.get('historical_features', True):
            self.logger.info("Creating historical features")
            df = self.historical_engineer.create_all_historical_features(df)

        # Weather features
        if self.enabled_features.get('weather_features', True):
            self.logger.info("Creating weather features")
            df = self.weather_engineer.create_all_weather_features(df)

        # Tire features
        if self.enabled_features.get('tire_features', True):
            self.logger.info("Creating tire features")
            df = self.tire_engineer.create_all_tire_features(df)

        # Temporal features (should be last as they depend on target)
        if self.enabled_features.get('temporal_features', True):
            self.logger.info("Creating temporal features")
            df = self.temporal_engineer.create_all_temporal_features(df, target_col)

        final_rows = len(df)
        if final_rows != original_rows:
            self.logger.warning(
                f"Row count changed: {original_rows} -> {final_rows} "
                f"(lost {original_rows - final_rows} rows)"
            )

        self.logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")

        return df

    def get_feature_columns(self, df: pd.DataFrame,
                           exclude_cols: list = None) -> list:
        """
        Get list of feature columns (excluding metadata and target).

        Args:
            df: DataFrame with features
            exclude_cols: Additional columns to exclude

        Returns:
            List of feature column names
        """
        # Columns to exclude (metadata, identifiers, target)
        base_exclude = [
            'lap_time_seconds', 'time', 'milliseconds',
            'season', 'round', 'race_id', 'driver_race_id',
            'driver_id', 'circuit_id', 'constructor_id',
            'race_name', 'circuit_name', 'country', 'locality',
            'date', 'url', 'status', 'position_text',
            'driver_code', 'driver_number', 'date_parsed',
            'q1', 'q2', 'q3', 'fastest_lap_time'
        ]

        if exclude_cols:
            base_exclude.extend(exclude_cols)

        feature_cols = [
            col for col in df.columns
            if col not in base_exclude and not col.endswith('_id')
        ]

        self.logger.info(f"Identified {len(feature_cols)} feature columns")

        return feature_cols

    def _remove_missing_targets(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Remove rows with missing target values."""
        valid_idx = y.notna()
        return X[valid_idx], y[valid_idx]

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing and infinite values in feature DataFrame."""
        # Fill missing values in features
        # Numeric columns: fill with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().any():
                X[col].fillna(X[col].median(), inplace=True)

        # Remove columns with all NaN
        X = X.dropna(axis=1, how='all')

        # Remove columns with high missing percentage (>50%)
        missing_pct = X.isna().mean()
        cols_to_keep = missing_pct[missing_pct < 0.5].index
        X = X[cols_to_keep]

        # Fill remaining missing values with 0
        X = X.fillna(0)

        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)

        return X

    def _perform_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Perform optional feature selection."""
        self.logger.info("Performing feature selection")
        selected_features = self.feature_selector.select_features(
            X, y, method='importance'
        )
        self.logger.info(f"Selected {len(selected_features)} features")
        return X[selected_features]

    def prepare_features_for_modeling(self, df: pd.DataFrame,
                                     target_col: str = 'lap_time_seconds',
                                     select_features: bool = False) -> tuple:
        """
        Prepare features for modeling.

        Args:
            df: DataFrame with features
            target_col: Target column name
            select_features: Whether to perform feature selection

        Returns:
            Tuple of (feature_df, target_series, feature_names)
        """
        self.logger.info("Preparing features for modeling")

        # Get feature columns
        feature_cols = self.get_feature_columns(df)

        # Handle missing values in features
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Remove rows with missing target
        X, y = self._remove_missing_targets(X, y)

        # Handle missing values and infinite values
        X = self._handle_missing_values(X)

        self.logger.info(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")

        # Feature selection (optional)
        if select_features:
            X = self._perform_feature_selection(X, y)

        return X, y, X.columns.tolist()

    def save_features(self, df: pd.DataFrame, output_path: str):
        """
        Save engineered features.

        Args:
            df: DataFrame with features
            output_path: Output file path
        """
        save_dataframe(df, output_path)
        self.logger.info(f"Features saved to {output_path}")

    def engineer_and_save(self, df: pd.DataFrame,
                         output_path: str,
                         target_col: str = 'lap_time_seconds'):
        """
        Engineer features and save to file.

        Args:
            df: Input DataFrame
            output_path: Output file path
            target_col: Target column name
        """
        # Create features
        df_features = self.create_all_features(df, target_col)

        # Save
        self.save_features(df_features, output_path)

        return df_features
