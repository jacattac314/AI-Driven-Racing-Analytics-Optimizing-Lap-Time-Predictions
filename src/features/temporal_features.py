"""Temporal and time-series feature engineering"""

import pandas as pd
import numpy as np


class TemporalFeatureEngineer:
    """Generate time-series features"""

    def __init__(self, config: dict = None):
        if config:
            self.rolling_windows = config['features']['temporal']['rolling_windows']
            self.lag_features = config['features']['temporal']['lag_features']
        else:
            self.rolling_windows = [3, 5, 10]
            self.lag_features = [1, 2, 3]

    def create_lap_number_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on lap number.

        Args:
            df: DataFrame with lap information

        Returns:
            DataFrame with lap number features
        """
        df = df.copy()

        if 'lap' in df.columns:
            # Lap number
            df['lap_number'] = df['lap']

            # Fuel load proxy (inversely related to lap number)
            # Assuming max 70 laps
            max_laps = df.groupby(['season', 'round'])['lap'].transform('max')
            df['fuel_load_proxy'] = (max_laps - df['lap']) / max_laps

            # Race progression (0-1)
            df['race_progress'] = df['lap'] / max_laps

        return df

    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'lap_time_seconds') -> pd.DataFrame:
        """
        Create lag features for target variable.

        Args:
            df: DataFrame with target column
            target_col: Column to create lags for

        Returns:
            DataFrame with lag features
        """
        df = df.copy()

        if target_col not in df.columns:
            return df

        df = df.sort_values(['driver_id', 'season', 'round', 'lap'])

        for lag in self.lag_features:
            df[f'{target_col}_lag_{lag}'] = df.groupby(['driver_id', 'season', 'round'])[target_col].shift(lag)

        return df

    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'lap_time_seconds') -> pd.DataFrame:
        """
        Create rolling window features.

        Args:
            df: DataFrame with target column
            target_col: Column to create rolling features for

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()

        if target_col not in df.columns:
            return df

        df = df.sort_values(['driver_id', 'season', 'round', 'lap'])

        for window in self.rolling_windows:
            # Rolling mean
            df[f'{target_col}_rolling_mean_{window}'] = df.groupby(['driver_id', 'season', 'round'])[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

            # Rolling std
            df[f'{target_col}_rolling_std_{window}'] = df.groupby(['driver_id', 'season', 'round'])[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

            # Rolling min
            df[f'{target_col}_rolling_min_{window}'] = df.groupby(['driver_id', 'season', 'round'])[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )

            # Rolling max
            df[f'{target_col}_rolling_max_{window}'] = df.groupby(['driver_id', 'season', 'round'])[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )

        return df

    def create_delta_features(self, df: pd.DataFrame, target_col: str = 'lap_time_seconds') -> pd.DataFrame:
        """
        Create delta (change) features.

        Args:
            df: DataFrame with target column
            target_col: Column to create deltas for

        Returns:
            DataFrame with delta features
        """
        df = df.copy()

        if target_col not in df.columns:
            return df

        df = df.sort_values(['driver_id', 'season', 'round', 'lap'])

        # Delta from previous lap
        df[f'{target_col}_delta'] = df.groupby(['driver_id', 'season', 'round'])[target_col].diff()

        # Delta from best lap so far
        df[f'{target_col}_delta_from_best'] = df[target_col] - df.groupby(['driver_id', 'season', 'round'])[target_col].cummin()

        return df

    def create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create position-related time features.

        Args:
            df: DataFrame with position information

        Returns:
            DataFrame with position features
        """
        df = df.copy()

        if 'position' in df.columns:
            # Current position (by lap)
            df['current_position'] = pd.to_numeric(df['position'], errors='coerce')

            # Position change from previous lap
            df['position_change'] = df.groupby(['driver_id', 'season', 'round'])['current_position'].diff()

        # Gap to leader (simplified - actual gap would need lap time cumsum)
        if 'lap_time_seconds' in df.columns:
            df['cumulative_time'] = df.groupby(['driver_id', 'season', 'round'])['lap_time_seconds'].cumsum()

            leader_time = df.groupby(['season', 'round', 'lap'])['cumulative_time'].transform('min')
            df['gap_to_leader'] = df['cumulative_time'] - leader_time

        return df

    def create_all_temporal_features(self, df: pd.DataFrame, target_col: str = 'lap_time_seconds') -> pd.DataFrame:
        """
        Create all temporal features.

        Args:
            df: DataFrame with race data
            target_col: Target column for lag/rolling features

        Returns:
            DataFrame with all temporal features
        """
        df = self.create_lap_number_features(df)
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = self.create_delta_features(df, target_col)
        df = self.create_position_features(df)

        return df
