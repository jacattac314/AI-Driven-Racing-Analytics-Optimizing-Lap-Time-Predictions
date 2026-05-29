"""Tire strategy feature engineering"""

import pandas as pd
import numpy as np


class TireFeatureEngineer:
    """Generate tire-related features"""

    def create_pit_stop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create pit stop related features.

        Args:
            df: DataFrame with pit stop data

        Returns:
            DataFrame with pit stop features
        """
        df = df.copy()

        # If pit stop data is available
        if 'stop' in df.columns:
            # Number of stops
            df['num_pit_stops'] = df['stop']

            # Stint number (which stint the driver is currently in)
            df['stint_number'] = df['stop'] + 1

        return df

    def create_tire_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tire age features (laps on current tire set).

        Args:
            df: DataFrame with lap data

        Returns:
            DataFrame with tire age features
        """
        df = df.copy()

        # This requires pit stop data merged with lap data
        # Simplified version: assume tire age based on laps since pit stop

        if 'lap' in df.columns and 'stop' in df.columns:
            # Calculate laps since last pit stop
            df = df.sort_values(['driver_id', 'season', 'round', 'lap'])

            # This is simplified - actual tire age would need pit stop lap info
            # Placeholder: assume fresh tires every 20 laps
            df['estimated_tire_age'] = df['lap'] % 20

        return df

    def create_all_tire_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all tire-related features.

        Args:
            df: DataFrame with race data

        Returns:
            DataFrame with all tire features
        """
        df = self.create_pit_stop_features(df)
        df = self.create_tire_age_features(df)

        return df
