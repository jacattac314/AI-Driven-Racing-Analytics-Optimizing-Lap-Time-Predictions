"""Driver-specific feature engineering"""

import pandas as pd
import numpy as np
from typing import Dict


class DriverFeatureEngineer:
    """Generate driver-related features"""

    def create_career_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cumulative career statistics for each driver.

        Args:
            df: DataFrame with race results

        Returns:
            DataFrame with career stats features
        """
        df = df.copy()
        df = df.sort_values(['driver_id', 'season', 'round'])

        # Career wins (cumulative)
        df['career_wins'] = df.groupby('driver_id')['position'].apply(
            lambda x: (x == 1).cumsum()
        )

        # Career podiums (cumulative)
        df['career_podiums'] = df.groupby('driver_id')['position_numeric'].apply(
            lambda x: (x <= 3).cumsum()
        )

        # Career points (cumulative)
        if 'points' in df.columns:
            df['career_points'] = df.groupby('driver_id')['points'].cumsum()

        # Career races (experience)
        df['career_races'] = df.groupby('driver_id').cumcount() + 1

        return df

    def create_recent_form(self, df: pd.DataFrame, windows: list = [3, 5]) -> pd.DataFrame:
        """
        Create rolling average features for recent performance.

        Args:
            df: DataFrame with race results
            windows: List of window sizes for rolling averages

        Returns:
            DataFrame with recent form features
        """
        df = df.copy()
        df = df.sort_values(['driver_id', 'season', 'round'])

        for window in windows:
            # Rolling average position
            if 'position_numeric' in df.columns:
                df[f'driver_avg_pos_last_{window}'] = df.groupby('driver_id')['position_numeric'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

            # Rolling average points
            if 'points' in df.columns:
                df[f'driver_avg_points_last_{window}'] = df.groupby('driver_id')['points'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

        return df

    def create_championship_standing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create championship standing features.

        Args:
            df: DataFrame with race results

        Returns:
            DataFrame with championship features
        """
        df = df.copy()
        df = df.sort_values(['season', 'round', 'driver_id'])

        # Season points so far
        df['season_points'] = df.groupby(['driver_id', 'season'])['points'].cumsum()

        # Championship rank at this point in season
        df['championship_rank'] = df.groupby(['season', 'round'])['season_points'].rank(
            ascending=False, method='min'
        )

        return df

    def create_driver_experience(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create driver experience features.

        Args:
            df: DataFrame with driver information

        Returns:
            DataFrame with experience features
        """
        df = df.copy()

        # First season
        df['first_season'] = df.groupby('driver_id')['season'].transform('min')

        # Years of experience
        df['years_experience'] = df['season'] - df['first_season']

        return df

    def create_all_driver_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all driver-related features.

        Args:
            df: DataFrame with race data

        Returns:
            DataFrame with all driver features
        """
        df = self.create_career_stats(df)
        df = self.create_recent_form(df)
        df = self.create_championship_standing(df)
        df = self.create_driver_experience(df)

        return df
