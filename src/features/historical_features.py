"""Historical performance feature engineering"""

import pandas as pd
import numpy as np


class HistoricalFeatureEngineer:
    """Generate historical performance features"""

    def __init__(self, config: dict = None):
        if config:
            self.races_lookback = config['features']['historical']['races_lookback']
            self.seasons_lookback = config['features']['historical']['seasons_lookback']
        else:
            self.races_lookback = 5
            self.seasons_lookback = 2

    def create_constructor_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create constructor (team) performance features.

        Args:
            df: DataFrame with constructor information

        Returns:
            DataFrame with constructor features
        """
        df = df.copy()
        df = df.sort_values(['constructor_id', 'season', 'round'])

        # Constructor season points
        if 'points' in df.columns:
            df['constructor_season_points'] = df.groupby(['constructor_id', 'season'])['points'].cumsum()

            # Constructor championship rank
            df['constructor_championship_rank'] = df.groupby(['season', 'round'])['constructor_season_points'].rank(
                ascending=False, method='min'
            )

        # Constructor average position (rolling)
        if 'position_numeric' in df.columns:
            df['constructor_avg_pos'] = df.groupby('constructor_id')['position_numeric'].transform(
                lambda x: x.rolling(self.races_lookback, min_periods=1).mean()
            )

        return df

    def create_head_to_head(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create head-to-head comparison features.

        Args:
            df: DataFrame with race results

        Returns:
            DataFrame with head-to-head features
        """
        df = df.copy()

        # Teammate comparison (simplified - would need teammate identification)
        # This is a placeholder for more sophisticated teammate analysis

        # Position within constructor
        if 'position_numeric' in df.columns:
            df['position_within_team'] = df.groupby(['season', 'round', 'constructor_id'])['position_numeric'].rank()

        return df

    def create_season_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create season-long trend features.

        Args:
            df: DataFrame with race results

        Returns:
            DataFrame with trend features
        """
        df = df.copy()
        df = df.sort_values(['driver_id', 'season', 'round'])

        # Races into season
        df['races_into_season'] = df.groupby(['driver_id', 'season']).cumcount() + 1

        # Performance trend (position improvement/decline)
        if 'position_numeric' in df.columns:
            df['position_trend'] = df.groupby(['driver_id', 'season'])['position_numeric'].transform(
                lambda x: x.diff()
            )

        return df

    def create_recent_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on recent race results.

        Args:
            df: DataFrame with race results

        Returns:
            DataFrame with recent results features
        """
        df = df.copy()
        df = df.sort_values(['driver_id', 'season', 'round'])

        # Did not finish in last race
        if 'finished' in df.columns:
            df['dnf_last_race'] = df.groupby('driver_id')['finished'].shift(1).apply(lambda x: 0 if x else 1)

        # Podium in last N races
        if 'position_numeric' in df.columns:
            df['podiums_last_5'] = df.groupby('driver_id')['position_numeric'].transform(
                lambda x: (x <= 3).rolling(self.races_lookback, min_periods=1).sum()
            )

            # Wins in last N races
            df['wins_last_5'] = df.groupby('driver_id')['position_numeric'].transform(
                lambda x: (x == 1).rolling(self.races_lookback, min_periods=1).sum()
            )

        return df

    def create_all_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all historical performance features.

        Args:
            df: DataFrame with race data

        Returns:
            DataFrame with all historical features
        """
        df = self.create_constructor_performance(df)
        df = self.create_head_to_head(df)
        df = self.create_season_trends(df)
        df = self.create_recent_results(df)

        return df
