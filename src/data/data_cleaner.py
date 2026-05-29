"""Data cleaning and preprocessing"""

import pandas as pd
import numpy as np
from typing import Dict, List
from src.utils.logging_utils import setup_logger
from src.data.data_validator import DataValidator


class DataCleaner:
    """Clean and preprocess F1 data"""

    def __init__(self):
        self.logger = setup_logger(__name__, 'logs/data_cleaner.log')
        self.validator = DataValidator()

    def clean_lap_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean lap time data.

        Args:
            df: Raw lap times DataFrame

        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Cleaning lap times data ({len(df)} rows)")

        df = df.copy()

        # Validate and parse lap times
        if 'time' in df.columns:
            df['lap_time_seconds'] = df['time'].apply(
                self.validator._parse_lap_time
            )

        # Remove invalid lap times
        original_count = len(df)

        # Remove null lap times
        df = df.dropna(subset=['lap_time_seconds'])

        # Remove unrealistic lap times (< 60s or > 200s)
        df = df[(df['lap_time_seconds'] >= 60) & (df['lap_time_seconds'] <= 200)]

        removed_count = original_count - len(df)
        self.logger.info(f"Removed {removed_count} invalid lap times")

        return df

    def clean_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean race results data.

        Args:
            df: Raw results DataFrame

        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Cleaning results data ({len(df)} rows)")

        df = df.copy()

        # Convert position to numeric (handle 'R', 'D', 'E', 'W', 'F', 'N')
        df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')

        # Parse finish time to milliseconds
        if 'milliseconds' in df.columns:
            df['finish_time_ms'] = pd.to_numeric(df['milliseconds'], errors='coerce')

        # Clean status (finished, crashed, retired, etc.)
        df['finished'] = df['status'].str.contains('Finished|Lap', case=False, na=False)

        return df

    def clean_qualifying(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean qualifying data.

        Args:
            df: Raw qualifying DataFrame

        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Cleaning qualifying data ({len(df)} rows)")

        df = df.copy()

        # Parse qualifying times
        for col in ['q1', 'q2', 'q3']:
            if col in df.columns:
                df[f'{col}_seconds'] = df[col].apply(
                    self.validator._parse_lap_time
                )

        return df

    def clean_pit_stops(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean pit stop data.

        Args:
            df: Raw pit stops DataFrame

        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Cleaning pit stops data ({len(df)} rows)")

        df = df.copy()

        # Convert duration to float
        if 'duration' in df.columns:
            df['duration_seconds'] = pd.to_numeric(df['duration'], errors='coerce')

            # Remove unrealistic pit stops (< 1s or > 120s)
            df = df[(df['duration_seconds'] >= 1) & (df['duration_seconds'] <= 120)]

        return df

    def _merge_base_results(self, results_df: pd.DataFrame, races_df: pd.DataFrame) -> pd.DataFrame:
        """Merge results with races info."""
        return results_df.merge(
            races_df,
            on=['season', 'round'],
            how='left',
            suffixes=('', '_race')
        )

    def _merge_qualifying(self, merged_df: pd.DataFrame, qualifying_df: pd.DataFrame) -> pd.DataFrame:
        """Merge qualifying data if available."""
        if qualifying_df is None:
            return merged_df

        return merged_df.merge(
            qualifying_df,
            on=['season', 'round', 'driver_id'],
            how='left',
            suffixes=('', '_quali')
        )

    def _apply_lap_times_base(self, results_df: pd.DataFrame, races_df: pd.DataFrame,
                              lap_times_df: pd.DataFrame) -> pd.DataFrame:
        """Use lap times as base and merge race/results info."""
        if lap_times_df is None:
            return None

        merged = lap_times_df.merge(
            races_df,
            on=['season', 'round'],
            how='left'
        )

        merged = merged.merge(
            results_df[['season', 'round', 'driver_id', 'constructor_id',
                      'grid', 'position', 'points', 'status']],
            on=['season', 'round', 'driver_id'],
            how='left'
        )
        return merged

    def merge_race_data(self, races_df: pd.DataFrame, results_df: pd.DataFrame,
                       qualifying_df: pd.DataFrame = None,
                       pit_stops_df: pd.DataFrame = None,
                       lap_times_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Merge all race-related data.

        Args:
            races_df: Races DataFrame
            results_df: Results DataFrame
            qualifying_df: Qualifying DataFrame (optional)
            pit_stops_df: Pit stops DataFrame (optional)
            lap_times_df: Lap times DataFrame (optional)

        Returns:
            Merged DataFrame
        """
        self.logger.info("Merging race data")

        # Start with results and merge races info
        merged = self._merge_base_results(results_df, races_df)

        # Merge qualifying if available
        merged = self._merge_qualifying(merged, qualifying_df)

        # If lap times available, use as base
        lap_times_merged = self._apply_lap_times_base(results_df, races_df, lap_times_df)
        if lap_times_merged is not None:
            merged = lap_times_merged

        self.logger.info(f"Merged data: {len(merged)} rows")

        return merged

    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values with specified strategies.

        Args:
            df: DataFrame with missing values
            strategy: Dictionary mapping column names to strategies
                     ('drop', 'mean', 'median', 'mode', 'ffill', 'bfill')

        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()

        if strategy is None:
            strategy = {}

        for col in df.columns:
            if df[col].isna().sum() > 0:
                strat = strategy.get(col, 'drop')

                if strat == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strat == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                elif strat == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif strat == 'ffill':
                    df[col].fillna(method='ffill', inplace=True)
                elif strat == 'bfill':
                    df[col].fillna(method='bfill', inplace=True)
                elif strat == 'drop':
                    pass  # Will drop at the end

        # Drop rows with remaining missing values in critical columns
        critical_cols = ['season', 'round', 'driver_id']
        critical_cols = [col for col in critical_cols if col in df.columns]
        df = df.dropna(subset=critical_cols)

        return df

    def create_unique_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create unique identifiers for races and drivers.

        Args:
            df: DataFrame

        Returns:
            DataFrame with added identifiers
        """
        df = df.copy()

        # Race ID
        if 'season' in df.columns and 'round' in df.columns:
            df['race_id'] = df['season'].astype(str) + '_' + df['round'].astype(str)

        # Driver-Race ID
        if 'race_id' in df.columns and 'driver_id' in df.columns:
            df['driver_race_id'] = df['race_id'] + '_' + df['driver_id']

        return df

    def clean_all_data(self, races_df: pd.DataFrame, results_df: pd.DataFrame,
                      qualifying_df: pd.DataFrame = None,
                      pit_stops_df: pd.DataFrame = None,
                      lap_times_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Clean all data end-to-end.

        Args:
            races_df: Races DataFrame
            results_df: Results DataFrame
            qualifying_df: Qualifying DataFrame (optional)
            pit_stops_df: Pit stops DataFrame (optional)
            lap_times_df: Lap times DataFrame (optional)

        Returns:
            Cleaned and merged DataFrame
        """
        self.logger.info("Starting comprehensive data cleaning")

        # Clean individual datasets
        results_clean = self.clean_results(results_df)

        if qualifying_df is not None:
            qualifying_clean = self.clean_qualifying(qualifying_df)
        else:
            qualifying_clean = None

        if pit_stops_df is not None:
            pit_stops_clean = self.clean_pit_stops(pit_stops_df)
        else:
            pit_stops_clean = None

        if lap_times_df is not None:
            lap_times_clean = self.clean_lap_times(lap_times_df)
        else:
            lap_times_clean = None

        # Merge data
        merged = self.merge_race_data(
            races_df,
            results_clean,
            qualifying_clean,
            pit_stops_clean,
            lap_times_clean
        )

        # Create unique identifiers
        merged = self.create_unique_identifiers(merged)

        # Handle missing values
        merged = self.handle_missing_values(merged)

        self.logger.info(f"Data cleaning completed: {len(merged)} rows")

        return merged
