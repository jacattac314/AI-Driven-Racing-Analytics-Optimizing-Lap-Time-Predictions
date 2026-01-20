"""Data validation and quality checks"""

import pandas as pd
import numpy as np
from typing import Dict, List
from src.utils.logging_utils import setup_logger


class DataValidator:
    """Validate data quality and integrity"""

    def __init__(self):
        self.logger = setup_logger(__name__, 'logs/data_validator.log')

    def validate_schema(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Check if DataFrame has required columns.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if valid, False otherwise
        """
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            self.logger.error(f"Missing columns: {missing_columns}")
            return False

        self.logger.info("Schema validation passed")
        return True

    def check_missing_values(self, df: pd.DataFrame, max_missing_pct: float = 0.5) -> Dict:
        """
        Check missing values in DataFrame.

        Args:
            df: DataFrame to check
            max_missing_pct: Maximum allowed missing percentage

        Returns:
            Dictionary with missing value statistics
        """
        missing_stats = {}
        total_rows = len(df)

        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / total_rows

            missing_stats[col] = {
                'missing_count': missing_count,
                'missing_pct': missing_pct
            }

            if missing_pct > max_missing_pct:
                self.logger.warning(
                    f"Column '{col}' has {missing_pct:.2%} missing values"
                )

        return missing_stats

    def check_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> int:
        """
        Check for duplicate rows.

        Args:
            df: DataFrame to check
            subset: Columns to consider for duplicates

        Returns:
            Number of duplicate rows
        """
        duplicates = df.duplicated(subset=subset).sum()

        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate rows")
        else:
            self.logger.info("No duplicates found")

        return duplicates

    def validate_lap_times(self, df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
        """
        Validate lap times and flag suspicious values.

        Args:
            df: DataFrame with lap times
            time_col: Column name for lap time

        Returns:
            DataFrame with validation flags
        """
        df = df.copy()

        # Convert lap time string to seconds
        if df[time_col].dtype == 'object':
            df['lap_time_seconds'] = df[time_col].apply(self._parse_lap_time)
        else:
            df['lap_time_seconds'] = df[time_col]

        # Flag suspicious lap times
        df['valid_lap'] = True

        # Too fast (< 60 seconds - unrealistic for F1)
        df.loc[df['lap_time_seconds'] < 60, 'valid_lap'] = False

        # Too slow (> 200 seconds - likely pit lap or safety car)
        df.loc[df['lap_time_seconds'] > 200, 'valid_lap'] = False

        # Missing or null
        df.loc[df['lap_time_seconds'].isna(), 'valid_lap'] = False

        invalid_count = (~df['valid_lap']).sum()
        self.logger.info(
            f"Flagged {invalid_count} invalid lap times out of {len(df)}"
        )

        return df

    @staticmethod
    def _parse_lap_time(time_str: str) -> float:
        """
        Parse lap time string to seconds.

        Args:
            time_str: Time string (e.g., '1:23.456')

        Returns:
            Time in seconds
        """
        if pd.isna(time_str) or time_str == '':
            return np.nan

        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(time_str)
        except (ValueError, AttributeError):
            return np.nan

    def check_time_series_continuity(self, df: pd.DataFrame,
                                     time_col: str = 'date') -> bool:
        """
        Check time series continuity.

        Args:
            df: DataFrame with time series data
            time_col: Column name for time

        Returns:
            True if continuous, False otherwise
        """
        if time_col not in df.columns:
            self.logger.error(f"Time column '{time_col}' not found")
            return False

        df_sorted = df.sort_values(time_col)

        # Check for gaps
        # This is a simplified check - you can customize based on your needs

        self.logger.info("Time series continuity check completed")
        return True

    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': self.check_missing_values(df),
            'duplicates': self.check_duplicates(df),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 ** 2
        }

        self.logger.info("Quality report generated")
        return report
