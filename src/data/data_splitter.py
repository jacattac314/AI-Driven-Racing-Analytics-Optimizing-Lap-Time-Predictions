"""Time-series aware data splitting"""

import pandas as pd
import numpy as np
from typing import Tuple
from src.utils.logging_utils import setup_logger


class TimeSeriesDataSplitter:
    """Split data chronologically for time-series forecasting"""

    def __init__(self, config: dict):
        """
        Initialize data splitter.

        Args:
            config: Configuration dictionary
        """
        self.train_end_year = config['training']['splits']['train_end_year']
        self.val_start_year = config['training']['splits']['validation_start_year']
        self.val_end_year = config['training']['splits']['validation_end_year']
        self.test_start_year = config['training']['splits']['test_start_year']

        self.logger = setup_logger(__name__, 'logs/data_splitter.log')

    def split_by_year(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically by year.

        Args:
            df: DataFrame with 'season' column

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info("Splitting data chronologically by year")

        # Ensure data is sorted by time
        df = df.sort_values(['season', 'round']).reset_index(drop=True)

        # Train set: up to train_end_year
        train_df = df[df['season'] <= self.train_end_year].copy()

        # Validation set: val_start_year to val_end_year
        val_df = df[
            (df['season'] >= self.val_start_year) &
            (df['season'] <= self.val_end_year)
        ].copy()

        # Test set: from test_start_year onwards
        test_df = df[df['season'] >= self.test_start_year].copy()

        self.logger.info(
            f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )
        self.logger.info(
            f"Year ranges - Train: {train_df['season'].min()}-{train_df['season'].max()}, "
            f"Val: {val_df['season'].min()}-{val_df['season'].max()}, "
            f"Test: {test_df['season'].min()}-{test_df['season'].max()}"
        )

        return train_df, val_df, test_df

    def get_train_test_split(self, df: pd.DataFrame,
                            feature_cols: list,
                            target_col: str) -> Tuple:
        """
        Get train/val/test splits with features and target separated.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Split data
        train_df, val_df, test_df = self.split_by_year(df)

        # Separate features and target
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        self.logger.info(f"Features: {len(feature_cols)}, Target: {target_col}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def stratified_split_by_track(self, df: pd.DataFrame,
                                  track_col: str = 'circuit_id') -> Tuple:
        """
        Split data ensuring all tracks are represented in each split.

        Args:
            df: DataFrame with track information
            track_col: Column name for track identifier

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info("Performing stratified split by track")

        # First do chronological split
        train_df, val_df, test_df = self.split_by_year(df)

        # Check track representation
        train_tracks = set(train_df[track_col].unique())
        val_tracks = set(val_df[track_col].unique())
        test_tracks = set(test_df[track_col].unique())

        self.logger.info(f"Unique tracks - Train: {len(train_tracks)}, "
                        f"Val: {len(val_tracks)}, Test: {len(test_tracks)}")

        # Warn if any split is missing tracks
        all_tracks = train_tracks | val_tracks | test_tracks
        if val_tracks != all_tracks:
            missing_val = all_tracks - val_tracks
            self.logger.warning(f"Validation set missing tracks: {missing_val}")

        if test_tracks != all_tracks:
            missing_test = all_tracks - test_tracks
            self.logger.warning(f"Test set missing tracks: {missing_test}")

        return train_df, val_df, test_df

    def create_time_series_folds(self, df: pd.DataFrame, n_folds: int = 5) -> list:
        """
        Create time-series cross-validation folds (expanding window).

        Args:
            df: DataFrame sorted by time
            n_folds: Number of folds

        Returns:
            List of (train_idx, val_idx) tuples
        """
        self.logger.info(f"Creating {n_folds} time-series CV folds")

        # Ensure data is sorted
        df = df.sort_values(['season', 'round']).reset_index(drop=True)

        # Get unique seasons in training data
        train_data = df[df['season'] <= self.train_end_year]
        seasons = sorted(train_data['season'].unique())

        if len(seasons) < n_folds:
            self.logger.warning(
                f"Not enough seasons ({len(seasons)}) for {n_folds} folds. "
                f"Using {len(seasons)} folds instead."
            )
            n_folds = len(seasons)

        folds = []
        fold_size = len(seasons) // n_folds

        for i in range(1, n_folds + 1):
            # Expanding window: use first (fold_size * i) seasons for training
            train_seasons = seasons[:fold_size * i]
            # Next fold_size seasons for validation
            if fold_size * i < len(seasons):
                val_seasons = seasons[fold_size * i:fold_size * (i + 1)]
            else:
                val_seasons = [seasons[-1]]

            train_idx = train_data[train_data['season'].isin(train_seasons)].index
            val_idx = train_data[train_data['season'].isin(val_seasons)].index

            folds.append((train_idx.tolist(), val_idx.tolist()))

            self.logger.info(
                f"Fold {i}: Train seasons {min(train_seasons)}-{max(train_seasons)}, "
                f"Val seasons {min(val_seasons)}-{max(val_seasons)}"
            )

        return folds

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   test_df: pd.DataFrame, output_dir: str):
        """
        Save train/val/test splits to files.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Output directory
        """
        from pathlib import Path
        from src.utils.io_utils import save_dataframe

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_dataframe(train_df, output_path / 'train.csv')
        save_dataframe(val_df, output_path / 'val.csv')
        save_dataframe(test_df, output_path / 'test.csv')

        self.logger.info(f"Saved splits to {output_dir}")
