"""Track-specific feature engineering"""

import pandas as pd
import numpy as np


class TrackFeatureEngineer:
    """Generate track-related features"""

    # Track metadata (simplified - can be expanded)
    TRACK_METADATA = {
        "monaco": {"type": "street", "difficulty": "high", "overtaking": "low"},
        "monza": {"type": "permanent", "difficulty": "medium", "overtaking": "high"},
        "spa": {"type": "permanent", "difficulty": "high", "overtaking": "medium"},
        "singapore": {"type": "street", "difficulty": "high", "overtaking": "low"},
        "silverstone": {
            "type": "permanent",
            "difficulty": "medium",
            "overtaking": "medium",
        },
    }

    def create_track_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create track type features.

        Args:
            df: DataFrame with circuit information

        Returns:
            DataFrame with track type features
        """
        df = df.copy()

        # Determine track type based on circuit_id
        df["track_type"] = df["circuit_id"].apply(
            lambda x: self.TRACK_METADATA.get(x, {}).get("type", "permanent")
        )

        # One-hot encode track type
        track_type_dummies = pd.get_dummies(df["track_type"], prefix="track_type")
        df = pd.concat([df, track_type_dummies], axis=1)

        return df

    def create_historical_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create historical track performance features.

        Args:
            df: DataFrame with race results

        Returns:
            DataFrame with historical track features
        """
        df = df.copy()
        df = df.sort_values(["circuit_id", "season", "round"])

        # Average lap time at this track (historical)
        if "lap_time_seconds" in df.columns:
            df["track_avg_lap_time"] = df.groupby("circuit_id")[
                "lap_time_seconds"
            ].transform("mean")

            # Standard deviation of lap times at this track
            df["track_std_lap_time"] = df.groupby("circuit_id")[
                "lap_time_seconds"
            ].transform("std")

        # Average grid position at this track
        if "grid" in df.columns:
            df["track_avg_grid"] = df.groupby("circuit_id")["grid"].transform("mean")

        return df

    def create_driver_track_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create driver performance at specific track.

        Args:
            df: DataFrame with race results

        Returns:
            DataFrame with driver-track features
        """
        df = df.copy()
        df = df.sort_values(["driver_id", "circuit_id", "season", "round"])

        # Driver's average position at this track
        if "position_numeric" in df.columns:
            df["driver_track_avg_pos"] = df.groupby(["driver_id", "circuit_id"])[
                "position_numeric"
            ].transform(lambda x: x.expanding(min_periods=1).mean())

        # Number of times driver raced at this track
        df["driver_track_experience"] = (
            df.groupby(["driver_id", "circuit_id"]).cumcount() + 1
        )

        return df

    def create_track_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create track characteristic features.

        Args:
            df: DataFrame with circuit information

        Returns:
            DataFrame with track characteristics
        """
        df = df.copy()

        # Encode circuit as categorical
        df["circuit_encoded"] = pd.Categorical(df["circuit_id"]).codes

        return df

    def create_all_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all track-related features.

        Args:
            df: DataFrame with race data

        Returns:
            DataFrame with all track features
        """
        df = self.create_track_type(df)
        df = self.create_historical_performance(df)
        df = self.create_driver_track_performance(df)
        df = self.create_track_characteristics(df)

        return df
