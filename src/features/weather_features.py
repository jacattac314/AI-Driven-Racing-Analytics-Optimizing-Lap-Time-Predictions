"""Weather-related feature engineering"""

import pandas as pd
import numpy as np


class WeatherFeatureEngineer:
    """Generate weather-related features"""

    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-related features.

        Note: Ergast API has limited weather data.
        This is a placeholder for weather features that would be enriched
        with external weather APIs.

        Args:
            df: DataFrame with race data

        Returns:
            DataFrame with weather features
        """
        df = df.copy()

        # Date-based features (proxy for season/weather)
        if "date" in df.columns:
            df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
            df["month"] = df["date_parsed"].dt.month
            df["day_of_year"] = df["date_parsed"].dt.dayofyear

            # Season (meteorological)
            df["season_weather"] = df["month"].apply(self._get_season)

            # One-hot encode season
            season_dummies = pd.get_dummies(
                df["season_weather"], prefix="weather_season"
            )
            df = pd.concat([df, season_dummies], axis=1)

        return df

    @staticmethod
    def _get_season(month: int) -> str:
        """
        Get meteorological season from month.

        Args:
            month: Month number (1-12)

        Returns:
            Season name
        """
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"

    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create location-based features (proxy for weather).

        Args:
            df: DataFrame with location data

        Returns:
            DataFrame with location features
        """
        df = df.copy()

        # Country encoding (proxy for climate)
        if "country" in df.columns:
            df["country_encoded"] = pd.Categorical(df["country"]).codes

        return df

    def create_all_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all weather-related features.

        Args:
            df: DataFrame with race data

        Returns:
            DataFrame with all weather features
        """
        df = self.create_weather_features(df)
        df = self.create_location_features(df)

        return df
