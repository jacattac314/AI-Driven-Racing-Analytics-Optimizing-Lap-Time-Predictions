import pandas as pd
import numpy as np
import pytest

from src.features.weather_features import WeatherFeatureEngineer

class TestWeatherFeatureEngineer:
    def setup_method(self):
        self.engineer = WeatherFeatureEngineer()

    def test_create_weather_features_no_date(self):
        """Test that dataframe is returned unchanged if 'date' column is missing."""
        df = pd.DataFrame({'other_col': [1, 2, 3]})
        result = self.engineer.create_weather_features(df)

        pd.testing.assert_frame_equal(df, result)
        assert 'date_parsed' not in result.columns

    def test_create_weather_features_with_date(self):
        """Test that date-related features are correctly created when 'date' is present."""
        # Using a date that falls in 'spring' (month 4)
        df = pd.DataFrame({'date': ['2023-04-15']})
        result = self.engineer.create_weather_features(df)

        assert 'date_parsed' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['date_parsed'])

        assert 'month' in result.columns
        assert result['month'].iloc[0] == 4

        assert 'day_of_year' in result.columns
        assert result['day_of_year'].iloc[0] == 105  # April 15th is the 105th day of a non-leap year

        assert 'season_weather' in result.columns
        assert result['season_weather'].iloc[0] == 'spring'

        assert 'weather_season_spring' in result.columns
        assert result['weather_season_spring'].iloc[0] == True

    def test_create_weather_features_invalid_date(self):
        """Test behavior when the date is invalid or unparseable."""
        df = pd.DataFrame({'date': ['invalid_date', '2023-04-15']})
        result = self.engineer.create_weather_features(df)

        # First row should be NaT
        assert pd.isna(result['date_parsed'].iloc[0])
        assert pd.isna(result['month'].iloc[0])

        # Second row should be correctly parsed
        assert result['month'].iloc[1] == 4
        assert result['season_weather'].iloc[1] == 'spring'

    def test_get_season(self):
        """Test the static method _get_season directly."""
        assert WeatherFeatureEngineer._get_season(1) == 'winter'
        assert WeatherFeatureEngineer._get_season(4) == 'spring'
        assert WeatherFeatureEngineer._get_season(7) == 'summer'
        assert WeatherFeatureEngineer._get_season(10) == 'autumn'
