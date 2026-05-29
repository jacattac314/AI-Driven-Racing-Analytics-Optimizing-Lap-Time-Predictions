"""Data collection from Ergast F1 API"""

import time
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from src.utils.logging_utils import setup_logger
from src.utils.io_utils import save_dataframe, load_json, save_json


class ErgastAPIClient:
    """Client for fetching F1 data from Ergast API"""

    def __init__(self, config: Dict):
        """
        Initialize Ergast API client.

        Args:
            config: Configuration dictionary
        """
        self.base_url = config['data']['api']['base_url']
        self.rate_limit = config['data']['api']['rate_limit']
        self.timeout = config['data']['api']['timeout']
        self.cache_enabled = config['data']['api']['cache_enabled']
        self.retry_attempts = config['data']['api']['retry_attempts']
        self.retry_delay = config['data']['api']['retry_delay']

        self.raw_data_path = Path(config['data']['paths']['raw'])
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(__name__, 'logs/data_collector.log')

        self.min_request_interval = 1.0 / self.rate_limit
        self.last_request_time = 0

    def _rate_limit_sleep(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, url: str) -> Optional[Dict]:
        """
        Make HTTP request with retry logic.

        Args:
            url: Request URL

        Returns:
            JSON response or None on failure
        """
        self._rate_limit_sleep()

        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.logger.error(f"Failed to fetch data from {url}")
                    return None

    def fetch_seasons(self, start_year: int, end_year: int) -> List[int]:
        """
        Get list of seasons between start and end year.

        Args:
            start_year: Starting year
            end_year: Ending year

        Returns:
            List of season years
        """
        return list(range(start_year, end_year + 1))

    def fetch_races(self, season: int) -> pd.DataFrame:
        """
        Fetch all races for a given season.

        Args:
            season: Season year

        Returns:
            DataFrame with race information
        """
        url = f"{self.base_url}/{season}.json?limit=100"
        data = self._make_request(url)

        if not data or 'MRData' not in data:
            self.logger.error(f"Failed to fetch races for season {season}")
            return pd.DataFrame()

        races = data['MRData']['RaceTable']['Races']

        if not races:
            self.logger.warning(f"No races found for season {season}")
            return pd.DataFrame()

        race_data = []
        for race in races:
            race_data.append({
                'season': season,
                'round': int(race['round']),
                'race_name': race['raceName'],
                'circuit_id': race['Circuit']['circuitId'],
                'circuit_name': race['Circuit']['circuitName'],
                'country': race['Circuit']['Location']['country'],
                'locality': race['Circuit']['Location']['locality'],
                'date': race['date'],
                'time': race.get('time', ''),
                'url': race['url']
            })

        return pd.DataFrame(race_data)

    def fetch_results(self, season: int, round_num: int) -> pd.DataFrame:
        """
        Fetch race results for a specific race.

        Args:
            season: Season year
            round_num: Round number

        Returns:
            DataFrame with race results
        """
        url = f"{self.base_url}/{season}/{round_num}/results.json?limit=100"
        data = self._make_request(url)

        if not data or 'MRData' not in data:
            return pd.DataFrame()

        races = data['MRData']['RaceTable']['Races']
        if not races:
            return pd.DataFrame()

        results = races[0]['Results']

        result_data = []
        for result in results:
            result_data.append({
                'season': season,
                'round': round_num,
                'driver_id': result['Driver']['driverId'],
                'driver_code': result['Driver'].get('code', ''),
                'driver_number': result['Driver'].get('permanentNumber', ''),
                'constructor_id': result['Constructor']['constructorId'],
                'grid': int(result['grid']),
                'position': result.get('position', None),
                'position_text': result['positionText'],
                'points': float(result['points']),
                'laps': int(result['laps']),
                'status': result['status'],
                'time': result.get('Time', {}).get('time', None),
                'milliseconds': result.get('Time', {}).get('millis', None),
                'fastest_lap': result.get('FastestLap', {}).get('lap', None),
                'fastest_lap_time': result.get('FastestLap', {}).get('Time', {}).get('time', None)
            })

        return pd.DataFrame(result_data)

    def fetch_qualifying(self, season: int, round_num: int) -> pd.DataFrame:
        """
        Fetch qualifying results for a specific race.

        Args:
            season: Season year
            round_num: Round number

        Returns:
            DataFrame with qualifying results
        """
        url = f"{self.base_url}/{season}/{round_num}/qualifying.json?limit=100"
        data = self._make_request(url)

        if not data or 'MRData' not in data:
            return pd.DataFrame()

        races = data['MRData']['RaceTable']['Races']
        if not races:
            return pd.DataFrame()

        qualifying = races[0]['QualifyingResults']

        quali_data = []
        for result in qualifying:
            quali_data.append({
                'season': season,
                'round': round_num,
                'driver_id': result['Driver']['driverId'],
                'constructor_id': result['Constructor']['constructorId'],
                'position': int(result['position']),
                'q1': result.get('Q1', None),
                'q2': result.get('Q2', None),
                'q3': result.get('Q3', None)
            })

        return pd.DataFrame(quali_data)

    def fetch_pit_stops(self, season: int, round_num: int) -> pd.DataFrame:
        """
        Fetch pit stop data for a specific race.

        Args:
            season: Season year
            round_num: Round number

        Returns:
            DataFrame with pit stop data
        """
        url = f"{self.base_url}/{season}/{round_num}/pitstops.json?limit=1000"
        data = self._make_request(url)

        if not data or 'MRData' not in data:
            return pd.DataFrame()

        races = data['MRData']['RaceTable']['Races']
        if not races or 'PitStops' not in races[0]:
            return pd.DataFrame()

        pit_stops = races[0]['PitStops']

        pit_data = []
        for stop in pit_stops:
            pit_data.append({
                'season': season,
                'round': round_num,
                'driver_id': stop['driverId'],
                'stop': int(stop['stop']),
                'lap': int(stop['lap']),
                'time': stop['time'],
                'duration': stop['duration']
            })

        return pd.DataFrame(pit_data)

    def fetch_lap_times(self, season: int, round_num: int) -> pd.DataFrame:
        """
        Fetch lap times for a specific race.

        Args:
            season: Season year
            round_num: Round number

        Returns:
            DataFrame with lap times
        """
        # Note: Ergast API requires lap-by-lap queries for lap times
        # This is a simplified version that fetches all laps

        all_laps = []

        # First, get the number of laps in the race
        results_df = self.fetch_results(season, round_num)
        if results_df.empty:
            return pd.DataFrame()

        max_laps = results_df['laps'].max()

        # Fetch lap times lap by lap (this is limited by API)
        # Due to API limitations, we'll fetch a sample of laps
        lap_sample = [1, 10, 20, 30, 40, 50, 60, max_laps]
        lap_sample = [lap for lap in lap_sample if lap <= max_laps]

        for lap in lap_sample:
            url = f"{self.base_url}/{season}/{round_num}/laps/{lap}.json?limit=100"
            data = self._make_request(url)

            if data and 'MRData' in data:
                races = data['MRData']['RaceTable']['Races']
                if races and 'Laps' in races[0]:
                    laps = races[0]['Laps'][0]['Timings']

                    for timing in laps:
                        all_laps.append({
                            'season': season,
                            'round': round_num,
                            'lap': lap,
                            'driver_id': timing['driverId'],
                            'position': int(timing['position']),
                            'time': timing['time']
                        })

        return pd.DataFrame(all_laps)

    def _collect_race_data(self, season: int, round_num: int) -> Dict[str, pd.DataFrame]:
        """Collect all data for a specific race."""
        race_data = {}

        # Results
        results_df = self.fetch_results(season, round_num)
        if not results_df.empty:
            race_data['results'] = results_df

        # Qualifying
        qualifying_df = self.fetch_qualifying(season, round_num)
        if not qualifying_df.empty:
            race_data['qualifying'] = qualifying_df

        # Pit stops (available from 2012 onwards)
        if season >= 2012:
            pit_stops_df = self.fetch_pit_stops(season, round_num)
            if not pit_stops_df.empty:
                race_data['pit_stops'] = pit_stops_df

        # Lap times (sample only due to API limitations)
        lap_times_df = self.fetch_lap_times(season, round_num)
        if not lap_times_df.empty:
            race_data['lap_times'] = lap_times_df

        return race_data

    def _save_dataset(self, data_list: List[pd.DataFrame], filename: str, description: str):
        """Combine and save a list of dataframes."""
        if data_list:
            combined_df = pd.concat(data_list, ignore_index=True)
            save_dataframe(combined_df, self.raw_data_path / filename)
            self.logger.info(f"Saved {len(combined_df)} {description}")

    def fetch_all_data(self, start_year: int, end_year: int):
        """
        Fetch all available data for specified seasons.

        Args:
            start_year: Starting year
            end_year: Ending year
        """
        self.logger.info(f"Fetching F1 data from {start_year} to {end_year}")

        seasons = self.fetch_seasons(start_year, end_year)

        all_races = []
        all_results = []
        all_qualifying = []
        all_pit_stops = []
        all_lap_times = []

        for season in tqdm(seasons, desc="Processing seasons"):
            self.logger.info(f"Fetching data for season {season}")

            # Fetch races
            races_df = self.fetch_races(season)
            if not races_df.empty:
                all_races.append(races_df)

                # Fetch race-specific data
                for _, race in races_df.iterrows():
                    round_num = race['round']
                    race_data = self._collect_race_data(season, round_num)

                    if 'results' in race_data:
                        all_results.append(race_data['results'])
                    if 'qualifying' in race_data:
                        all_qualifying.append(race_data['qualifying'])
                    if 'pit_stops' in race_data:
                        all_pit_stops.append(race_data['pit_stops'])
                    if 'lap_times' in race_data:
                        all_lap_times.append(race_data['lap_times'])

        # Combine all data
        self.logger.info("Combining data...")

        self._save_dataset(all_races, 'races.csv', 'races')
        self._save_dataset(all_results, 'results.csv', 'race results')
        self._save_dataset(all_qualifying, 'qualifying.csv', 'qualifying results')
        self._save_dataset(all_pit_stops, 'pit_stops.csv', 'pit stops')
        self._save_dataset(all_lap_times, 'lap_times.csv', 'lap times')

        self.logger.info("Data collection completed!")
