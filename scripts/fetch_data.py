"""Script to fetch F1 data from Ergast API"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_collector import ErgastAPIClient
from src.utils.io_utils import load_config
from src.utils.logging_utils import setup_logger


def main():
    """Main function to fetch F1 data."""
    logger = setup_logger('fetch_data', 'logs/fetch_data.log')
    logger.info("Starting data collection")

    # Load configuration
    config = load_config('config/config.yaml')

    # Initialize API client
    client = ErgastAPIClient(config)

    # Fetch data
    start_year = config['data']['seasons']['start']
    end_year = config['data']['seasons']['end']

    client.fetch_all_data(start_year, end_year)

    logger.info("Data collection completed successfully")


if __name__ == "__main__":
    main()
