"""Script to preprocess and clean F1 data"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_cleaner import DataCleaner
from src.utils.io_utils import load_config, load_dataframe, save_dataframe
from src.utils.logging_utils import setup_logger


def main():
    """Main function to preprocess data."""
    logger = setup_logger('preprocess_data', 'logs/preprocess_data.log')
    logger.info("Starting data preprocessing")

    # Load configuration
    config = load_config('config/config.yaml')

    # Load raw data
    raw_path = Path(config['data']['paths']['raw'])

    logger.info("Loading raw data")
    races_df = load_dataframe(raw_path / 'races.csv')
    results_df = load_dataframe(raw_path / 'results.csv')

    # Load optional data
    try:
        qualifying_df = load_dataframe(raw_path / 'qualifying.csv')
    except:
        qualifying_df = None
        logger.warning("Qualifying data not found")

    try:
        pit_stops_df = load_dataframe(raw_path / 'pit_stops.csv')
    except:
        pit_stops_df = None
        logger.warning("Pit stops data not found")

    try:
        lap_times_df = load_dataframe(raw_path / 'lap_times.csv')
    except:
        lap_times_df = None
        logger.warning("Lap times data not found")

    # Clean data
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_all_data(
        races_df,
        results_df,
        qualifying_df,
        pit_stops_df,
        lap_times_df
    )

    # Save cleaned data
    processed_path = Path(config['data']['paths']['processed'])
    processed_path.mkdir(parents=True, exist_ok=True)

    output_file = processed_path / 'cleaned_data.csv'
    save_dataframe(cleaned_df, str(output_file))

    logger.info(f"Preprocessed data saved to {output_file}")
    logger.info(f"Total rows: {len(cleaned_df)}, Total columns: {len(cleaned_df.columns)}")


if __name__ == "__main__":
    main()
