"""Script to engineer features from cleaned data"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.feature_engineer import FeatureEngineer
from src.data.data_splitter import TimeSeriesDataSplitter
from src.utils.io_utils import load_config, load_dataframe, save_dataframe
from src.utils.logging_utils import setup_logger


def main():
    """Main function to engineer features."""
    logger = setup_logger('engineer_features', 'logs/engineer_features.log')
    logger.info("Starting feature engineering")

    # Load configuration
    config = load_config('config/config.yaml')

    # Load cleaned data
    processed_path = Path(config['data']['paths']['processed'])
    cleaned_data_file = processed_path / 'cleaned_data.csv'

    logger.info(f"Loading cleaned data from {cleaned_data_file}")
    df = load_dataframe(str(cleaned_data_file))

    # Engineer features
    feature_engineer = FeatureEngineer(config)
    df_features = feature_engineer.create_all_features(df)

    # Save features
    features_path = Path(config['data']['paths']['features'])
    features_path.mkdir(parents=True, exist_ok=True)

    features_file = features_path / 'features.csv'
    save_dataframe(df_features, str(features_file))

    logger.info(f"Features saved to {features_file}")

    # Split data
    splitter = TimeSeriesDataSplitter(config)
    train_df, val_df, test_df = splitter.split_by_year(df_features)

    # Save splits
    splits_path = Path(config['data']['paths']['splits'])
    splitter.save_splits(train_df, val_df, test_df, str(splits_path))

    logger.info("Feature engineering and data splitting completed")


if __name__ == "__main__":
    main()
