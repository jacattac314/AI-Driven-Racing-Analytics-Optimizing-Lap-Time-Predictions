"""Script to train all models"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import ModelTrainer
from src.features.feature_engineer import FeatureEngineer
from src.utils.io_utils import load_config, load_dataframe
from src.utils.logging_utils import setup_logger


def main():
    """Main function to train all models."""
    logger = setup_logger('train_all_models', 'logs/train_all_models.log')
    logger.info("Starting model training")

    # Load configuration
    config = load_config('config/config.yaml')

    # Load data splits
    splits_path = Path(config['data']['paths']['splits'])

    logger.info("Loading data splits")
    train_df = load_dataframe(str(splits_path / 'train.csv'))
    val_df = load_dataframe(str(splits_path / 'val.csv'))
    test_df = load_dataframe(str(splits_path / 'test.csv'))

    # Prepare features
    feature_engineer = FeatureEngineer(config)

    logger.info("Preparing features for modeling")
    X_train, y_train, feature_names = feature_engineer.prepare_features_for_modeling(
        train_df,
        target_col='lap_time_seconds',
        select_features=False
    )

    X_val, y_val, _ = feature_engineer.prepare_features_for_modeling(
        val_df,
        target_col='lap_time_seconds',
        select_features=False
    )

    # Ensure same features
    common_features = list(set(X_train.columns) & set(X_val.columns))
    X_train = X_train[common_features]
    X_val = X_val[common_features]

    logger.info(f"Training with {len(X_train)} samples and {len(common_features)} features")

    # Train models
    trainer = ModelTrainer(config)
    models = trainer.train_all_models(X_train, y_train, X_val, y_val)

    # Get training summary
    summary = trainer.get_training_summary()
    logger.info("\nTraining Summary:")
    logger.info(f"\n{summary.to_string()}")

    # Save models
    models_path = Path(config['paths']['models'])
    trainer.save_models(str(models_path))

    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()
