"""End-to-end pipeline execution script"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
import fetch_data
import preprocess_data
import engineer_features
import train_all_models
import evaluate_models


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description='Run F1 Lap Time Forecasting Pipeline')
    parser.add_argument(
        '--steps',
        nargs='+',
        default=['all'],
        choices=['fetch', 'preprocess', 'features', 'train', 'evaluate', 'all'],
        help='Pipeline steps to run'
    )
    args = parser.parse_args()

    logger = setup_logger('pipeline', 'logs/pipeline.log')
    logger.info("="*80)
    logger.info("Starting F1 Lap Time Forecasting Pipeline")
    logger.info("="*80)

    steps = args.steps
    if 'all' in steps:
        steps = ['fetch', 'preprocess', 'features', 'train', 'evaluate']

    try:
        # Step 1: Fetch data
        if 'fetch' in steps:
            logger.info("\n[1/5] Fetching data from Ergast API...")
            fetch_data.main()

        # Step 2: Preprocess data
        if 'preprocess' in steps:
            logger.info("\n[2/5] Preprocessing and cleaning data...")
            preprocess_data.main()

        # Step 3: Engineer features
        if 'features' in steps:
            logger.info("\n[3/5] Engineering features...")
            engineer_features.main()

        # Step 4: Train models
        if 'train' in steps:
            logger.info("\n[4/5] Training all models...")
            train_all_models.main()

        # Step 5: Evaluate models
        if 'evaluate' in steps:
            logger.info("\n[5/5] Evaluating models...")
            evaluate_models.main()

        logger.info("\n" + "="*80)
        logger.info("Pipeline completed successfully!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"\nPipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
