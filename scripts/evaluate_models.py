"""Script to evaluate all trained models"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.interpretability import ModelInterpreter
from src.features.feature_engineer import FeatureEngineer
from src.utils.io_utils import load_config, load_dataframe
from src.utils.logging_utils import setup_logger
import joblib


def main():
    """Main function to evaluate models."""
    logger = setup_logger('evaluate_models', 'logs/evaluate_models.log')
    logger.info("Starting model evaluation")

    # Load configuration
    config = load_config('config/config.yaml')

    # Load test data
    splits_path = Path(config['data']['paths']['splits'])
    test_df = load_dataframe(str(splits_path / 'test.csv'))

    # Prepare features
    feature_engineer = FeatureEngineer(config)

    logger.info("Preparing test features")
    X_test, y_test, feature_names = feature_engineer.prepare_features_for_modeling(
        test_df,
        target_col='lap_time_seconds',
        select_features=False
    )

    # Load models
    models_path = Path(config['paths']['models'])
    models = {}

    logger.info("Loading trained models")
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name

            # Skip if not enabled
            model_config = config['models'].get(model_name, {})
            if not model_config.get('enabled', True):
                continue

            try:
                # Load model
                if model_name == 'neural_net':
                    from tensorflow import keras
                    model = keras.models.load_model(str(model_dir / 'model.h5'))
                else:
                    model = joblib.load(str(model_dir / 'model.pkl'))

                models[model_name] = type('Model', (), {'predict': lambda self, X: model.predict(X), 'get_model': lambda self: model})()

                logger.info(f"Loaded {model_name}")

            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")

    # Evaluate models
    evaluator = ModelEvaluator()

    # Get additional data for grouped metrics
    additional_data = {}
    if 'circuit_id' in test_df.columns:
        additional_data['track'] = test_df['circuit_id'].values
    if 'driver_id' in test_df.columns:
        additional_data['driver'] = test_df['driver_id'].values

    # Evaluate all models
    comparison_df = evaluator.evaluate_all_models(models, X_test, y_test, additional_data)

    logger.info("\nModel Comparison:")
    logger.info(f"\n{comparison_df.to_string()}")

    # Save results
    results_path = Path(config['paths']['results'])
    evaluator.save_results(str(results_path / 'metrics'))

    # Generate visualizations
    evaluator.generate_visualizations(str(results_path / 'plots'))

    # Generate comparison plot
    evaluator.generate_comparison_plot(
        comparison_df,
        str(results_path / 'plots/model_comparison.png')
    )

    # Get best model
    best_model = evaluator.get_best_model(metric='mae')
    logger.info(f"\nBest model: {best_model}")

    logger.info("Model evaluation completed successfully")


if __name__ == "__main__":
    main()
