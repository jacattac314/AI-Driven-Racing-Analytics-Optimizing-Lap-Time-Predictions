"""Hyperparameter tuning using GridSearchCV"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from src.utils.logging_utils import setup_logger
import pandas as pd


class HyperparameterTuner:
    """Hyperparameter optimization"""

    def __init__(self, config: dict):
        """
        Initialize hyperparameter tuner.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cv_folds = config['training']['cv']['n_folds']
        self.scoring = 'neg_mean_absolute_error'
        self.logger = setup_logger(__name__, 'logs/hyperparameter_tuner.log')

        self.best_params = {}
        self.best_scores = {}

    def grid_search(self, model, param_grid, X_train, y_train):
        """
        Perform grid search for hyperparameter tuning.

        Args:
            model: Model instance
            param_grid: Parameter grid dictionary
            X_train: Training features
            y_train: Training target

        Returns:
            Best parameters dictionary
        """
        self.logger.info("Starting GridSearchCV")

        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        # Fit
        grid_search.fit(X_train, y_train)

        # Log results
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best score: {-grid_search.best_score_:.4f}")

        self.best_params = grid_search.best_params_
        self.best_scores = {
            'best_score': -grid_search.best_score_,
            'mean_test_score': -grid_search.cv_results_['mean_test_score'].max()
        }

        return grid_search.best_params_, grid_search.best_estimator_

    def randomized_search(self, model, param_distributions, X_train, y_train,
                         n_iter=10):
        """
        Perform randomized search for hyperparameter tuning.

        Args:
            model: Model instance
            param_distributions: Parameter distribution dictionary
            X_train: Training features
            y_train: Training target
            n_iter: Number of parameter settings sampled

        Returns:
            Best parameters dictionary
        """
        self.logger.info("Starting RandomizedSearchCV")

        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=tscv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )

        # Fit
        random_search.fit(X_train, y_train)

        # Log results
        self.logger.info(f"Best parameters: {random_search.best_params_}")
        self.logger.info(f"Best score: {-random_search.best_score_:.4f}")

        self.best_params = random_search.best_params_
        self.best_scores = {
            'best_score': -random_search.best_score_,
            'mean_test_score': -random_search.cv_results_['mean_test_score'].max()
        }

        return random_search.best_params_, random_search.best_estimator_

    def tune_model(self, model_type: str, model, X_train, y_train):
        """
        Tune hyperparameters for a specific model type.

        Args:
            model_type: Model type name (e.g., 'xgboost', 'random_forest')
            model: Model instance
            X_train: Training features
            y_train: Training target

        Returns:
            Best parameters and tuned model
        """
        # Get parameter grid from config
        model_config = self.config['models'].get(model_type, {})
        tuning_config = model_config.get('tuning', {})

        if not tuning_config.get('enabled', False):
            self.logger.info(f"Tuning disabled for {model_type}")
            return None, model

        param_grid = tuning_config.get('param_grid', {})

        if not param_grid:
            self.logger.warning(f"No parameter grid defined for {model_type}")
            return None, model

        # Perform grid search
        best_params, best_model = self.grid_search(model, param_grid, X_train, y_train)

        return best_params, best_model

    def get_tuning_results(self) -> pd.DataFrame:
        """
        Get hyperparameter tuning results.

        Returns:
            DataFrame with tuning results
        """
        if not self.best_params:
            return pd.DataFrame()

        results = {
            'best_params': [self.best_params],
            'best_score': [self.best_scores.get('best_score', None)]
        }

        return pd.DataFrame(results)
