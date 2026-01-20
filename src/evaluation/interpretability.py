"""Model interpretability using SHAP and feature importance"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logging_utils import setup_logger
from src.utils.visualization import plot_feature_importance


class ModelInterpreter:
    """Interpret model predictions and features"""

    def __init__(self):
        self.logger = setup_logger(__name__, 'logs/interpretability.log')
        self.shap_values = {}
        self.feature_importances = {}

    def calculate_shap_values(self, model, X, model_name: str,
                             sample_size: int = 100):
        """
        Calculate SHAP values for model interpretation.

        Args:
            model: Trained model
            X: Feature matrix
            model_name: Model name
            sample_size: Sample size for SHAP calculation

        Returns:
            SHAP values
        """
        self.logger.info(f"Calculating SHAP values for {model_name}")

        try:
            # Sample data if too large
            if len(X) > sample_size:
                sample_idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_idx] if hasattr(X, 'iloc') else X[sample_idx]
            else:
                X_sample = X

            # Get underlying model if it's our custom model class
            if hasattr(model, 'get_model'):
                base_model = model.get_model()
            else:
                base_model = model

            # Create SHAP explainer
            if hasattr(base_model, 'predict_proba'):
                explainer = shap.TreeExplainer(base_model)
            else:
                explainer = shap.Explainer(base_model.predict, X_sample)

            # Calculate SHAP values
            shap_values = explainer(X_sample)

            self.shap_values[model_name] = shap_values

            self.logger.info(f"SHAP values calculated for {model_name}")

            return shap_values

        except Exception as e:
            self.logger.error(f"Failed to calculate SHAP values: {str(e)}")
            return None

    def get_feature_importance(self, model, feature_names: list,
                              model_name: str) -> pd.DataFrame:
        """
        Get feature importance from model.

        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Model name

        Returns:
            DataFrame with feature importance
        """
        self.logger.info(f"Extracting feature importance for {model_name}")

        # Get importance from model
        if hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            self.logger.warning(f"No feature importance available for {model_name}")
            return pd.DataFrame()

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        self.feature_importances[model_name] = importance_df

        return importance_df

    def plot_feature_importance(self, model_name: str, top_n: int = 20,
                               output_path: str = None):
        """
        Plot feature importance.

        Args:
            model_name: Model name
            top_n: Number of top features to plot
            output_path: Path to save plot (optional)
        """
        if model_name not in self.feature_importances:
            self.logger.warning(f"No feature importance for {model_name}")
            return

        importance_df = self.feature_importances[model_name]

        plot_feature_importance(
            importance_df['feature'].tolist(),
            importance_df['importance'].values,
            top_n=top_n,
            title=f"{model_name} - Feature Importance",
            save_path=output_path
        )

    def plot_shap_summary(self, model_name: str, output_path: str = None):
        """
        Plot SHAP summary.

        Args:
            model_name: Model name
            output_path: Path to save plot (optional)
        """
        if model_name not in self.shap_values:
            self.logger.warning(f"No SHAP values for {model_name}")
            return

        shap_values = self.shap_values[model_name]

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, show=False)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"SHAP summary plot saved to {output_path}")

        plt.show()

    def analyze_prediction(self, model, X, index: int, feature_names: list):
        """
        Analyze a single prediction.

        Args:
            model: Trained model
            X: Feature matrix
            index: Index of prediction to analyze
            feature_names: List of feature names

        Returns:
            Dictionary with prediction analysis
        """
        # Get prediction
        if hasattr(X, 'iloc'):
            X_single = X.iloc[[index]]
        else:
            X_single = X[index:index+1]

        prediction = model.predict(X_single)[0]

        # Get feature values
        feature_values = X_single.iloc[0] if hasattr(X_single, 'iloc') else X_single[0]

        analysis = {
            'prediction': prediction,
            'features': dict(zip(feature_names, feature_values))
        }

        return analysis

    def generate_interpretation_report(self, output_dir: str):
        """
        Generate comprehensive interpretation report.

        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating interpretation report")

        # Save feature importances
        for model_name, importance_df in self.feature_importances.items():
            importance_df.to_csv(
                output_path / f"{model_name}_feature_importance.csv",
                index=False
            )

            # Plot
            self.plot_feature_importance(
                model_name,
                output_path=str(output_path / f"{model_name}_importance.png")
            )

        # Generate SHAP plots
        for model_name in self.shap_values.keys():
            self.plot_shap_summary(
                model_name,
                output_path=str(output_path / f"{model_name}_shap_summary.png")
            )

        self.logger.info(f"Interpretation report saved to {output_dir}")
