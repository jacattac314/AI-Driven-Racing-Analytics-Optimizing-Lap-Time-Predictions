"""Neural Network model implementation using TensorFlow/Keras"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseModel
from src.utils.logging_utils import setup_logger


class NeuralNetModel(BaseModel):
    """Deep Learning model using TensorFlow/Keras"""

    def __init__(self, config: dict = None):
        """
        Initialize Neural Network model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.logger = setup_logger(__name__, 'logs/neural_net_model.log')

        # Default parameters
        self.params = {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'validation_split': 0.2
        }

        # Update with config if provided
        if config and 'params' in config:
            self.params.update(config['params'])

        # Scaler for feature normalization
        self.scaler = StandardScaler()

        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    def _build_model(self, input_dim: int):
        """
        Build neural network architecture.

        Args:
            input_dim: Number of input features

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_dim,)))

        # Hidden layers
        for i, units in enumerate(self.params['hidden_layers']):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i}'))
            model.add(layers.BatchNormalization(name=f'batch_norm_{i}'))
            model.add(layers.Dropout(self.params['dropout_rate'], name=f'dropout_{i}'))

        # Output layer
        model.add(layers.Dense(1, name='output'))

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train neural network model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training arguments
        """
        self.logger.info("Training Neural Network model")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Build model
        input_dim = X_train_scaled.shape[1]
        self.model = self._build_model(input_dim)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.params['early_stopping_patience'],
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                verbose=0
            )
        ]

        # Prepare validation data
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
            validation_split = None
        else:
            validation_data = None
            validation_split = self.params['validation_split']

        # Train model
        history = self.model.fit(
            X_train_scaled,
            y_train,
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )

        self.is_trained = True
        self.logger.info(
            f"Neural Network training completed. "
            f"Final loss: {history.history['loss'][-1]:.4f}"
        )

    def predict(self, X):
        """
        Make predictions using neural network.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)

        return predictions.flatten()

    def save_model(self, filepath: str):
        """
        Save neural network model.

        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        # Save Keras model
        self.model.save(filepath)

        # Save scaler separately
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load neural network model.

        Args:
            filepath: Path to load model from
        """
        # Load Keras model
        self.model = keras.models.load_model(filepath)

        # Load scaler
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)

        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")
