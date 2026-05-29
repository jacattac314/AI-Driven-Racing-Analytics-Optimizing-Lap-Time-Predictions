import pytest
import numpy as np
from src.models.base_model import BaseModel

class DummyModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.is_trained = True

    def predict(self, X):
        # Dummy prediction: returns array of zeros matching length of X
        return np.zeros(len(X))

def test_base_model_evaluate_success():
    """Test evaluate method correctly computes metrics when trained."""
    model = DummyModel()
    model.train(None, None)

    # Dummy data
    X = np.array([[1, 2], [3, 4]])
    # Targets are 2, 4. Predictions are 0, 0.
    y = np.array([2.0, 4.0])

    metrics = model.evaluate(X, y)

    # mae: (2 + 4) / 2 = 3.0
    assert metrics['mae'] == 3.0
    # mse: (4 + 16) / 2 = 10.0 => rmse = sqrt(10)
    assert np.isclose(metrics['rmse'], np.sqrt(10.0))
    # mape: abs((2-0)/2) + abs((4-0)/4) / 2 = 1 + 1 / 2 = 1 => 100%
    assert metrics['mape'] == 100.0

    # Ensure metrics contains expected keys
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics

def test_base_model_evaluate_untrained():
    """Test evaluate method raises error when not trained."""
    model = DummyModel()

    X = np.array([[1, 2]])
    y = np.array([1])

    with pytest.raises(ValueError, match="Model must be trained before evaluation"):
        model.evaluate(X, y)
