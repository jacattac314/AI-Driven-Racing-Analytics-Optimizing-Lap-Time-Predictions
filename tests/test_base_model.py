import pytest
import numpy as np
from src.models.base_model import BaseModel

class DummyModel(BaseModel):
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.is_trained = True

    def predict(self, X):
        return np.ones(len(X))

def test_evaluate_not_trained_raises_error():
    model = DummyModel()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])

    with pytest.raises(ValueError, match="Model must be trained before evaluation"):
        model.evaluate(X, y)

def test_evaluate_success():
    model = DummyModel()
    X = np.array([[1, 2], [3, 4]])
    # predictions will be [1.0, 1.0]
    y = np.array([1.0, 1.0])

    model.train(X, y)
    metrics = model.evaluate(X, y)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics

    # y = [1, 1], pred = [1, 1]
    # mae = 0, rmse = 0, r2 = 1.0, mape = 0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["mape"] == 0.0

def test_save_model_not_trained_raises_error():
    model = DummyModel()

    with pytest.raises(ValueError, match="Model must be trained before saving"):
        model.save_model("dummy_path.pkl")
