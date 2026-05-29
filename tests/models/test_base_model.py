import pytest
import numpy as np
from src.models.base_model import BaseModel

class DummyModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.is_trained = True

    def predict(self, X):
        # Just return a dummy prediction based on X
        # For testing evaluate, we can just return predefined predictions
        # Or better yet, we can set the predictions on the instance and return them
        return self.mock_predictions

def test_evaluate_not_trained():
    model = DummyModel()
    with pytest.raises(ValueError, match="Model must be trained before evaluation"):
        model.evaluate(np.array([[1]]), np.array([1]))

def test_evaluate_trained():
    model = DummyModel()
    model.is_trained = True

    y = np.array([10.0, 20.0, 30.0])
    model.mock_predictions = np.array([12.0, 18.0, 30.0])
    X = np.array([[1], [2], [3]]) # X doesn't matter for DummyModel.predict

    metrics = model.evaluate(X, y)

    # mae = (2 + 2 + 0) / 3 = 1.333...
    assert np.isclose(metrics['mae'], 4/3)

    # mse = (4 + 4 + 0) / 3 = 8/3
    # rmse = sqrt(8/3)
    assert np.isclose(metrics['rmse'], np.sqrt(8/3))

    # mape = mean([2/10, 2/20, 0/30]) * 100 = mean([0.2, 0.1, 0]) * 100 = 0.1 * 100 = 10.0
    assert np.isclose(metrics['mape'], 10.0)

    # r2 test
    # mean_y = 20
    # ss_tot = (10-20)**2 + (20-20)**2 + (30-20)**2 = 100 + 0 + 100 = 200
    # ss_res = (10-12)**2 + (20-18)**2 + (30-30)**2 = 4 + 4 + 0 = 8
    # r2 = 1 - (8/200) = 1 - 0.04 = 0.96
    assert np.isclose(metrics['r2'], 0.96)
