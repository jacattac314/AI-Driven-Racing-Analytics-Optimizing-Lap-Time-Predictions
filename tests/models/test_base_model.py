import pytest
from src.models.base_model import BaseModel

class DummyModel(BaseModel):
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.is_trained = True
        self.model = "dummy_trained_model"

    def predict(self, X):
        return [0] * len(X)

def test_save_untrained_model_raises_error():
    model = DummyModel()
    with pytest.raises(ValueError, match="Model must be trained before saving"):
        model.save_model("dummy_path.joblib")

def test_evaluate_untrained_model_raises_error():
    model = DummyModel()
    with pytest.raises(ValueError, match="Model must be trained before evaluation"):
        model.evaluate([[1]], [1])
