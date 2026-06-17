import sys
from unittest.mock import MagicMock, patch

try:
    import numpy as np
except ImportError:
    sys.modules['numpy'] = MagicMock()
    sys.modules['sklearn'] = MagicMock()
    sys.modules['sklearn.metrics'] = MagicMock()
    sys.modules['joblib'] = MagicMock()
    import numpy as np

import pytest
from src.models.base_model import BaseModel

class DummyModel(BaseModel):
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        pass
    def predict(self, X):
        pass

def test_get_feature_importance_with_feature_importances_():
    """Test get_feature_importance when model has feature_importances_ attribute."""
    model = DummyModel()
    model.model = MagicMock()

    # Set feature_importances_ and ensure coef_ is not checked or doesn't matter
    expected_importances = [0.1, 0.2, 0.7]
    model.model.feature_importances_ = expected_importances

    result = model.get_feature_importance()
    assert result == expected_importances

@patch('src.models.base_model.np.abs')
def test_get_feature_importance_with_coef_(mock_abs):
    """Test get_feature_importance when model has coef_ attribute but not feature_importances_."""
    model = DummyModel()
    model.model = MagicMock()

    # Remove feature_importances_ to fallback to coef_
    del model.model.feature_importances_

    model.model.coef_ = [-0.5, 0.2, -0.1]

    # Mock np.abs to return absolute values
    expected_importances = [0.5, 0.2, 0.1]
    mock_abs.return_value = expected_importances

    result = model.get_feature_importance()
    assert result == expected_importances
    mock_abs.assert_called_once_with(model.model.coef_)

def test_get_feature_importance_neither():
    """Test get_feature_importance when model has neither feature_importances_ nor coef_."""
    model = DummyModel()
    model.model = MagicMock()

    # Remove both attributes
    del model.model.feature_importances_
    del model.model.coef_

    result = model.get_feature_importance()
    assert result is None
