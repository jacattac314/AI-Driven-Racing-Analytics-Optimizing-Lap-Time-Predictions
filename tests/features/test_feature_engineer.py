import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock pandas, numpy, etc. to bypass environment limitations
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['src.features.driver_features'] = MagicMock()
sys.modules['src.features.track_features'] = MagicMock()
sys.modules['src.features.temporal_features'] = MagicMock()
sys.modules['src.features.historical_features'] = MagicMock()
sys.modules['src.features.weather_features'] = MagicMock()
sys.modules['src.features.tire_features'] = MagicMock()
sys.modules['src.features.feature_selector'] = MagicMock()
sys.modules['src.utils.logging_utils'] = MagicMock()
sys.modules['src.utils.io_utils'] = MagicMock()

# Now we can import the module to test
from src.features.feature_engineer import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        config = {
            'features': {
                'enabled_features': {
                    'driver_features': True,
                    'track_features': True,
                    'historical_features': True,
                    'weather_features': True,
                    'tire_features': True,
                    'temporal_features': True,
                }
            }
        }
        self.fe = FeatureEngineer(config)

    def test_new_methods_exist(self):
        self.assertTrue(hasattr(self.fe, '_remove_missing_targets'))
        self.assertTrue(hasattr(self.fe, '_handle_missing_values'))
        self.assertTrue(hasattr(self.fe, '_perform_feature_selection'))

    def test_methods_callable(self):
        # We can't really call them cleanly with mocked pandas unless we mock all df behavior
        # But we can verify they are bound methods
        self.assertTrue(callable(getattr(self.fe, '_remove_missing_targets')))
        self.assertTrue(callable(getattr(self.fe, '_handle_missing_values')))
        self.assertTrue(callable(getattr(self.fe, '_perform_feature_selection')))

if __name__ == '__main__':
    unittest.main()
