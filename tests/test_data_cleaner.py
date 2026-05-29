import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['src.utils.logging_utils'] = MagicMock()
sys.modules['src.data.data_validator'] = MagicMock()

def test_data_cleaner_imports():
    """Verify that data_cleaner.py can be imported without syntax errors."""
    import src.data.data_cleaner as data_cleaner
    assert hasattr(data_cleaner, 'DataCleaner')
