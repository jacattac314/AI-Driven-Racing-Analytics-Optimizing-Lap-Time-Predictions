import pandas as pd
from src.data.data_validator import DataValidator

class TestDataValidator:
    def setup_method(self):
        """Set up a DataValidator instance for tests."""
        self.validator = DataValidator()

    def test_validate_schema_success(self):
        """Test schema validation succeeds when all required columns are present."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]})

        # Should return True because all required columns are in df.columns
        result = self.validator.validate_schema(df, ['col1', 'col2'])
        assert result is True

    def test_validate_schema_missing_columns(self):
        """Test schema validation fails when required columns are missing."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Should return False because 'col3' is missing
        result = self.validator.validate_schema(df, ['col1', 'col3'])
        assert result is False

    def test_validate_schema_empty_required_columns(self):
        """Test schema validation succeeds when no columns are required."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Should return True because required list is empty
        result = self.validator.validate_schema(df, [])
        assert result is True

    def test_validate_schema_exact_match(self):
        """Test schema validation succeeds with exact column match."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Should return True because required columns exactly match df.columns
        result = self.validator.validate_schema(df, ['col1', 'col2'])
        assert result is True
