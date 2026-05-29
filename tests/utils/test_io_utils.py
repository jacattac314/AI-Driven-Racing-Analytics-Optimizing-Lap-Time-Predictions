import os
import pytest
import pandas as pd
from src.utils.io_utils import load_dataframe

def test_load_dataframe_csv(tmp_path):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)

    loaded_df = load_dataframe(str(file_path))
    pd.testing.assert_frame_equal(df, loaded_df)

def test_load_dataframe_parquet(tmp_path):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    file_path = tmp_path / "test.parquet"
    df.to_parquet(file_path, index=False)

    loaded_df = load_dataframe(str(file_path))
    pd.testing.assert_frame_equal(df, loaded_df)

def test_load_dataframe_unsupported_format(tmp_path):
    file_path = tmp_path / "test.txt"
    # Create an empty file
    file_path.touch()

    with pytest.raises(ValueError, match="Unsupported file format:"):
        load_dataframe(str(file_path))
