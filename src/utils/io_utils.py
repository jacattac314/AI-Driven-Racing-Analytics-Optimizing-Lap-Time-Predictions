"""I/O utility functions"""

import json
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to config file

    Returns:
        Dict containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """Load JSON file to dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: str):
    """Save object to pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_dataframe(df: pd.DataFrame, filepath: str, **kwargs):
    """
    Save DataFrame to file (CSV or Parquet).

    Args:
        df: DataFrame to save
        filepath: Output file path
        **kwargs: Additional arguments for to_csv or to_parquet
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if filepath.endswith('.csv'):
        df.to_csv(filepath, index=False, **kwargs)
    elif filepath.endswith('.parquet'):
        df.to_parquet(filepath, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def load_dataframe(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from file (CSV or Parquet).

    Args:
        filepath: Input file path
        **kwargs: Additional arguments for read_csv or read_parquet

    Returns:
        pd.DataFrame
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, **kwargs)
    elif filepath.endswith('.parquet'):
        return pd.read_parquet(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
