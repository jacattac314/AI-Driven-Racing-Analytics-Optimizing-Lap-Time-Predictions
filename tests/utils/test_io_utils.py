import os
import tempfile
import pytest
import yaml

from src.utils.io_utils import load_config

def test_load_config_valid_yaml():
    """Test loading a valid YAML file."""
    config_data = {
        'model': {
            'name': 'xgboost',
            'learning_rate': 0.1
        },
        'data': {
            'batch_size': 32
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.dump(config_data, temp_file)
        temp_filepath = temp_file.name

    try:
        loaded_config = load_config(temp_filepath)
        assert loaded_config == config_data
    finally:
        os.remove(temp_filepath)


def test_load_config_non_existent_file():
    """Test loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config_file_that_should_not_exist.yaml")


def test_load_config_invalid_yaml():
    """Test loading an invalid YAML file raises yaml.YAMLError."""
    invalid_yaml_content = "invalid: yaml: content: {"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write(invalid_yaml_content)
        temp_filepath = temp_file.name

    try:
        with pytest.raises(yaml.YAMLError):
            load_config(temp_filepath)
    finally:
        os.remove(temp_filepath)
