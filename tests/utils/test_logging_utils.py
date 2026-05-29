import logging
import os
from pathlib import Path
import sys

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils.logging_utils import setup_logger

def test_setup_logger_console_only():
    """Test logger setup without file handler."""
    logger_name = "test_logger_console"
    logger = setup_logger(logger_name, level=logging.DEBUG)

    assert logger.name == logger_name
    assert logger.level == logging.DEBUG

    # Should only have one handler (Console)
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert not isinstance(logger.handlers[0], logging.FileHandler)

def test_setup_logger_with_file(tmp_path):
    """Test logger setup with both console and file handlers."""
    logger_name = "test_logger_file"
    log_dir = tmp_path / "logs"
    log_file = log_dir / "test.log"

    # Notice the directory 'logs' does not exist yet. setup_logger should create it.
    logger = setup_logger(logger_name, log_file=str(log_file), level=logging.INFO)

    assert logger.name == logger_name
    assert logger.level == logging.INFO

    # Should have two handlers (Console and File)
    assert len(logger.handlers) == 2

    # Verify handler types
    handler_types = {type(h) for h in logger.handlers}
    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types

    # Verify file was actually created
    assert log_dir.exists()
    assert log_dir.is_dir()
    assert log_file.exists()
    assert log_file.is_file()

    # Let's log something and check if it's written
    test_msg = "This is a test log message."
    logger.info(test_msg)

    # Close handlers to flush
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    with open(log_file, "r") as f:
        log_content = f.read()

    assert test_msg in log_content
