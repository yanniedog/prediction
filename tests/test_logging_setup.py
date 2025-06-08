import pytest
import logging
import sys
import os
from pathlib import Path
import tempfile
import shutil
from typing import Generator
import logging_setup
import config
import warnings
from datetime import datetime, timedelta
import time
from unittest.mock import patch, MagicMock

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Ensure all handlers are closed before cleanup
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
    logger.handlers = []
    try:
        shutil.rmtree(temp_dir)
    except PermissionError:
        # If we still can't delete, wait a moment and try again
        time.sleep(0.1)
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # If we still can't delete, log a warning but don't fail the test
            print(f"Warning: Could not delete temporary directory {temp_dir}", file=sys.stderr)

@pytest.fixture(autouse=True)
def reset_logging() -> Generator[None, None, None]:
    """Reset logging configuration before and after each test."""
    # Store original handlers
    original_handlers = logging.getLogger().handlers[:]
    original_level = logging.getLogger().level
    
    # Reset logging
    logging.getLogger().handlers = []
    logging.getLogger().level = logging.NOTSET
    
    yield
    
    # Restore original state
    logging.getLogger().handlers = original_handlers
    logging.getLogger().level = original_level

@pytest.fixture(scope="function")
def temp_log_dir():
    """Create a temporary directory for test logs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def setup_logging(temp_log_dir):
    """Set up logging with temporary directory."""
    with patch('logging_setup.LOG_DIR', temp_log_dir):
        logging_setup.setup_logging()
        yield
        # Clean up logging handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

def test_setup_logging_basic(temp_dir: Path) -> None:
    """Test basic logging setup functionality."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Temporarily modify config.LOG_DIR
        original_log_dir = config.LOG_DIR
        try:
            config.LOG_DIR = temp_dir / "logs"
            # Test default setup
            logging_setup.setup_logging()
            logger = logging.getLogger()
            # Verify handlers
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
            assert len(console_handlers) == 1
            # Verify log file creation
            log_file = config.LOG_DIR / "logfile.txt"
            assert log_file.exists()
            # Verify log levels
            assert file_handlers[0].level == logging.WARNING
            expected_console_level = logging.WARNING if 'PYTEST_CURRENT_TEST' in os.environ else logging.INFO
            assert console_handlers[0].level == expected_console_level
            assert logger.level == min(file_handlers[0].level, console_handlers[0].level)
            # Test logging at different levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            # Verify log file contents
            with open(log_file, 'r') as f:
                log_contents = f.read()
                assert "Debug message" not in log_contents  # Debug not logged
                assert "Info message" not in log_contents   # Info not logged to file
                assert "Warning message" in log_contents    # Warning logged
                assert "Error message" in log_contents      # Error logged
        finally:
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
            logger.handlers = []
            config.LOG_DIR = original_log_dir

def test_setup_logging_custom_levels(temp_dir: Path) -> None:
    """Test logging setup with custom levels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_log_dir = config.LOG_DIR
        try:
            config.LOG_DIR = temp_dir / "logs"
            logging_setup.setup_logging(
                file_level=logging.DEBUG,
                console_level=logging.WARNING
            )
            logger = logging.getLogger()
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
            assert file_handlers[0].level == logging.DEBUG
            assert console_handlers[0].level == logging.WARNING
            assert logger.level == logging.DEBUG
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            log_file = config.LOG_DIR / "logfile.txt"
            with open(log_file, 'r') as f:
                log_contents = f.read()
                assert "Debug message" in log_contents
                assert "Info message" in log_contents
                assert "Warning message" in log_contents
        finally:
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
            logger.handlers = []
            config.LOG_DIR = original_log_dir

def test_setup_logging_file_modes(temp_dir: Path) -> None:
    """Test logging setup with different file modes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_log_dir = config.LOG_DIR
        try:
            config.LOG_DIR = temp_dir / "logs"
            logging_setup.setup_logging(file_mode='a')
            logger = logging.getLogger()
            logger.warning("First message")
            logging.getLogger().handlers = []
            logging_setup.setup_logging(file_mode='a')
            logger = logging.getLogger()
            logger.warning("Second message")
            log_file = config.LOG_DIR / "logfile.txt"
            with open(log_file, 'r') as f:
                log_contents = f.read()
                assert "First message" in log_contents
                assert "Second message" in log_contents
            logging.getLogger().handlers = []
            logging_setup.setup_logging(file_mode='w')
            logger = logging.getLogger()
            logger.warning("Third message")
            with open(log_file, 'r') as f:
                log_contents = f.read()
                assert "First message" not in log_contents
                assert "Second message" not in log_contents
                assert "Third message" in log_contents
            logging.getLogger().handlers = []
            logging_setup.setup_logging(file_mode='invalid')
            logger = logging.getLogger()
            logger.warning("Fourth message")
            with open(log_file, 'r') as f:
                log_contents = f.read()
                assert "Third message" not in log_contents
                assert "Fourth message" in log_contents
        finally:
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
            logger.handlers = []
            config.LOG_DIR = original_log_dir

def test_set_console_log_level() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging_setup.setup_logging()
        logger = logging.getLogger()
        logging_setup.set_console_log_level(logging.DEBUG)
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert console_handlers[0].level == logging.DEBUG
        assert logger.level == min(file_handlers[0].level, console_handlers[0].level)
        logging_setup.set_console_log_level(logging.WARNING)
        assert console_handlers[0].level == logging.WARNING
        assert logger.level == min(file_handlers[0].level, console_handlers[0].level)
        logging_setup.set_console_log_level(logging.ERROR)
        assert console_handlers[0].level == logging.ERROR
        assert logger.level == min(file_handlers[0].level, console_handlers[0].level)

def test_reset_console_log_level() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging_setup.setup_logging(console_level=logging.WARNING)
        logger = logging.getLogger()
        logging_setup.set_console_log_level(logging.DEBUG)
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert console_handlers[0].level == logging.DEBUG
        logging_setup.reset_console_log_level()
        assert console_handlers[0].level == logging.WARNING
        assert logger.level == logging.WARNING

def test_library_quieting() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging_setup.setup_logging()
        libraries_to_quiet = [
            'matplotlib', 'requests', 'urllib3', 'PIL', 'skopt', 'statsmodels',
            'numexpr', 'asyncio'
        ]
        for lib_name in libraries_to_quiet:
            lib_logger = logging.getLogger(lib_name)
            assert lib_logger.level == logging.WARNING

def test_error_handling(temp_dir: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_log_dir = config.LOG_DIR
        try:
            if hasattr(os, 'chmod'):
                read_only_dir = temp_dir / "readonly"
                read_only_dir.mkdir()
                os.chmod(str(read_only_dir), 0o444)
                config.LOG_DIR = read_only_dir
                logging_setup.setup_logging()
                logger = logging.getLogger()
                console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
                assert len(console_handlers) == 1
                expected_level = logging.WARNING if 'PYTEST_CURRENT_TEST' in os.environ else logging.INFO
                assert console_handlers[0].level == expected_level
                for handler in logger.handlers[:]:
                    try:
                        handler.close()
                    except Exception:
                        pass
                logger.handlers = []
                os.chmod(str(read_only_dir), 0o777)
        finally:
            config.LOG_DIR = original_log_dir

def test_multiple_setup_calls(temp_dir: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_log_dir = config.LOG_DIR
        try:
            config.LOG_DIR = temp_dir / "logs"
            logging_setup.setup_logging()
            logger = logging.getLogger()
            assert len(logger.handlers) == 2
            logging_setup.setup_logging()
            assert len(logger.handlers) == 2
            first_handlers = logger.handlers[:]
            logging_setup.setup_logging()
            second_handlers = logger.handlers[:]
            assert first_handlers != second_handlers
        finally:
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
            logger.handlers = []
            config.LOG_DIR = original_log_dir

def test_log_formatting(temp_dir: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_log_dir = config.LOG_DIR
        try:
            config.LOG_DIR = temp_dir / "logs"
            logging_setup.setup_logging()
            logger = logging.getLogger("test_logger")
            logger.warning("Test warning message")
            logger.error("Test error message")
            log_file = config.LOG_DIR / "logfile.txt"
            with open(log_file, 'r') as f:
                log_contents = f.read()
                assert " - WARNING  - [test_logger:" in log_contents
                assert " - ERROR    - [test_logger:" in log_contents
                assert "Test warning message" in log_contents
                assert "Test error message" in log_contents
        finally:
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
            logger.handlers = []
            config.LOG_DIR = original_log_dir 

def test_logger_initialization(setup_logging, temp_log_dir):
    """Test logger initialization and basic configuration."""
    # Get root logger
    logger = logging.getLogger()
    
    # Verify logger has handlers
    assert len(logger.handlers) > 0
    
    # Verify log file was created
    log_files = list(temp_log_dir.glob('*.log'))
    assert len(log_files) > 0
    
    # Verify log file is writable
    log_file = log_files[0]
    assert log_file.stat().st_size == 0  # Should be empty initially
    
    # Test logging
    test_message = "Test log message"
    logger.info(test_message)
    
    # Verify message was written
    with open(log_file, 'r') as f:
        content = f.read()
        assert test_message in content

def test_log_rotation(setup_logging, temp_log_dir):
    """Test log rotation functionality."""
    logger = logging.getLogger()
    
    # Find the file handler
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            file_handler = handler
            break
    
    assert file_handler is not None
    
    # Test log rotation by writing large amounts of data
    large_message = "x" * 1000  # 1KB message
    for _ in range(1000):  # Write 1MB of data
        logger.info(large_message)
    
    # Verify rotation occurred
    log_files = list(temp_log_dir.glob('*.log*'))
    assert len(log_files) > 1  # Should have rotated files

def test_log_levels(setup_logging):
    """Test different log levels."""
    logger = logging.getLogger()
    
    # Test all log levels
    test_messages = {
        logging.DEBUG: "Debug message",
        logging.INFO: "Info message",
        logging.WARNING: "Warning message",
        logging.ERROR: "Error message",
        logging.CRITICAL: "Critical message"
    }
    
    for level, message in test_messages.items():
        logger.log(level, message)
    
    # Verify messages were written
    log_file = next(Path(logging_setup.LOG_DIR).glob('*.log'))
    with open(log_file, 'r') as f:
        content = f.read()
        for message in test_messages.values():
            assert message in content

def test_log_formatting(setup_logging):
    """Test log message formatting."""
    logger = logging.getLogger()
    
    # Test logging with different types of data
    test_data = {
        'string': 'test string',
        'number': 42,
        'list': [1, 2, 3],
        'dict': {'key': 'value'},
        'exception': Exception('test exception')
    }
    
    # Log each type
    logger.info("String: %s", test_data['string'])
    logger.info("Number: %d", test_data['number'])
    logger.info("List: %s", test_data['list'])
    logger.info("Dict: %s", test_data['dict'])
    
    try:
        raise test_data['exception']
    except Exception as e:
        logger.exception("Exception occurred")
    
    # Verify formatting
    log_file = next(Path(logging_setup.LOG_DIR).glob('*.log'))
    with open(log_file, 'r') as f:
        content = f.read()
        assert test_data['string'] in content
        assert str(test_data['number']) in content
        assert str(test_data['list']) in content
        assert str(test_data['dict']) in content
        assert 'Exception occurred' in content
        assert 'test exception' in content

def test_log_directory_creation(temp_log_dir):
    """Test log directory creation."""
    # Remove directory if it exists
    if temp_log_dir.exists():
        shutil.rmtree(temp_log_dir)
    
    # Set up logging with non-existent directory
    with patch('logging_setup.LOG_DIR', temp_log_dir):
        logging_setup.setup_logging()
        
        # Verify directory was created
        assert temp_log_dir.exists()
        assert temp_log_dir.is_dir()
        
        # Verify log file was created
        log_files = list(temp_log_dir.glob('*.log'))
        assert len(log_files) > 0

def test_multiple_loggers(setup_logging):
    """Test multiple logger instances."""
    # Create multiple loggers
    loggers = {
        'main': logging.getLogger('main'),
        'data': logging.getLogger('data'),
        'analysis': logging.getLogger('analysis')
    }
    
    # Log from each logger
    for name, logger in loggers.items():
        logger.info(f"Test message from {name} logger")
    
    # Verify all messages were written
    log_file = next(Path(logging_setup.LOG_DIR).glob('*.log'))
    with open(log_file, 'r') as f:
        content = f.read()
        for name in loggers:
            assert f"Test message from {name} logger" in content

def test_log_cleanup(setup_logging, temp_log_dir):
    """Test log cleanup functionality."""
    logger = logging.getLogger()
    
    # Create some test log files
    test_files = [
        temp_log_dir / f"test_{i}.log" for i in range(5)
    ]
    for file in test_files:
        file.touch()
    
    # Set up logging with cleanup
    with patch('logging_setup.LOG_DIR', temp_log_dir):
        logging_setup.setup_logging(cleanup_old_logs=True)
        
        # Verify only current log file exists
        log_files = list(temp_log_dir.glob('*.log'))
        assert len(log_files) == 1  # Only the current log file should remain

def test_error_handling(setup_logging, temp_log_dir):
    """Test error handling in logging setup."""
    # Test with invalid log directory
    with patch('logging_setup.LOG_DIR', Path('/invalid/path')):
        with pytest.raises(OSError):
            logging_setup.setup_logging()
    
    # Test with read-only directory
    os.chmod(temp_log_dir, 0o444)  # Make directory read-only
    with patch('logging_setup.LOG_DIR', temp_log_dir):
        with pytest.raises(OSError):
            logging_setup.setup_logging()
    os.chmod(temp_log_dir, 0o755)  # Restore permissions
    
    # Test with invalid log level
    with patch('logging_setup.LOG_LEVEL', 'INVALID_LEVEL'):
        with pytest.raises(ValueError):
            logging_setup.setup_logging() 