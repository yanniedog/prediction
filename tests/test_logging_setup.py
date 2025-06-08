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
        import time
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