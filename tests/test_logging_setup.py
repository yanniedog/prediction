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

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

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
    # Temporarily modify config.LOG_DIR
    original_log_dir = config.LOG_DIR
    try:
        config.LOG_DIR = temp_dir / "logs"
        
        # Test default setup
        logging_setup.setup_logging()
        logger = logging.getLogger()
        
        # Verify handlers
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        # Only count StreamHandlers that are not also FileHandlers (i.e., the true console handler)
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert len(console_handlers) == 1
        
        # Verify log file creation
        log_file = config.LOG_DIR / "logfile.txt"
        assert log_file.exists()
        
        # Verify log levels
        assert file_handlers[0].level == logging.WARNING
        assert console_handlers[0].level == logging.INFO
        assert logger.level == logging.INFO  # Minimum of file and console levels
        
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
        # Close all handlers to release file locks before deleting temp_dir
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
        config.LOG_DIR = original_log_dir

def test_setup_logging_custom_levels(temp_dir: Path) -> None:
    """Test logging setup with custom levels."""
    original_log_dir = config.LOG_DIR
    try:
        config.LOG_DIR = temp_dir / "logs"
        
        # Test with custom levels
        logging_setup.setup_logging(
            file_level=logging.DEBUG,
            console_level=logging.WARNING
        )
        logger = logging.getLogger()
        
        # Verify levels
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        # Only count StreamHandlers that are not also FileHandlers (i.e., the true console handler)
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert file_handlers[0].level == logging.DEBUG
        assert console_handlers[0].level == logging.WARNING
        assert logger.level == logging.DEBUG  # Minimum of file and console levels
        
        # Test logging
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        # Verify log file contents
        log_file = config.LOG_DIR / "logfile.txt"
        with open(log_file, 'r') as f:
            log_contents = f.read()
            assert "Debug message" in log_contents    # Debug logged to file
            assert "Info message" in log_contents     # Info logged to file
            assert "Warning message" in log_contents  # Warning logged to file
    finally:
        # Close all handlers to release file locks before deleting temp_dir
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
        config.LOG_DIR = original_log_dir

def test_setup_logging_file_modes(temp_dir: Path) -> None:
    """Test logging setup with different file modes."""
    original_log_dir = config.LOG_DIR
    try:
        config.LOG_DIR = temp_dir / "logs"
        
        # Test append mode
        logging_setup.setup_logging(file_mode='a')
        logger = logging.getLogger()
        logger.warning("First message")
        
        # Reset logging
        logging.getLogger().handlers = []
        
        # Setup again in append mode
        logging_setup.setup_logging(file_mode='a')
        logger = logging.getLogger()
        logger.warning("Second message")
        
        # Verify both messages in file
        log_file = config.LOG_DIR / "logfile.txt"
        with open(log_file, 'r') as f:
            log_contents = f.read()
            assert "First message" in log_contents
            assert "Second message" in log_contents
        
        # Test overwrite mode
        logging.getLogger().handlers = []
        logging_setup.setup_logging(file_mode='w')
        logger = logging.getLogger()
        logger.warning("Third message")
        
        # Verify only new message in file
        with open(log_file, 'r') as f:
            log_contents = f.read()
            assert "First message" not in log_contents
            assert "Second message" not in log_contents
            assert "Third message" in log_contents
        
        # Test invalid mode
        logging.getLogger().handlers = []
        logging_setup.setup_logging(file_mode='invalid')
        logger = logging.getLogger()
        logger.warning("Fourth message")
        
        # Verify file was overwritten (default mode)
        with open(log_file, 'r') as f:
            log_contents = f.read()
            assert "Third message" not in log_contents
            assert "Fourth message" in log_contents
    finally:
        # Close all handlers to release file locks before deleting temp_dir
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
        config.LOG_DIR = original_log_dir

def test_set_console_log_level() -> None:
    """Test dynamic console log level changes."""
    logging_setup.setup_logging()
    logger = logging.getLogger()
    
    # Test setting to DEBUG
    logging_setup.set_console_log_level(logging.DEBUG)
    # Only count StreamHandlers that are not also FileHandlers (i.e., the true console handler)
    console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert console_handlers[0].level == logging.DEBUG
    assert logger.level == min(file_handlers[0].level, console_handlers[0].level)
    
    # Test setting to WARNING
    logging_setup.set_console_log_level(logging.WARNING)
    assert console_handlers[0].level == logging.WARNING
    assert logger.level == min(file_handlers[0].level, console_handlers[0].level)
    
    # Test setting to ERROR
    logging_setup.set_console_log_level(logging.ERROR)
    assert console_handlers[0].level == logging.ERROR
    assert logger.level == min(file_handlers[0].level, console_handlers[0].level)

def test_reset_console_log_level() -> None:
    """Test resetting console log level to default."""
    # Setup with custom console level
    logging_setup.setup_logging(console_level=logging.WARNING)
    logger = logging.getLogger()
    
    # Change level
    logging_setup.set_console_log_level(logging.DEBUG)
    # Only count StreamHandlers that are not also FileHandlers (i.e., the true console handler)
    console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    assert console_handlers[0].level == logging.DEBUG
    
    # Reset to default
    logging_setup.reset_console_log_level()
    assert console_handlers[0].level == logging.WARNING
    assert logger.level == logging.WARNING  # Minimum of file and console levels

def test_library_quieting() -> None:
    """Test that noisy libraries are set to WARNING level."""
    logging_setup.setup_logging()
    
    # Test each library's logger level
    libraries_to_quiet = [
        'matplotlib', 'requests', 'urllib3', 'PIL', 'skopt', 'statsmodels',
        'numexpr', 'asyncio'
    ]
    
    for lib_name in libraries_to_quiet:
        lib_logger = logging.getLogger(lib_name)
        assert lib_logger.level == logging.WARNING

def test_error_handling(temp_dir: Path) -> None:
    """Test error handling in logging setup."""
    original_log_dir = config.LOG_DIR
    try:
        # Test with read-only directory
        if hasattr(os, 'chmod'):
            read_only_dir = temp_dir / "readonly"
            read_only_dir.mkdir()
            os.chmod(str(read_only_dir), 0o444)  # Read-only
            
            config.LOG_DIR = read_only_dir
            # Should not raise exception, but should print error
            logging_setup.setup_logging()
            
            # Verify console handler still works
            logger = logging.getLogger()
            console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
            assert len(console_handlers) == 1
            assert console_handlers[0].level == logging.INFO
            
            os.chmod(str(read_only_dir), 0o777)  # Restore permissions
    finally:
        config.LOG_DIR = original_log_dir

def test_multiple_setup_calls(temp_dir: Path) -> None:
    """Test multiple calls to setup_logging."""
    original_log_dir = config.LOG_DIR
    try:
        config.LOG_DIR = temp_dir / "logs"
        
        # First setup
        logging_setup.setup_logging()
        logger = logging.getLogger()
        assert len(logger.handlers) == 2
        
        # Second setup should clear previous handlers
        logging_setup.setup_logging()
        assert len(logger.handlers) == 2  # Still 2 handlers, but new ones
        
        # Verify handlers are new instances
        first_handlers = logger.handlers[:]
        logging_setup.setup_logging()
        second_handlers = logger.handlers[:]
        assert first_handlers != second_handlers
    finally:
        config.LOG_DIR = original_log_dir

def test_log_formatting(temp_dir: Path) -> None:
    """Test log message formatting."""
    original_log_dir = config.LOG_DIR
    try:
        config.LOG_DIR = temp_dir / "logs"
        logging_setup.setup_logging()
        logger = logging.getLogger("test_logger")
        
        # Log messages at different levels
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Verify file format
        log_file = config.LOG_DIR / "logfile.txt"
        with open(log_file, 'r') as f:
            log_contents = f.read()
            # Check for timestamp format
            assert " - WARNING  - [test_logger:" in log_contents
            assert " - ERROR    - [test_logger:" in log_contents
            assert "Test warning message" in log_contents
            assert "Test error message" in log_contents
    finally:
        config.LOG_DIR = original_log_dir 