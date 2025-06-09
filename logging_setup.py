# logging_setup.py
import logging
import sys
from datetime import datetime
import config # Assuming config.py defines LOG_DIR
import os
from typing import List, Dict
from pathlib import Path

# Global variable to hold the console handler reference
_console_handler = None
# Store the default CONSOLE level (INFO)
_default_console_level = logging.INFO
# Define the desired FILE level (WARNING or DEBUG based on needs)
# Use WARNING for production/normal runs, DEBUG for detailed tracing
_file_log_level = logging.WARNING # Default to WARNING

LOG_DIR = config.LOG_DIR
LOG_LEVEL = logging.INFO  # Default log level for compatibility with tests

def setup_logging(file_level=logging.WARNING, console_level=logging.INFO, file_mode='w', cleanup_old_logs=False):
    """Configures logging with levels for console/file, overwriting log file by default."""
    global _console_handler, _default_console_level, _file_log_level
    # Force WARNING for console if running under pytest
    if 'PYTEST_CURRENT_TEST' in os.environ:
        console_level = logging.WARNING
        # Keep file level as specified, don't override to INFO
    _default_console_level = console_level # Update default if changed
    _file_log_level = file_level        # Update file level if changed

    # Get the current LOG_DIR from config (allows tests to patch it)
    current_log_dir = config.LOG_DIR
    
    # Ensure LOG_DIR is a Path object and create it
    if isinstance(current_log_dir, str):
        log_dir = Path(current_log_dir)
    else:
        log_dir = current_log_dir
    
    # Create the log directory if it doesn't exist
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"CRITICAL ERROR creating log directory {log_dir}: {e}", file=sys.stderr)
        # Fallback to current directory
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old log files if requested
    if cleanup_old_logs:
        try:
            for log_file in log_dir.glob('*.txt'):
                if log_file.name != 'logfile.txt':  # Keep current log file
                    log_file.unlink()
        except Exception as e:
            print(f"Warning: Could not cleanup old log files: {e}", file=sys.stderr)

    log_filename = log_dir / "logfile.txt"
    
    # Define formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # Get root logger and clear existing handlers to prevent duplicate logs
    logger = logging.getLogger()
    if logger.hasHandlers():
        # Close existing handlers before removing to release file locks etc.
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler) or isinstance(handler, logging.StreamHandler):
                try:
                    handler.close()
                except Exception as e:
                    # Use basic print for errors during logging setup itself
                    print(f"Error closing handler {handler}: {e}", file=sys.stderr)
                logger.removeHandler(handler)

    # Set root logger level to the lowest level needed by any handler
    # This ensures messages pass through the root logger to potentially reach handlers
    lowest_level = min(_file_log_level, _default_console_level)
    logger.setLevel(lowest_level)

    # --- File Handler ---
    try:
        # Validate file mode
        valid_modes = ['w', 'a']
        if file_mode not in valid_modes:
             print(f"Warning: Invalid file_mode '{file_mode}'. Using 'w' (overwrite).", file=sys.stderr)
             file_mode = 'w'

        # Use utf-8 encoding for broader compatibility
        file_handler = logging.FileHandler(log_filename, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(_file_log_level) # Set level for this specific handler
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Force a flush to ensure the file is created and writable
        file_handler.flush()
        
        # Write a test message to ensure the file is working
        logger.info(f"Logging initialized (Console: {logging.getLevelName(_default_console_level)}, File: {logging.getLevelName(_file_log_level)}, Mode: '{file_mode}', Path: {log_filename})")
        
        # Force another flush after the test message
        file_handler.flush()
        
    except Exception as e:
        print(f"CRITICAL ERROR setting up file logging: {e}", file=sys.stderr)
        # Consider exiting if file logging is essential and fails
        # sys.exit(1)

    # --- Console Handler ---
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(_default_console_level) # Set initial console level
    _console_handler.setFormatter(console_formatter)
    logger.addHandler(_console_handler)

    # --- Reduce Verbosity of Common Libraries ---
    # Set the log level higher for noisy libraries to keep logs cleaner
    libraries_to_quiet = [
        'matplotlib', 'requests', 'urllib3', 'PIL', 'skopt', 'statsmodels',
        'numexpr', # Often chatty with pandas/numpy
        'asyncio'  # Can be verbose if used by dependencies
    ]
    for lib_name in libraries_to_quiet:
        logging.getLogger(lib_name).setLevel(logging.WARNING)

    # Log initialization info (will go to handlers based on their levels)
    logger.info(f"Logging initialized (Console: {logging.getLevelName(_default_console_level)}, File: {logging.getLevelName(_file_log_level)}, Mode: '{file_mode}', Path: {log_filename})")
    
    # Force flush all handlers to ensure messages are written
    for handler in logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()

def set_console_log_level(level: int):
    """Sets the logging level for the console handler dynamically."""
    global _console_handler
    root_logger = logging.getLogger() # Get root logger
    if _console_handler:
        current_level_name = logging.getLevelName(_console_handler.level)
        new_level_name = logging.getLevelName(level)
        # Use root logger's info level for this message, ensuring it likely goes to file
        root_logger.info(f"Changing console log level from {current_level_name} to {new_level_name}")
        _console_handler.setLevel(level)
        # Log the change confirmation at INFO level (will show on console if >= INFO)
        root_logger.info(f"Console log level set to {new_level_name}")
        # Adjust root logger level if necessary to allow messages to pass through
        root_logger.setLevel(min(_file_log_level, level))
    else:
        root_logger.error("Console handler not initialized. Cannot set level.")

def reset_console_log_level():
    """Resets the console logging level to its default."""
    global _console_handler, _default_console_level
    root_logger = logging.getLogger()
    if _console_handler:
        current_level_name = logging.getLevelName(_console_handler.level)
        default_level_name = logging.getLevelName(_default_console_level)
        root_logger.info(f"Resetting console log level from {current_level_name} to {default_level_name}")
        _console_handler.setLevel(_default_console_level)
        # Log the reset at the default level (INFO)
        root_logger.info(f"Console log level reset to {default_level_name}")
        # Adjust root logger level back
        root_logger.setLevel(min(_file_log_level, _default_console_level))
    else:
        root_logger.error("Console handler not initialized. Cannot reset level.")

# Module-level functions for backward compatibility
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Module-level function to setup a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def create_log_directory(path: str) -> bool:
    """Module-level function to create log directory."""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception:
        return False

def configure_log_formatting(formatter: logging.Formatter) -> None:
    """Module-level function to configure log formatting."""
    # This is a placeholder for any formatting configuration
    pass

def setup_log_rotation(logger: logging.Logger, max_bytes: int = 1024*1024, backup_count: int = 5) -> None:
    """Module-level function to setup log rotation."""
    # This is a placeholder for log rotation setup
    pass

def set_log_levels(logger: logging.Logger, level: int) -> None:
    """Module-level function to set log levels."""
    logger.setLevel(level)

def cleanup_logs(path: str) -> bool:
    """Module-level function to cleanup logs."""
    try:
        # This is a placeholder for log cleanup logic
        return True
    except Exception:
        return False

def setup_multiple_loggers(names: List[str]) -> Dict[str, logging.Logger]:
    """Module-level function to setup multiple loggers."""
    loggers = {}
    for name in names:
        loggers[name] = setup_logger(name)
    return loggers

def force_flush_logs():
    """Force flush all log handlers to ensure messages are written."""
    logger = logging.getLogger()
    for handler in logger.handlers:
        try:
            if hasattr(handler, 'flush'):
                handler.flush()
        except Exception:
            pass

def cleanup_logging():
    """Clean up logging handlers to prevent ResourceWarnings."""
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        try:
            if hasattr(handler, 'flush'):
                handler.flush()
            if hasattr(handler, 'close'):
                handler.close()
        except Exception:
            pass
        try:
            logger.removeHandler(handler)
        except Exception:
            pass
    logger.handlers = []

def close_file_handlers():
    """Specifically close file handlers to prevent ResourceWarnings."""
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass

# Example usage within main.py (adjust levels as needed):
# import logging_setup
# logging_setup.setup_logging(file_level=logging.DEBUG, console_level=logging.INFO, file_mode='a')
# ... later ...
# logging_setup.set_console_log_level(logging.DEBUG)
# ... later ...
# logging_setup.reset_console_log_level()