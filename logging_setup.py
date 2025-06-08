# logging_setup.py
import logging
import sys
from datetime import datetime
import config # Assuming config.py defines LOG_DIR

# Global variable to hold the console handler reference
_console_handler = None
# Store the default CONSOLE level (INFO)
_default_console_level = logging.INFO
# Define the desired FILE level (WARNING or DEBUG based on needs)
# Use WARNING for production/normal runs, DEBUG for detailed tracing
_file_log_level = logging.WARNING # Default to WARNING

def setup_logging(file_level=logging.WARNING, console_level=logging.INFO, file_mode='w'):
    """Configures logging with levels for console/file, overwriting log file by default."""
    global _console_handler, _default_console_level, _file_log_level
    _default_console_level = console_level # Update default if changed
    _file_log_level = file_level        # Update file level if changed

    log_filename = config.LOG_DIR / "logfile.txt"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

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
        # Reset logging system state if necessary
        # logging.shutdown() # Can be too aggressive, only use if handlers persist strangely

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

# Example usage within main.py (adjust levels as needed):
# import logging_setup
# logging_setup.setup_logging(file_level=logging.DEBUG, console_level=logging.INFO, file_mode='a')
# ... later ...
# logging_setup.set_console_log_level(logging.DEBUG)
# ... later ...
# logging_setup.reset_console_log_level()