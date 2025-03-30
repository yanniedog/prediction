# logging_setup.py
import logging
import sys
from datetime import datetime
import config # Assuming config.py defines LOG_DIR

# Global variable to hold the console handler reference
_console_handler = None
# ** Store the default CONSOLE level (INFO) **
_default_console_level = logging.INFO
# ** Define the desired FILE level (WARNING) ** # <--- MODIFIED HERE
_file_log_level = logging.WARNING # <--- MODIFIED HERE

def setup_logging():
    """Configures logging with levels for console/file, overwriting log file."""
    global _console_handler, _default_console_level, _file_log_level
    log_filename = config.LOG_DIR / "logfile.txt"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Define formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # Get root logger and clear existing handlers
    logger = logging.getLogger()
    if logger.hasHandlers():
        # Close existing handlers before removing to release file locks etc.
        for handler in logger.handlers[:]:
            try:
                handler.close()
            except Exception as e:
                print(f"Error closing handler {handler}: {e}", file=sys.stderr)
            logger.removeHandler(handler)
        logging.shutdown() # Ensure logging subsystem is reset properly

    # Set root logger level to the lowest level needed by any handler (now INFO, as console is INFO)
    logger.setLevel(min(_default_console_level, _file_log_level)) # This ensures console INFO still works

    # File Handler (WARNING+, Overwrite 'w' or Append 'a') # <--- MODIFIED HERE
    try:
        # ***** CHANGE THIS MODE TO 'a' IF YOU WANT TO APPEND *****
        file_handler_mode = 'w'
        # **********************************************************
        file_handler = logging.FileHandler(log_filename, mode=file_handler_mode, encoding='utf-8')
        # ***** Set File Handler Level to WARNING ***** # <--- MODIFIED HERE
        file_handler.setLevel(_file_log_level)
        # ******************************************
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logging: {e}", file=sys.stderr)

    # Console Handler (INFO+ by default)
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(_default_console_level) # Set initial console level
    _console_handler.setFormatter(console_formatter)
    logger.addHandler(_console_handler)

    # Reduce verbosity of libraries (Set their level higher)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('skopt').setLevel(logging.WARNING)
    # Add statsmodels if it becomes verbose
    logging.getLogger('statsmodels').setLevel(logging.WARNING)


    # Use the root logger to log initialization info (this will go to BOTH handlers initially)
    # --- MODIFIED Log Message ---
    logger.info(f"Logging initialized (Console: {logging.getLevelName(_default_console_level)}, File: {logging.getLevelName(_file_log_level)}, Mode: '{file_handler_mode}', Path: {log_filename})")

def set_console_log_level(level: int):
    """Sets the logging level for the console handler."""
    global _console_handler
    if _console_handler:
        current_level_name = logging.getLevelName(_console_handler.level)
        new_level_name = logging.getLevelName(level)
        # Use root logger for this message, ensuring it goes to file even if console is high
        logging.getLogger().debug(f"Changing console log level from {current_level_name} to {new_level_name}")
        _console_handler.setLevel(level)
        # Log the change confirmation at INFO level (will always show on console when resetting to INFO)
        logging.getLogger().info(f"Console log level set to {new_level_name}")
    else:
        logging.getLogger().error("Console handler not initialized. Cannot set level.")

def reset_console_log_level():
    """Resets the console logging level to its default."""
    global _console_handler, _default_console_level
    if _console_handler:
        current_level_name = logging.getLevelName(_console_handler.level)
        default_level_name = logging.getLevelName(_default_console_level)
        # Use root logger for this message
        logging.getLogger().debug(f"Resetting console log level from {current_level_name} to {default_level_name}")
        _console_handler.setLevel(_default_console_level)
        # Log the reset at the default level (INFO)
        logging.getLogger().info(f"Console log level reset to {default_level_name}")
    else:
        logging.getLogger().error("Console handler not initialized. Cannot reset level.")