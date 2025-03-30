# logging_setup.py
import logging
import sys
from datetime import datetime
import config # Assuming config.py defines LOG_DIR

def setup_logging():
    """Configures logging with levels for console/file, overwriting log file."""
    log_filename = config.LOG_DIR / "logfile.txt"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    logger = logging.getLogger()
    # Clear existing handlers before adding new ones
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG) # Root level

    # File Handler (DEBUG+, Overwrite)
    try:
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logging: {e}", file=sys.stderr)

    # Console Handler (INFO+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Reduce verbosity of libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('skopt').setLevel(logging.WARNING) # Silence skopt warnings if desired

    logger.info(f"Logging initialized (Console: INFO+, File: DEBUG+, Overwriting: {log_filename})")
