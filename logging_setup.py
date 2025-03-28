# logging_setup.py
import logging
import sys
from datetime import datetime
import config # Assuming config.py defines LOG_DIR

def setup_logging():
    """Configures logging with different levels for console and file, overwriting the log file each run."""
    # --- Use a fixed filename ---
    log_filename = config.LOG_DIR / "prediction.log"

    # Ensure log directory exists
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Define formats
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s' # Simpler format for console
    )

    # Get root logger
    logger = logging.getLogger()
    # Clear existing handlers to prevent duplicate logging if setup_logging is called multiple times (e.g., in tests)
    # Or if re-running in an interactive session.
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG) # Set root logger to lowest level

    # --- File Handler (Detailed, Overwrite Mode) ---
    # Use mode='w' to overwrite the file on each run
    try:
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG) # Log DEBUG and above to file for maximum detail
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file handler for logging: {e}", file=sys.stderr)
        # Continue without file logging if it fails

    # --- Console Handler (Cleaner) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Show only INFO and above on console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Reduce verbosity of noisy libraries if needed
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING) # Pillow can be noisy

    # Use logger.info AFTER handlers are added
    logger.info(f"Logging initialized (Console: INFO+, File: DEBUG+, Overwriting: {log_filename})")