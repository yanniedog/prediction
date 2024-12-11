# Filename: logging_setup.py

import logging
from logging import StreamHandler
from pathlib import Path
import sys

def configure_logging(log_file: str = 'prediction.log') -> None:
    """
    Configures logging for the application.
    All log messages are directed to both the console and a log file.
    
    Args:
        log_file (str): Path to the log file.
    """
    logger = logging.getLogger()
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
        
    # Define the absolute path for the log file using current working directory
    log_path = Path.cwd() / log_file
        
    try:
        # Create a FileHandler that overwrites the log file each run
        f_handler = logging.FileHandler(log_path, mode='w')  # Overwrite each run
        f_handler.setLevel(logging.INFO)
            
        # Create a StreamHandler for console output
        c_handler = StreamHandler(sys.stdout)  # Direct to stdout to avoid loop
        c_handler.setLevel(logging.INFO)
            
        # Create formatters and add them to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
            
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
            
        # Optionally, redirect print statements to logging
        class StreamToLogger(object):
            """
            Fake file-like stream object that redirects writes to a logger instance.
            """
            def __init__(self, logger, log_level):
                self.logger = logger
                self.log_level = log_level
                self.linebuf = ''
    
            def write(self, buf):
                for line in buf.rstrip().splitlines():
                    self.logger.log(self.log_level, line.rstrip())
    
            def flush(self):
                pass
    
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
            
        # Log the absolute path of the log file for verification
        logger.info(f"Logging configured. Log file will be at: {log_path.resolve()}")
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        sys.exit(1)