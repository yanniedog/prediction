# logging_setup.py

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
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
        
    log_path = Path.cwd() / log_file
        
    try:
        f_handler = logging.FileHandler(log_path, mode='w')
        f_handler.setLevel(logging.INFO)
            
        c_handler = StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
            
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
            
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
            
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
            
        logger.info(f"Logging configured. Log file will be at: {log_path.resolve()}")
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        sys.exit(1)