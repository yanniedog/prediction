# logging_setup.py
import logging
import sys
from pathlib import Path
from logging import StreamHandler

def configure_logging(log_file='prediction.log'):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    log_path = Path.cwd() / log_file

    try:
        # Configure file handler
        file_handler = logging.FileHandler(log_path, 'w')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - Line %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Suppress console debug output
        console_handler = StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Redirect stdout and stderr to logger
        class StreamToLogger:
            def __init__(self, logger, level):
                self.logger = logger
                self.level = level

            def write(self, msg):
                if msg.strip():
                    self.logger.log(self.level, msg.strip())

            def flush(self):
                pass

        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

        # Handle uncaught exceptions
        def exception_handler(exc_type, exc_value, exc_traceback):
            if not issubclass(exc_type, KeyboardInterrupt):
                logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = exception_handler

        logger.info(f"Logging configured. Log file: {log_path.resolve()}")
        return log_path

    except Exception as e:
        print(f"Logging setup failed: {e}")
        sys.exit(1)
