# logging_setup.py
import logging
import sys
from pathlib import Path

def configure_logging(log_file='prediction.log'):
    log_path = Path.cwd() / log_file
    if log_path.exists():
        log_path.unlink()
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, 'w')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - Line %(lineno)d - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    class StreamToLogger:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level

        def write(self, message):
            if message.strip():
                self.logger.log(self.level, message.strip())

        def flush(self):
            pass

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    def exception_handler(exc_type, exc_value, exc_traceback):
        if not issubclass(exc_type, KeyboardInterrupt):
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = exception_handler

    return log_path
