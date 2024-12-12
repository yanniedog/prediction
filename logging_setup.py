# logging_setup.py
import logging
import sys
from pathlib import Path

def configure_logging(log_file='prediction.log'):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    log_path = Path.cwd() / log_file

    try:
        # Configure file handler
        file_handler = logging.FileHandler(log_path, 'w')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(levelname)s - [%(filename)s:%(lineno)d(%(funcName)s)]: %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Suppress unwanted logs
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('font_manager').setLevel(logging.WARNING)

        # Redirect print to both console and log file
        class StreamToLogger:
            def __init__(self, stream, log_func):
                self.stream = stream
                self.log_func = log_func

            def write(self, msg):
                if msg.strip():
                    self.log_func(msg.strip())
                self.stream.write(msg)

            def flush(self):
                self.stream.flush()

        sys.stdout = StreamToLogger(sys.stdout, logger.info)
        sys.stderr = StreamToLogger(sys.stderr, logger.error)

        # Handle uncaught exceptions
        def exception_handler(exc_type, exc_value, exc_traceback):
            if not issubclass(exc_type, KeyboardInterrupt):
                logger.error(
                    "Uncaught exception",
                    exc_info=(exc_type, exc_value, exc_traceback),
                )

        sys.excepthook = exception_handler

        logger.info(f"Logging configured. Log file at: {log_path.resolve()}")
        return log_path

    except Exception as e:
        print(f"Logging setup failed: {e}")
        sys.exit(1)
