# logging_setup.py
import logging
import sys
from pathlib import Path
import inspect

class TaskAwareFormatter(logging.Formatter):
    def format(self, record):
        # Trace the actual caller in the stack to avoid capturing the logging library frames
        for frame in inspect.stack():
            module = inspect.getmodule(frame[0])
            if module and not module.__name__.startswith('logging'):
                record.filename = Path(module.__file__).name
                record.funcName = frame.function
                record.lineno = frame.lineno
                break
        return super().format(record)

def configure_logging(log_file='prediction.log'):
    logger = logging.getLogger()
    logger.handlers = []  # Remove existing handlers to prevent duplication
    logger.setLevel(logging.DEBUG)
    log_path = Path.cwd() / log_file

    try:
        file_handler = logging.FileHandler(log_path, 'w')
        file_handler.setLevel(logging.DEBUG)
        formatter = TaskAwareFormatter('%(levelname)s - [%(filename)s:%(lineno)d(%(funcName)s)]: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Redirect stdout and stderr to logger
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

        # Suppress unwanted logs
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('font_manager').setLevel(logging.WARNING)

        # Handle uncaught exceptions
        def exception_handler(exc_type, exc_value, exc_traceback):
            if not issubclass(exc_type, KeyboardInterrupt):
                logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = exception_handler

        logger.info(f"Logging configured. Log file at: {log_path.resolve()}")
        return log_path

    except Exception as e:
        print(f"Logging setup failed: {e}")
        sys.exit(1)
