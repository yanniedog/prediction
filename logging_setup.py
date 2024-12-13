# logging_setup.py
import logging
import sys
from pathlib import Path
import inspect

class TaskAwareFormatter(logging.Formatter):
    def format(self, record):
        for frame in inspect.stack():
            module = inspect.getmodule(frame[0])
            if module and not module.__name__.startswith('logging'):
                record.filename = Path(module.__file__).name
                record.funcName = frame.function
                record.lineno = frame.lineno
                break
        return super().format(record)

class StreamToLogger:
    def __init__(self, stream):
        self.stream = stream

    def write(self, msg):
        if msg.strip():
            self.stream.write(msg)

    def flush(self):
        self.stream.flush()

_configured = False

def configure_logging(log_file_prefix='predictions'):
    global _configured
    if _configured:
        return

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    log_path = Path.cwd() / f"{log_file_prefix}_{Path.cwd().stem}.log"

    try:
        file_handler = logging.FileHandler(log_path, 'w')
        file_handler.setLevel(logging.DEBUG)
        formatter = TaskAwareFormatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d(%(funcName)s)]: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))  # Plain messages to screen
        logger.addHandler(stream_handler)

        sys.stdout = StreamToLogger(sys.stdout)
        sys.stderr = StreamToLogger(sys.stderr)

        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('font_manager').setLevel(logging.WARNING)

        def exception_handler(exc_type, exc_value, exc_traceback):
            if not issubclass(exc_type, KeyboardInterrupt):
                logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = exception_handler

        logger.info(f"Logging configured. Log file at: {log_path.resolve()}")
        _configured = True
        return log_path

    except Exception as e:
        print(f"Logging setup failed: {e}")
        sys.exit(1)
