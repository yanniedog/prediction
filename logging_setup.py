import logging, sys
from pathlib import Path
from logging import StreamHandler

def configure_logging(log_file='prediction.log'):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    log_path = Path.cwd() / log_file
    try:
        f_handler = logging.FileHandler(log_path, 'w')
        f_handler.setLevel(logging.INFO)
        c_handler = StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        for handler in [f_handler, c_handler]:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        class StreamToLogger:
            def __init__(self, logger, level):
                self.logger, self.level = logger, level
            def write(self, msg):
                if msg.strip(): self.logger.log(self.level, msg.strip())
            def flush(self): pass
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
        logger.info(f"Logging configured. Log file at: {log_path.resolve()}")
    except Exception as e:
        print(f"Logging setup failed: {e}")
        sys.exit(1)
