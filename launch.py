# launch.py
import os, sys, logging, runpy
from pathlib import Path
from datetime import datetime

# Function to delete old log files
def delete_old_logs(log_dir: Path, log_extension: str = ".log"):
    for log_file in log_dir.glob(f"*{log_extension}"):
        try:
            log_file.unlink()
        except Exception as e:
            print(f"Error deleting log file {log_file}: {e}")

# Clean up old logs
current_dir = Path.cwd()
delete_old_logs(current_dir)

# Configure logging
log_filename = f"{current_dir.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
logger = logging.getLogger("unique_logger")
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_filename, 'w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

class StreamLogger:
    def __init__(self, stream, log_func):
        self.stream = stream
        self.log_func = log_func
    def write(self, message):
        if message.strip():
            self.log_func(message.strip())
        self.stream.write(message)
    def flush(self):
        self.stream.flush()

if not isinstance(sys.stdout, StreamLogger):
    sys.stdout = StreamLogger(sys.__stdout__, logger.info)
if not isinstance(sys.stderr, StreamLogger):
    sys.stderr = StreamLogger(sys.__stderr__, logger.error)

try:
    from sqlite_data_manager import initialize_database
    from config import DB_PATH
    initialize_database(DB_PATH)
    runpy.run_path("start.py", run_name="__main__")
except SystemExit as e:
    sys.exit(e.code)
except:
    logger.exception("Uncaught exception")
