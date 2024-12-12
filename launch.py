import os, sys, logging, runpy
from pathlib import Path
from datetime import datetime

for f in Path.cwd().glob('*.log'): f.unlink()

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
log_filename = f"{Path.cwd().name}_{timestamp}.log"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = []
file_handler = logging.FileHandler(log_filename, 'w')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class DoubleWriter:
    def __init__(self, stdout, stderr, logger):
        self.stdout, self.stderr, self.logger = stdout, stderr, logger
    def write(self, msg):
        if msg.strip(): self.logger.info(msg.strip())
        self.stdout.write(msg)
    def flush(self):
        self.stdout.flush()
    def isatty(self):
        return self.stdout.isatty()

sys.stdout = DoubleWriter(sys.__stdout__, sys.__stderr__, logger)
sys.stderr = DoubleWriter(sys.__stderr__, sys.__stderr__, logger)

try:
    runpy.run_path("start.py", run_name="__main__")
except SystemExit as e:
    sys.exit(e.code)
except:
    raise
