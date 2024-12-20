# launch.py
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import runpy

for f in Path.cwd().glob('*.log'):
    f.unlink()

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
working_dir_name = Path.cwd().name
log_filename = f"{working_dir_name}_{timestamp}.log"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
for h in logger.handlers[:]:
    logger.removeHandler(h)
file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class DoubleWriter:
    def __init__(self, stdout, stderr, logger):
        self.stdout = stdout
        self.stderr = stderr
        self.logger = logger
    def write(self, msg):
        if msg.strip():
            self.logger.info(msg.strip())
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