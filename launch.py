# launch.py
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import runpy

if 'VIRTUAL_ENV' not in os.environ:

    venv_path = Path.cwd() / 'venv'
    if not venv_path.is_dir():
        print("Virtual environment not found. Please create it using 'python -m venv venv'.")
        sys.exit(1)

    if sys.platform == 'win32':
        activate_script = venv_path / 'Scripts' / 'activate.bat'
    else:
        activate_script = venv_path / 'bin' / 'activate'

    if sys.platform == 'win32':
        os.system(f'call {activate_script}')
    else:
        os.system(f'source {activate_script}')

    print("Virtual environment activated.")

sys.path.append(str(Path.cwd() / 'scripts'))

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

font_manager_logger = logging.getLogger('matplotlib.font_manager')
font_manager_logger.setLevel(logging.WARNING)

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

    start_path = str(Path.cwd() / 'scripts' / 'start.py')
    runpy.run_path(start_path, run_name='__main__')
except SystemExit as e:
    sys.exit(e.code)
except Exception as e:
    logger.exception("An error occurred while running the script.")
    sys.exit(1)
