import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import runpy
import subprocess

if 'VIRTUAL_ENV' not in os.environ:
    venv_path = Path.cwd() / 'venv'
    if not venv_path.is_dir():
        logging.error("Virtual environment not found. Please create it using 'python -m venv venv'.")
        sys.exit(1)

    activate_script = venv_path / ('Scripts' / 'activate.bat' if sys.platform == 'win32' else 'bin' / 'activate')
    try:
        subprocess.run([str(activate_script)], shell=True, check=True)
        logging.info("Virtual environment activated.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to activate virtual environment: {e}")
        sys.exit(1)

sys.path.append(str(Path.cwd() / 'scripts'))

for f in Path.cwd().glob('*.log'):
    f.unlink()

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
working_dir_name = Path.cwd().name
log_filename = f"{working_dir_name}_{timestamp}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

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

sys.stdout = DoubleWriter(sys.__stdout__, sys.__stderr__, logging.getLogger())
sys.stderr = DoubleWriter(sys.__stderr__, sys.__stderr__, logging.getLogger())

try:
    start_path = str(Path.cwd() / 'scripts' / 'start.py')
    runpy.run_path(start_path, run_name='__main__')
except SystemExit as e:
    sys.exit(e.code)
except Exception as e:
    logging.exception("An error occurred while running the script.")
    sys.exit(1)