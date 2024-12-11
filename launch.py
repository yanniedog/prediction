# filename: launch.py
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# On Windows, 'clear' isn't recognized; use 'cls' instead
if os.name == 'nt':
    os.system('cls')
else:
    os.system('clear')

# Remove existing log files before launching
for f in Path.cwd().glob('*.log'):
    f.unlink()

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
working_dir_name = Path.cwd().name
log_filename = f"{working_dir_name}_{timestamp}.log"

# On Windows, simply run start.py directly so that output is displayed on screen and logging is handled by start.py.
# On Unix-like systems, similarly run without using 'script' command.
if os.name == 'nt':
    # Just run start.py and let start.py handle logging and console output
    subprocess.run([sys.executable, "start.py"], check=True)
else:
    # On Unix-like systems, similarly run without 'script' to allow console output to appear
    subprocess.run([sys.executable, "start.py"], check=True)