import os
import sys
import subprocess
from pathlib import Path

def run_backup_cleanup() -> None:
    backup_script = Path(__file__).resolve().parent / 'backup_cleanup.py'
    if not backup_script.exists():
        sys.exit(1)
    result = subprocess.run([sys.executable, str(backup_script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)