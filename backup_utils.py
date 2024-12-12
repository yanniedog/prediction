# backup_utils.py
import subprocess, sys
from pathlib import Path

def run_backup_cleanup():
    backup_script = Path(__file__).resolve().parent / 'backup_cleanup.py'
    if not backup_script.exists():
        print(f"Backup script not found: {backup_script}")
        sys.exit(1)
    result = subprocess.run([sys.executable, str(backup_script)], text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print("Backup executed successfully.")
