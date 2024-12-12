# backup_utils.py
import subprocess, sys
from pathlib import Path

def run_backup_cleanup():
    try:
        backup_script = Path(__file__).resolve().parent / 'backup_cleanup.py'
        if not backup_script.exists():
            print(f"Backup script not found: {backup_script}")
            sys.exit(1)
        print(f"Executing backup script: {backup_script}")
        result = subprocess.run([sys.executable, str(backup_script)], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"Backup failed with code {result.returncode}")
            sys.exit(result.returncode)
        print("Backup executed successfully.")
    except Exception as e:
        print(f"Backup utils error: {e}")
        sys.exit(1)
