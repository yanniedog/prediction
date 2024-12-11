# filename: backup_utils.py
import os
import sys
import subprocess
from pathlib import Path

def run_backup_cleanup() -> None:
    try:
        backup_script = Path(__file__).resolve().parent / 'backup_cleanup.py'
        if not backup_script.exists():
            print(f"Backup script '{backup_script}' not found.")
            sys.exit(1)
        print(f"Executing backup script: {backup_script}")
        # Capture output from subprocess so it also goes to the logfile
        result = subprocess.run([sys.executable, str(backup_script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Print captured output to ensure it is logged by DoubleWriter
        if result.stdout:
            for line in result.stdout.splitlines():
                print(line)
        if result.stderr:
            for line in result.stderr.splitlines():
                print(line, file=sys.stderr)
        if result.returncode != 0:
            print(f"Backup cleanup failed with return code {result.returncode}")
            sys.exit(result.returncode)
        print("Backup cleanup executed successfully.")
    except Exception as e:
        print(f"Unexpected error during backup cleanup: {e}")
        sys.exit(1)