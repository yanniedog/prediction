# filename: backup_utils.py
import os
import sys
import subprocess
from pathlib import Path

def run_backup_cleanup()->None:
    try:
        backup_script=Path(__file__).resolve().parent/'backup_cleanup.py'
        if not backup_script.exists():
            print(f"Backup script '{backup_script}' not found.")
            sys.exit(1)
        print(f"Executing backup script: {backup_script}")
        subprocess.run([sys.executable,str(backup_script)],check=True)
        print("Backup cleanup executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Backup cleanup failed with error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during backup cleanup: {e}")
        sys.exit(1)