# filename: backup_utils.py
import os
import sys
import logging
import subprocess
from pathlib import Path
def run_backup_cleanup()->None:
    try:
        backup_script=Path(__file__).resolve().parent/'backup_cleanup.py'
        if not backup_script.exists():
            logging.error(f"Backup script '{backup_script}' not found.")
            print(f"Backup script '{backup_script}' not found.")
            sys.exit(1)
        logging.info(f"Executing backup script: {backup_script}")
        subprocess.run([sys.executable,str(backup_script)],check=True)
        logging.info("Backup cleanup executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Backup cleanup failed with error: {e}")
        print(f"Backup cleanup failed with error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during backup cleanup: {e}")
        print(f"Unexpected error during backup cleanup: {e}")
        sys.exit(1)