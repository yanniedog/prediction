# backup_utils.py
import subprocess, sys
from pathlib import Path
import logging

def run_backup_cleanup():
    logger = logging.getLogger(__name__)
    backup_script = Path(__file__).resolve().parent / 'backup_cleanup.py'
    if not backup_script.exists():
        logger.error(f"Backup script not found: {backup_script}")
        sys.exit(1)
    result = subprocess.run([sys.executable, str(backup_script)], text=True, capture_output=True)
    if result.returncode != 0:
        logger.error(f"Backup script failed with return code {result.returncode}")
        logger.error(f"stderr: {result.stderr}")
        sys.exit(result.returncode)
    logger.info("Backup executed successfully.")
    logger.debug(f"stdout: {result.stdout}")
