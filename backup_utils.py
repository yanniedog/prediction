# backup_utils.py
import sys
import logging
from backup_cleanup import backup_cleanup

def run_backup_cleanup():
    logger = logging.getLogger(__name__)
    try:
        backup_cleanup()
        logger.info("Backup executed successfully.")
    except Exception as e:
        logger.error(f"Backup cleanup failed: {e}")
        sys.exit(1)
