# launch.py
import os
import sys
import logging
import runpy
from pathlib import Path
from datetime import datetime
from logging_setup import configure_logging
from sqlite_data_manager import initialize_database
from config import DB_PATH
from backup_utils import run_backup_cleanup

def delete_old_logs(log_dir: Path, log_extension: str = ".log"):
    for log_file in log_dir.glob(f"*{log_extension}"):
        try:
            log_file.unlink()
        except Exception as e:
            print(f"Error deleting log file {log_file}: {e}")

def main():
    current_dir = Path.cwd()
    delete_old_logs(current_dir)

    log_file = f"{current_dir.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    configure_logging(log_file)
    logger = logging.getLogger()

    try:
        run_backup_cleanup()
        logger.info("Backup executed successfully.")
        logger.info("Backup completed.")
        logger.info("Backup cleanup completed.")
        initialize_database(DB_PATH)
        logger.info("Database initialized.")

        runpy.run_path("start.py", run_name="__main__")
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        logger.exception("Uncaught exception during execution")
        raise

if __name__ == "__main__":
    main()
