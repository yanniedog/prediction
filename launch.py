# launch.py
import sys
import logging
import runpy
from pathlib import Path
from logging_setup import configure_logging
from backup_utils import run_backup_cleanup

logger = logging.getLogger()

def delete_old_logs(log_dir: Path, log_extension: str = ".log"):
    """
    Deletes old log files in the specified directory with the given extension.

    Args:
        log_dir (Path): Directory containing log files.
        log_extension (str): Extension of log files to delete.
    """
    for log_file in log_dir.glob(f"*{log_extension}"):
        try:
            log_file.unlink()
            logger.info(f"Deleted log file: {log_file}")
        except Exception as e:
            logger.error(f"Error deleting log file {log_file}: {e}")

def main():
    """
    Main function to execute backup cleanup, run primary script, and execute additional scripts.
    """
    current_dir = Path.cwd()

    # Delete old log files
    delete_old_logs(current_dir)

    # Configure logging with a prefix based on the current directory name
    configure_logging(log_file_prefix=current_dir.name)

    try:
        # Run backup cleanup
        run_backup_cleanup()
        logger.info("Backup cleanup completed successfully.")

        # Execute the primary script: start.py
        logger.info("Executing start.py...")
        runpy.run_path("start.py", run_name="__main__")
        logger.info("start.py executed successfully.")

        # Execute the first additional script: copyscript.py
        logger.info("Executing copyscript.py...")
        runpy.run_path("copyscript.py", run_name="__main__")
        logger.info("copyscript.py executed successfully.")

        # Execute the second additional script: COPYSCRIPTS_SELECTIVE.py
        logger.info("Executing COPYSCRIPTS_SELECTIVE.py...")
        runpy.run_path("COPYSCRIPTS_SELECTIVE.py", run_name="__main__")
        logger.info("COPYSCRIPTS_SELECTIVE.py executed successfully.")

    except SystemExit as e:
        logger.warning(f"SystemExit encountered with code: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        logger.exception("Uncaught exception during execution")
        raise

if __name__ == "__main__":
    main()
