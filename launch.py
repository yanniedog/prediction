import sys
import logging
import runpy
from pathlib import Path
from logging_setup import configure_logging
from backup_utils import run_backup_cleanup

logger = logging.getLogger()

def delete_old_logs(log_dir: Path, log_extension: str = ".log"):
    for log_file in log_dir.glob(f"*{log_extension}"):
        try:
            log_file.unlink()
        except Exception as e:
            logger.error(f"Error deleting log file {log_file}: {e}")

def main():
    current_dir = Path.cwd()

    delete_old_logs(current_dir)
    configure_logging(log_file_prefix=current_dir.name)

    try:
        run_backup_cleanup()
        runpy.run_path("start.py", run_name="__main__")
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        logger.exception("Uncaught exception during execution")
        raise

if __name__ == "__main__":
    main()