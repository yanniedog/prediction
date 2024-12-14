# LAUNCH.py
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

def run_script(script_name: str) -> int:
    """
    Executes a Python script using runpy.run_path.

    Args:
        script_name (str): The name of the script to execute.

    Returns:
        int: Exit code from the script. 0 if successful, 1 otherwise.
    """
    try:
        logger.info(f"Executing {script_name}...")
        runpy.run_path(script_name, run_name="__main__")
        logger.info(f"{script_name} executed successfully.")
        return 0
    except SystemExit as e:
        logger.warning(f"SystemExit encountered in {script_name} with code: {e.code}")
        return e.code
    except Exception as e:
        logger.exception(f"Uncaught exception during execution of {script_name}")
        return 1

def main():
    """
    Main function to execute backup cleanup, run primary script, execute additional scripts,
    and handle error codes appropriately.
    """
    current_dir = Path.cwd()

    delete_old_logs(current_dir)

    configure_logging(log_file_prefix=current_dir.name)

    exit_code = 0

    primary_scripts = [
        ("run_backup_cleanup", lambda: run_backup_cleanup()),
        ("start.py", lambda: run_script("start.py"))
    ]

    for script_name, script_func in primary_scripts:
        try:
            if script_name == "run_backup_cleanup":
                logger.info(f"Executing {script_name}...")
                script_func()
                logger.info(f"{script_name} executed successfully.")
            else:
                script_exit_code = script_func()
                if script_exit_code != 0:
                    logger.warning(f"{script_name} exited with code: {script_exit_code}")
                    if exit_code == 0:
                        exit_code = script_exit_code
        except Exception as e:
            logger.exception(f"Exception occurred while executing {script_name}")
            if exit_code == 0:
                exit_code = 1

    additional_scripts = ["copyscripts.py", "COPYSCRIPTS_SELECTIVE.py"]
    for script in additional_scripts:
        script_exit_code = run_script(script)
        if script_exit_code != 0:
            logger.warning(f"{script} exited with code: {script_exit_code}")
            if exit_code == 0:
                exit_code = script_exit_code

    logger.info(f"launch.py exiting with code: {exit_code}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
