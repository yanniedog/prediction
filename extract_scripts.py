# extract_scripts.py
import re
import os
import sys
from pathlib import Path
import logging
import backup_utils # Import the backup utility

# --- Configuration ---
INPUT_FILENAME = "google.txt"
BACKUP_DIR_NAME = "project_backup"
# Regex to find script blocks: Starts with '# filename.ext', captures filename and content
# It assumes content ends before the next '# filename.ext' or EOF. Includes newline handling.
SCRIPT_BLOCK_REGEX = re.compile(
    r"^#\s*([\w.-]+\.(?:py|json))\s*$(.*?)(?=^#\s*[\w.-]+\.(?:py|json)\s*$|\Z)",
    re.MULTILINE | re.DOTALL
)

# --- Logging Setup ---
# Basic logging for the extractor script itself
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Script Extraction ---")

    # 1. Create Backup
    logger.info(f"Attempting to back up existing .py and .json files to '{BACKUP_DIR_NAME}'...")
    if not backup_utils.create_backup(backup_dir_name=BACKUP_DIR_NAME):
        logger.error("Backup failed. Aborting script extraction to prevent data loss.")
        sys.exit(1)
    logger.info("Backup completed successfully.")

    # 2. Read Input File
    input_path = Path(INPUT_FILENAME)
    if not input_path.exists():
        logger.error(f"Input file '{INPUT_FILENAME}' not found. Cannot extract scripts.")
        sys.exit(1)

    try:
        logger.info(f"Reading content from '{INPUT_FILENAME}'...")
        content = input_path.read_text(encoding='utf-8')
        logger.info(f"Read {len(content)} characters.")
    except Exception as e:
        logger.error(f"Error reading input file '{INPUT_FILENAME}': {e}", exc_info=True)
        sys.exit(1)

    # 3. Find and Extract Script Blocks
    matches = SCRIPT_BLOCK_REGEX.finditer(content)
    extracted_count = 0
    error_count = 0

    logger.info("Starting extraction process...")
    for match in matches:
        filename = match.group(1).strip()
        script_content = match.group(2).strip() # Remove leading/trailing whitespace from content block

        if not filename or not script_content:
            logger.warning("Found a block with missing filename or content. Skipping.")
            continue

        output_path = Path(filename)
        try:
            output_path.write_text(script_content, encoding='utf-8')
            logger.info(f"Successfully extracted and saved: '{filename}'")
            extracted_count += 1
        except IOError as e:
            logger.error(f"Error writing file '{filename}': {e}", exc_info=True)
            error_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing file '{filename}': {e}", exc_info=True)
            error_count += 1

    # 4. Report Results
    logger.info("--- Script Extraction Finished ---")
    logger.info(f"Successfully extracted: {extracted_count} files.")
    if error_count > 0:
        logger.error(f"Encountered errors during extraction for {error_count} files.")
        sys.exit(1) # Exit with error status if any extraction failed
    elif extracted_count == 0:
        logger.warning("No script blocks were found or extracted. Please check the input file and regex.")
    else:
        logger.info("Extraction completed without errors.")

    sys.exit(0) # Exit with success status