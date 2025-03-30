# extract_scripts.py
# Note: This script performs the same function as extract_project_files.py but uses a slightly different regex name.
# Keeping both if they serve different subtle purposes or legacy reasons, otherwise could consolidate.
import re
import os
import sys
from pathlib import Path
import logging

# Import the standard backup utility
try:
    import backup_utils
    BACKUP_UTIL_AVAILABLE = True
except ImportError:
    BACKUP_UTIL_AVAILABLE = False
    logging.error("Could not import backup_utils. Backup step will be skipped.")


# --- Configuration ---
INPUT_FILENAME = "google.txt"

# --- Regex ---
# Regex to find script blocks (same pattern as extract_project_files.py)
SCRIPT_BLOCK_REGEX = re.compile(
    r"^(?:={70}\n)?(?:={3}\sSTART:\s)(.*?)(?:\s={3}\n)(?:={70}\n\n?)"
    r"(.*?)"
    r"(?:\n\n?^(?:={70}\n)?(?:={3}\sEND:\s)\1(?:\s={3}\n)(?:={70}\n)?)",
    re.MULTILINE | re.DOTALL
)

# --- Logging Setup ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

def clean_content(raw_content: str) -> str:
    """Cleans extracted content (identical to extract_project_files)."""
    lines = raw_content.splitlines()
    if not lines: return "\n"
    start_index = 1 if lines[0].strip().startswith('```') else 0
    end_index = -1 if lines[-1].strip() == '```' else len(lines)
    cleaned_lines = lines[start_index:end_index] if start_index <= end_index else []
    cleaned_text = '\n'.join(cleaned_lines).replace('\r\n', '\n').replace('\r', '\n')
    return cleaned_text.rstrip('\n') + '\n'


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Script Extraction (using extract_scripts.py) ---")
    current_dir = Path.cwd()

    # 1. Create Backup
    if BACKUP_UTIL_AVAILABLE:
        logger.info(f"Backing up existing files in '{current_dir}' using backup_utils...")
        try:
            backup_utils.create_backup_flat(str(current_dir))
            logger.info("Backup completed successfully via backup_utils.")
        except Exception as bk_err:
            logger.error(f"Backup failed via backup_utils: {bk_err}. Aborting.", exc_info=True)
            sys.exit(1)
    else:
        logger.warning("backup_utils not found. Skipping backup.")

    # 2. Read Input File
    input_path = current_dir / INPUT_FILENAME
    if not input_path.exists():
        logger.error(f"Input file '{INPUT_FILENAME}' not found in {current_dir}.")
        sys.exit(1)
    try:
        logger.info(f"Reading content from '{INPUT_FILENAME}'...")
        content = input_path.read_text(encoding='utf-8', errors='replace')
        logger.info(f"Read {len(content)} characters.")
    except Exception as e:
        logger.error(f"Error reading '{INPUT_FILENAME}': {e}", exc_info=True)
        sys.exit(1)

    # 3. Find and Extract Script Blocks
    matches = SCRIPT_BLOCK_REGEX.finditer(content)
    extracted_count = 0; error_count = 0; found_match = False; current_pos = 0

    logger.info("Starting extraction process...")
    for match in matches:
        found_match = True
        # Check for content between matches
        if match.start() > current_pos:
            missed_content = content[current_pos:match.start()].strip()
            is_header = missed_content.startswith(("# Backup created on:", "# Source directory", "# Included files"))
            if missed_content and not is_header:
                logger.warning(f"Found content between blocks (pos {current_pos}-{match.start()}):\n---\n{missed_content[:200]}...\n---")

        filename = ""
        try:
            filename = match.group(1).strip()
            raw_content = match.group(2)

            if not filename: logger.warning("Found block with missing filename. Skipping."); continue

            cleaned_script_content = clean_content(raw_content)
            if not cleaned_script_content.strip(): logger.warning(f"Skipping '{filename}' (empty after cleaning)."); continue

            output_path = Path(filename).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(cleaned_script_content, encoding='utf-8', newline='\n')

            logger.info(f"Successfully extracted and saved: '{output_path}'")
            extracted_count += 1
        except IOError as e:
            logger.error(f"IOError writing file '{output_path}': {e}", exc_info=True)
            error_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing file '{filename or 'Unknown'}': {e}", exc_info=True)
            error_count += 1

        current_pos = match.end()

    # Check for trailing content
    if current_pos < len(content):
         trailing_content = content[current_pos:].strip()
         if trailing_content:
              logger.warning(f"Found trailing content after last block:\n---\n{trailing_content[:200]}...\n---")

    if not found_match:
         logger.warning(f"No script blocks matching pattern found in '{INPUT_FILENAME}'. Check format/regex.")

    # 4. Report Results
    logger.info("--- Script Extraction Finished ---")
    logger.info(f"Successfully extracted: {extracted_count} files.")
    if error_count > 0:
        logger.error(f"Encountered errors during extraction for {error_count} files.")
        sys.exit(1)
    elif extracted_count == 0:
         logger.warning("No scripts were successfully extracted.")
    else:
        logger.info("Extraction completed without errors.")

    sys.exit(0) # Success
