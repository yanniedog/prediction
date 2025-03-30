# extract_project_files.py
import re
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Import the standard backup utility
try:
    import backup_utils
    BACKUP_UTIL_AVAILABLE = True
except ImportError:
    BACKUP_UTIL_AVAILABLE = False
    logging.error("Could not import backup_utils. Backup step will be skipped in main().")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_FILENAME = "google.txt" # The file containing the concatenated code from Gemini/Google

# --- Regex ---
# Updated regex to be more robust with different line endings and START/END markers
PATTERN = re.compile(
    # Match the start block, capturing filename (group 1)
    r"^(?:={70}\n)?(?:={3}\sSTART:\s)(.*?)(?:\s={3}\n)(?:={70}\n\n?)"
    # Capture the content (group 2, non-greedy)
    r"(.*?)"
    # Match the end block, using backreference \1 for filename
    r"(?:\n\n?^(?:={70}\n)?(?:={3}\sEND:\s)\1(?:\s={3}\n)(?:={70}\n)?)",
    re.DOTALL | re.MULTILINE
)


def clean_content(raw_content: str) -> str:
    """
    Cleans the extracted file content: removes markdown fences, normalizes line endings.
    """
    lines = raw_content.splitlines()
    if not lines: return "\n" # Empty file with newline

    # Remove Markdown Fences (``` or ```python etc.)
    start_index = 0
    end_index = len(lines)
    if lines[0].strip().startswith('```'): start_index = 1
    if lines[-1].strip() == '```': end_index = -1

    cleaned_lines = lines[start_index:end_index] if start_index <= end_index else []

    # Normalize Line Endings and Ensure Single Trailing Newline
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = cleaned_text.replace('\r\n', '\n').replace('\r', '\n')
    cleaned_text = cleaned_text.rstrip('\n') + '\n'
    return cleaned_text


def extract_files_from_source(source_filepath: Path):
    """Extracts individual files from the concatenated source file."""
    logging.info(f"Reading source file: {source_filepath}")
    if not source_filepath.exists():
        logging.error(f"Source file '{source_filepath}' not found.")
        sys.exit(1)

    try:
        content = source_filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        logging.error(f"Failed to read source file '{source_filepath}': {e}")
        sys.exit(1)

    matches = PATTERN.finditer(content)
    extracted_count = 0; error_count = 0; found_match = False; current_pos = 0

    logging.info("Starting file extraction...")
    for match in matches:
        found_match = True
        # Check for content between matches
        if match.start() > current_pos:
            missed_content = content[current_pos:match.start()].strip()
            is_header = missed_content.startswith(("# Backup created on:", "# Source directory", "# Included files"))
            if missed_content and not is_header:
                logging.warning(f"Found content between file blocks (pos {current_pos}-{match.start()}):\n---\n{missed_content[:200]}...\n---")

        filename_str = ""
        try:
            filename_str = match.group(1).strip()
            raw_file_content = match.group(2)

            if not filename_str: logging.warning("Found match with empty filename. Skipping."); continue

            cleaned_file_content = clean_content(raw_file_content)
            if not cleaned_file_content.strip(): logging.warning(f"Skipping '{filename_str}' (empty after cleaning)."); continue

            target_path = Path(filename_str).resolve()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(cleaned_file_content, encoding='utf-8', newline='\n')

            logging.info(f"Extracted and saved: {target_path}")
            extracted_count += 1

        except Exception as e:
            logging.error(f"Error processing/writing file '{filename_str or 'Unknown'}': {e}", exc_info=True)
            error_count += 1

        current_pos = match.end()

    # Check for trailing content
    if current_pos < len(content):
         trailing_content = content[current_pos:].strip()
         if trailing_content:
              logging.warning(f"Found trailing content after last block:\n---\n{trailing_content[:200]}...\n---")

    if not found_match:
         logging.warning(f"No file blocks matching pattern found in '{INPUT_FILENAME}'. Check format/regex.")

    logging.info(f"Extraction complete. Extracted: {extracted_count}. Errors: {error_count}.")
    if error_count > 0: sys.exit(1) # Exit with error if issues occurred

def main():
    """Main execution function."""
    current_dir = Path.cwd()
    source_file = current_dir / INPUT_FILENAME

    # --- Create Backup using standard utility ---
    if BACKUP_UTIL_AVAILABLE:
        print("\n--- Creating Backup ---")
        try:
            backup_utils.create_backup_flat(str(current_dir))
            print("Backup potentially created (check output).")
        except Exception as bk_err:
            logging.error(f"Error running standard backup utility: {bk_err}. Aborting extraction.")
            print(f"ERROR: Backup failed: {bk_err}")
            sys.exit(1)
    else:
        print("WARNING: backup_utils not found. Skipping backup.")

    print("\n--- Extracting Files ---")
    extract_files_from_source(source_file)

if __name__ == "__main__":
    main()
