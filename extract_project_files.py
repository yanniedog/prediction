# extract_project_files.py
import re
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILENAME = "google.txt"
BACKUP_FILE_PREFIX = "prediction_backup_google_"
FILES_TO_BACKUP_PATTERNS = ['*.py', '*.json', 'requirements.txt']

def create_backup(backup_dir: Path):
    """Finds specified files and creates a single concatenated backup file."""
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    backup_filename = backup_dir / f"{BACKUP_FILE_PREFIX}{timestamp}.bak"
    files_backed_up_count = 0

    logging.info(f"Creating backup file: {backup_filename}")

    try:
        with backup_filename.open('w', encoding='utf-8') as bak_file:
            for pattern in FILES_TO_BACKUP_PATTERNS:
                matched_files = list(Path.cwd().glob(pattern))
                if pattern == 'requirements.txt' and not matched_files:
                     req_path = Path.cwd() / 'requirements.txt'
                     if req_path.exists(): matched_files = [req_path]

                for file_path in matched_files:
                    if file_path.name == backup_filename.name or file_path.name == Path(__file__).name:
                        continue

                    if file_path.is_file():
                        try:
                            logging.debug(f"Backing up {file_path.name}...")
                            content = file_path.read_text(encoding='utf-8', errors='replace')
                            bak_file.write(f"--- START OF BACKUP FILE {file_path.relative_to(Path.cwd())} ---\n")
                            bak_file.write(content)
                            bak_file.write(f"\n--- END OF BACKUP FILE {file_path.relative_to(Path.cwd())} ---\n\n")
                            files_backed_up_count += 1
                        except Exception as e:
                            logging.error(f"Error reading file {file_path} for backup: {e}")

        if files_backed_up_count > 0:
            logging.info(f"Successfully backed up {files_backed_up_count} files to {backup_filename}")
        else:
            logging.warning("No existing project files found to back up.")

    except Exception as e:
        logging.error(f"Failed to create backup file {backup_filename}: {e}")
        return False
    return True

def clean_content(raw_content: str) -> str:
    """Removes leading/trailing Markdown code fences if present."""
    lines = raw_content.splitlines()
    if not lines:
        return ""

    first_line_stripped = lines[0].strip()
    last_line_stripped = lines[-1].strip()

    has_leading_fence = first_line_stripped.startswith('```')
    has_trailing_fence = last_line_stripped == '```'

    if has_leading_fence and has_trailing_fence and len(lines) >= 2:
        return '\n'.join(lines[1:-1])
    elif has_leading_fence and len(lines) == 1:
        return ""
    else:
        return raw_content


def extract_files_from_source(source_filepath: Path):
    """Extracts individual files from the concatenated source file, cleaning content."""
    logging.info(f"Reading source file: {source_filepath}")
    if not source_filepath.exists():
        logging.error(f"Source file '{source_filepath}' not found.")
        sys.exit(1)

    try:
        content = source_filepath.read_text(encoding='utf-8')
    except Exception as e:
        logging.error(f"Failed to read source file '{source_filepath}': {e}")
        sys.exit(1)

    pattern = re.compile(
        r"^--- START OF FILE (.*?) ---\n(.*?)\n^--- END OF FILE \1 ---",
        re.DOTALL | re.MULTILINE
    )

    matches = pattern.finditer(content)
    extracted_count = 0
    error_count = 0

    logging.info("Starting file extraction...")
    for match in matches:
        filename_str = ""
        try:
            filename_str = match.group(1).strip()
            raw_file_content = match.group(2)

            if not filename_str:
                logging.warning("Found a match with an empty filename. Skipping.")
                continue

            cleaned_file_content = clean_content(raw_file_content)
            if raw_file_content != cleaned_file_content:
                 logging.debug(f"Removed Markdown fences from {filename_str}")

            target_path = Path(filename_str).resolve()

            target_path.parent.mkdir(parents=True, exist_ok=True)

            target_path.write_text(cleaned_file_content, encoding='utf-8', newline='\n')

            logging.info(f"Extracted and saved: {target_path}")
            extracted_count += 1

        except Exception as e:
            logging.error(f"Error processing or writing file '{filename_str or 'Unknown'}': {e}")
            error_count += 1

    logging.info(f"Extraction complete. Successfully extracted: {extracted_count} files.")
    if error_count > 0:
        logging.warning(f"Encountered {error_count} errors during extraction.")

def main():
    """Main execution function."""
    current_dir = Path.cwd()
    source_file = current_dir / INPUT_FILENAME

    if not create_backup(current_dir):
        logging.error("Backup creation failed. Aborting extraction.")
        sys.exit(1)

    extract_files_from_source(source_file)

if __name__ == "__main__":
    main()