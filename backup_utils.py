#!/usr/bin/env python3

import os
import datetime
import argparse
import sys
from pathlib import Path # Use pathlib

# --- Configuration ---
BACKUP_SUBDIR = "project_backup"
FILENAME_PREFIX = "prediction_gemini_"
# Define allowed file extensions EXCLUDING generic .txt
# requirements.txt will be handled explicitly later.
FILE_EXTENSIONS = ('.py', '.json')
SEPARATOR_LINE_LENGTH = 70 # Length of the separator line like "====="

def create_backup_flat(project_dir: str):
    """
    Finds .py, .json files AND the specific file 'requirements.txt' ONLY
    in the top-level project_dir (no subdirs), concatenates them into a
    timestamped backup file in the BACKUP_SUBDIR. Excludes other .txt files.
    """
    project_path = Path(project_dir).resolve() # Use Path object and resolve
    backup_dir_path = project_path / BACKUP_SUBDIR

    print(f"Scanning ONLY the directory: {project_path}")
    print(f"Including file extensions: {FILE_EXTENSIONS} and the specific file 'requirements.txt'.")
    print(f"Excluding all other '.txt' files.")
    print(f"Ignoring all subdirectories.")
    print(f"Backup destination directory: {backup_dir_path}")

    # --- 1. Ensure backup directory exists ---
    try:
        backup_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Backup directory '{BACKUP_SUBDIR}' ensured.")
    except OSError as e:
        print(f"Error: Could not create backup directory '{backup_dir_path}': {e}", file=sys.stderr)
        # Don't exit here, allow script using this utility to decide
        raise # Re-raise the exception

    # --- 2. Generate backup filename ---
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{FILENAME_PREFIX}{timestamp}.txt" # Backup file itself is .txt
    backup_filepath = backup_dir_path / backup_filename

    print(f"Generated backup filename: {backup_filename}")

    # --- 3. Find relevant files (ONLY in project_dir) ---
    files_to_backup = []
    try:
        for item in project_path.iterdir(): # Use iterdir()
            # Check if it's a file AND
            # (has an extension in FILE_EXTENSIONS OR is exactly 'requirements.txt')
            if item.is_file() and (item.suffix in FILE_EXTENSIONS or item.name == 'requirements.txt'):
                 # Ensure we don't accidentally pick up the backup file itself (unlikely here but good practice)
                if item.resolve() != backup_filepath.resolve():
                     # Check if the file is inside the backup directory itself
                    if item.parent.resolve() != backup_dir_path.resolve():
                        # Check if it's this script itself
                        if item.name != Path(__file__).name:
                            files_to_backup.append(item) # Store Path objects

    except FileNotFoundError:
        print(f"Error: Project directory not found: {project_path}", file=sys.stderr)
        raise # Re-raise
    except OSError as e:
        print(f"Error: Could not read project directory '{project_path}': {e}", file=sys.stderr)
        raise # Re-raise

    if not files_to_backup:
        print(f"No files matching {FILE_EXTENSIONS} or 'requirements.txt' found directly within '{project_path}'.")
        # Let it proceed to create an empty (header-only) backup file
        pass

    print(f"Found {len(files_to_backup)} files to include in the backup:")
    for f in sorted(files_to_backup, key=lambda p: p.name): # Print sorted list for clarity
        print(f"  - {f.name}")

    # --- 4. Concatenate files into backup ---
    try:
        # Use pathlib open method
        with backup_filepath.open('w', encoding='utf-8') as outfile:
            print(f"\nCreating backup file: {backup_filepath}")
            outfile.write(f"# Backup created on: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write(f"# Source directory (non-recursive): {project_path}\n")
            outfile.write(f"# Included files ({len(files_to_backup)}): {', '.join(sorted(p.name for p in files_to_backup))}\n\n")

            # Sort files by name for consistent order in the output file
            files_to_backup.sort(key=lambda p: p.name)

            content = "" # Initialize content to handle edge case of empty file
            for i, filepath_obj in enumerate(files_to_backup):
                filename = filepath_obj.name # Get filename from Path object
                print(f"  Adding ({i+1}/{len(files_to_backup)}): {filename}")

                # Add a separator header
                outfile.write(f"\n{'=' * SEPARATOR_LINE_LENGTH}\n")
                outfile.write(f"=== START: {filename}\n")
                outfile.write(f"{'=' * SEPARATOR_LINE_LENGTH}\n\n")

                try:
                    # Use pathlib read_text method
                    content = filepath_obj.read_text(encoding='utf-8', errors='replace') # Use replace for safety
                    outfile.write(content)
                except FileNotFoundError:
                     # This check is less likely needed with pathlib unless file deleted between find and read
                     print(f"    Warning: File not found during read: {filepath_obj}", file=sys.stderr)
                     outfile.write(f"*** ERROR: File not found at read time: {filename} ***\n")
                     content = "" # Reset content on error
                except UnicodeDecodeError:
                    print(f"    Warning: Could not decode file as UTF-8: {filepath_obj}. Adding placeholder.", file=sys.stderr)
                    outfile.write(f"*** ERROR: Could not decode file content (not valid UTF-8): {filename} ***\n")
                    content = "" # Reset content on error
                except Exception as e:
                    print(f"    Error reading file {filepath_obj}: {e}", file=sys.stderr)
                    outfile.write(f"*** ERROR reading file {filename}: {e} ***\n")
                    content = "" # Reset content on error

                # Add a separator footer (ensure newline before it if content didn't have one)
                # Check if content was successfully read and doesn't end with newline
                if content and not content.endswith('\n'):
                    outfile.write('\n')
                outfile.write(f"\n{'=' * SEPARATOR_LINE_LENGTH}\n")
                outfile.write(f"=== END: {filename}\n")
                outfile.write(f"{'=' * SEPARATOR_LINE_LENGTH}\n")

        print(f"\nSuccessfully created backup: {backup_filepath}")
        return True # Indicate success

    except IOError as e:
        print(f"Error: Could not write to backup file '{backup_filepath}': {e}", file=sys.stderr)
        raise # Re-raise
    except Exception as e:
        print(f"An unexpected error occurred during backup creation: {e}", file=sys.stderr)
        # Attempt to clean up partially created backup file on unexpected error
        if backup_filepath.exists():
             try:
                 backup_filepath.unlink()
                 print(f"Cleaned up partial backup file: {backup_filepath}")
             except OSError as rm_err:
                 print(f"Warning: Could not remove partial backup file '{backup_filepath}': {rm_err}", file=sys.stderr)
        raise # Re-raise

# This allows the script to be run directly for backup purposes
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Concatenate {', '.join(FILE_EXTENSIONS)} files and the specific file 'requirements.txt' FOUND ONLY in the specified directory (non-recursive) into a single backup file within the '{BACKUP_SUBDIR}' subdirectory. Excludes other .txt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "project_dir",
        nargs='?',
        default=os.getcwd(), # Default to current working directory
        help="The directory containing the files to back up (subdirectories are ignored). Defaults to the current working directory."
    )

    args = parser.parse_args()
    project_dir_arg = args.project_dir

    # Validate input directory using pathlib
    project_path_arg = Path(project_dir_arg).resolve()
    if not project_path_arg.is_dir():
        print(f"Error: Specified path is not a valid directory: {project_path_arg}", file=sys.stderr)
        sys.exit(1)

    try:
        create_backup_flat(str(project_path_arg)) # Pass string representation back if needed
        sys.exit(0) # Exit successfully
    except Exception as main_err:
        print(f"\nBackup failed: {main_err}", file=sys.stderr)
        sys.exit(1) # Exit with error