#!/usr/bin/env python3

import os
import datetime
import argparse
import sys

# --- Configuration ---
BACKUP_SUBDIR = "project_backup"
FILENAME_PREFIX = "prediction_gemini_"
FILE_EXTENSIONS = ('.py', '.json')
SEPARATOR_LINE_LENGTH = 70 # Length of the separator line like "====="

def create_backup_flat(project_dir):
    """
    Finds .py and .json files ONLY in the top-level project_dir (no subdirs),
    concatenates them into a timestamped backup file in the BACKUP_SUBDIR.
    """
    project_dir = os.path.abspath(project_dir)
    backup_dir = os.path.join(project_dir, BACKUP_SUBDIR)

    print(f"Scanning ONLY the directory: {project_dir}")
    print(f"Ignoring all subdirectories.")
    print(f"Backup destination directory: {backup_dir}")

    # --- 1. Ensure backup directory exists ---
    try:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Backup directory '{BACKUP_SUBDIR}' ensured.")
    except OSError as e:
        print(f"Error: Could not create backup directory '{backup_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Generate backup filename ---
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{FILENAME_PREFIX}{timestamp}.txt"
    backup_filepath = os.path.join(backup_dir, backup_filename)

    print(f"Generated backup filename: {backup_filename}")

    # --- 3. Find relevant files (ONLY in project_dir) ---
    files_to_backup = []
    try:
        for filename in os.listdir(project_dir):
            # Check if it has the right extension
            if filename.endswith(FILE_EXTENSIONS):
                filepath = os.path.join(project_dir, filename)
                # Check if it's actually a file (and not a directory ending in .py/.json)
                if os.path.isfile(filepath):
                     # Ensure we don't accidentally pick up the backup file if run multiple times quickly
                     # or if script is located in the target directory
                    if filepath != backup_filepath:
                         # Check if the file is inside the backup directory itself
                        if os.path.dirname(filepath) != backup_dir:
                            files_to_backup.append(filepath)

    except FileNotFoundError:
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: Could not read project directory '{project_dir}': {e}", file=sys.stderr)
        sys.exit(1)


    if not files_to_backup:
        print(f"No '.py' or '.json' files found directly within '{project_dir}'.")
        # Decide if this is an error or just an empty backup case
        # sys.exit(0) # Exit successfully with empty backup
        # Let it proceed to create an empty (header-only) backup file
        pass # Keep going

    print(f"Found {len(files_to_backup)} files to include in the backup.")

    # --- 4. Concatenate files into backup ---
    try:
        with open(backup_filepath, 'w', encoding='utf-8') as outfile:
            print(f"Creating backup file: {backup_filepath}")
            outfile.write(f"# Backup created on: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write(f"# Source directory (non-recursive): {project_dir}\n")
            outfile.write(f"# Included files: {len(files_to_backup)}\n\n")

            # Sort files for consistent order (optional, but nice)
            files_to_backup.sort()

            for i, filepath in enumerate(files_to_backup):
                # Since we are only in the top level, the relative path is just the filename
                filename = os.path.basename(filepath)
                print(f"  Adding ({i+1}/{len(files_to_backup)}): {filename}")

                # Add a separator header
                outfile.write(f"\n{'=' * SEPARATOR_LINE_LENGTH}\n")
                outfile.write(f"=== START: {filename}\n")
                outfile.write(f"{'=' * SEPARATOR_LINE_LENGTH}\n\n")

                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except FileNotFoundError:
                     print(f"    Warning: File not found during read (should not happen?): {filepath}", file=sys.stderr)
                     outfile.write(f"*** ERROR: File not found at read time: {filename} ***\n")
                except UnicodeDecodeError:
                    print(f"    Warning: Could not decode file as UTF-8: {filepath}. Adding placeholder.", file=sys.stderr)
                    outfile.write(f"*** ERROR: Could not decode file content (not valid UTF-8): {filename} ***\n")
                except Exception as e:
                    print(f"    Error reading file {filepath}: {e}", file=sys.stderr)
                    outfile.write(f"*** ERROR reading file {filename}: {e} ***\n")

                # Add a separator footer
                outfile.write(f"\n\n{'=' * SEPARATOR_LINE_LENGTH}\n")
                outfile.write(f"=== END: {filename}\n")
                outfile.write(f"{'=' * SEPARATOR_LINE_LENGTH}\n")

        print(f"\nSuccessfully created backup: {backup_filepath}")

    except IOError as e:
        print(f"Error: Could not write to backup file '{backup_filepath}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during backup creation: {e}", file=sys.stderr)
        # Attempt to clean up partially created backup file on unexpected error
        if os.path.exists(backup_filepath):
             try:
                 os.remove(backup_filepath)
                 print(f"Cleaned up partial backup file: {backup_filepath}")
             except OSError as rm_err:
                 print(f"Warning: Could not remove partial backup file '{backup_filepath}': {rm_err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Concatenate .py and .json files FOUND ONLY in the specified directory (non-recursive) into a single backup file within the '{BACKUP_SUBDIR}' subdirectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "project_dir",
        nargs='?',
        default=os.getcwd(),
        help="The directory containing the .py and .json files to back up (subdirectories are ignored). Defaults to the current working directory."
    )

    args = parser.parse_args()

    if not os.path.isdir(args.project_dir):
        print(f"Error: Specified directory does not exist or is not a directory: {args.project_dir}", file=sys.stderr)
        sys.exit(1)

    create_backup_flat(args.project_dir)