# backup_cleanup.py
import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def backup_cleanup():
    """
    Locates all .py files in the current directory, creates backup copies with a .bak extension
    (appending the last modified datetime in YYYY-MM-DD__HHMMSS format if not already present),
    zips all backed-up files, moves the zip to the specified archive directory, and removes the
    .bak files after successful backup.
    """
    try:
        # Define the current working directory
        cwd = Path.cwd()
        print(f"Current working directory: {cwd}")

        # Define the target archive directory
        archive_dir = Path(r"C:\code\(archive) prediction\automated_zip_backups")
        print(f"Archive directory: {archive_dir}")

        # Create the archive directory if it doesn't exist
        if not archive_dir.exists():
            print(f"Archive directory does not exist. Creating: {archive_dir}")
            archive_dir.mkdir(parents=True, exist_ok=True)

        # Locate all .py files in the current directory
        py_files = list(cwd.glob("*.py"))
        print(f"Found {len(py_files)} .py file(s) to backup.")

        bak_files = []

        for py_file in py_files:
            # Get the last modified datetime of the .py file
            last_modified_timestamp = py_file.stat().st_mtime
            last_modified_dt = datetime.fromtimestamp(last_modified_timestamp)
            formatted_dt = last_modified_dt.strftime("%Y-%m-%d__%H%M%S")
            # print(f"Last modified datetime for {py_file.name}: {formatted_dt}")

            # Construct the backup filename
            backup_filename = f"{py_file.stem}__{formatted_dt}.bak"
            backup_file = cwd / backup_filename

            # Check if the backup file already exists
            if backup_file.exists():
                print(f"Backup already exists for {py_file.name}: {backup_filename}")
            else:
                # Create the backup copy
                shutil.copy2(py_file, backup_file)
                #print(f"Created backup: {backup_filename}")

            bak_files.append(backup_file)

        if not bak_files:
            print("No backup files to zip. Exiting backup_cleanup.")
            return

        # Prepare the zip filename with current datetime
        zip_creation_dt = datetime.now().strftime("%Y-%m-%d__%H%M%S")
        zip_filename = f"prediction_project_zip_backup_{zip_creation_dt}.zip"
        zip_path = cwd / zip_filename
        print(f"Creating zip file: {zip_filename}")

        # Create the zip archive and add all .bak files
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for bak_file in bak_files:
                zipf.write(bak_file, arcname=bak_file.name)
                #print(f"Added to zip: {bak_file.name}")

        print(f"Zip file created: {zip_path}")

        # Verify that all .bak files are in the zip
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zip_contents = zipf.namelist()
            missing_files = [bak.name for bak in bak_files if bak.name not in zip_contents]
            if missing_files:
                print(f"Error: The following backup files are missing in the zip: {missing_files}")
                return
            else:
                print("All backup files successfully zipped.")

        # Move the zip file to the archive directory
        destination = archive_dir / zip_filename
        shutil.move(str(zip_path), destination)
        print(f"Moved zip to archive directory: {destination}")

        # Confirm the zip file exists in the archive directory
        if not destination.exists():
            print(f"Error: Zip file was not moved to the archive directory.")
            return
        else:
            print(f"Zip file successfully stored in: {archive_dir}")

        # Remove the .bak files from the working directory
        for bak_file in bak_files:
            try:
                bak_file.unlink()
                # print(f"Deleted backup file: {bak_file.name}")
            except Exception as e:
                print(f"Error deleting {bak_file.name}: {e}")

        print("Backup cleanup completed successfully.")

    except Exception as e:
        print(f"An error occurred during backup cleanup: {e}")

if __name__ == "__main__":
    backup_cleanup()