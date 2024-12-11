import os
from pathlib import Path

def rename_files(directory):
    """
    Renames .bak files in the specified directory by removing the date/time suffix
    and changing the extension to .py.

    Args:
        directory (str): The path to the directory containing the files.
    """
    dir_path = Path(directory)

    # Check if the directory exists
    if not dir_path.is_dir():
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Iterate over all .bak files in the directory
    for file_path in dir_path.glob("*.bak"):
        # Example filename: "logging_setup__2024-11-14__180610.bak"
        # We want to rename it to "logging_setup.py"

        # Split the filename at each double underscore
        parts = file_path.stem.split('__')

        if len(parts) >= 1:
            # The base name is the first part before the first '__'
            base_name = parts[0]
            new_name = f"{base_name}.py"
            new_path = dir_path / new_name

            try:
                # Rename the file
                file_path.rename(new_path)
                print(f'Renamed: "{file_path.name}" --> "{new_name}"')
            except FileExistsError:
                print(f'Error: The file "{new_name}" already exists. Skipping renaming of "{file_path.name}".')
            except Exception as e:
                print(f'Error renaming "{file_path.name}": {e}')
        else:
            print(f'Warning: The file "{file_path.name}" does not match the expected pattern. Skipping.')

if __name__ == "__main__":
    # Specify the directory containing the files
    directory = r"C:\code\prediction"

    # Call the rename function
    rename_files(directory)