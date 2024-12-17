# filename: restore_backup.py
import os
from pathlib import Path

def rename_files(directory):
    dir_path=Path(directory)
    if not dir_path.is_dir():
        print(f"Error: The directory '{directory}' does not exist.")
        return
    for file_path in dir_path.glob("*.bak"):
        parts=file_path.stem.split('__')
        if len(parts)>=1:
            base_name=parts[0]
            new_name=f"{base_name}.py"
            new_path=dir_path/new_name
            try:
                file_path.rename(new_path)
                print(f'Renamed: "{file_path.name}" --> "{new_name}"')
            except FileExistsError:
                print(f'Error: The file "{new_name}" already exists. Skipping "{file_path.name}".')
            except Exception as e:
                print(f'Error renaming "{file_path.name}": {e}')
        else:
            print(f'Warning: The file "{file_path.name}" does not match the expected pattern. Skipping.')

if __name__=="__main__":
    directory=r"C:\code\prediction"
    rename_files(directory)