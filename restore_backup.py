import os
from pathlib import Path

def rename_files(directory):
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: '{directory}' does not exist.")
        return
    for bak in dir_path.glob("*.bak"):
        parts = bak.stem.split('__')
        if parts:
            new_name = f"{parts[0]}.py"
            new_path = dir_path / new_name
            try:
                bak.rename(new_path)
                print(f'Renamed: "{bak.name}" --> "{new_name}"')
            except FileExistsError:
                print(f'Error: "{new_name}" exists. Skipping "{bak.name}".')
            except Exception as e:
                print(f'Error renaming "{bak.name}": {e}')
        else:
            print(f'Warning: "{bak.name}" pattern mismatch. Skipping.')

if __name__=="__main__":
    rename_files(r"C:\code\prediction")
