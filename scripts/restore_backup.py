# restore_backup.py
import os
from pathlib import Path

def rename_files(directory):
    dir_path=Path(directory)
    if not dir_path.is_dir():
        return
    for file_path in dir_path.glob("*.bak"):
        parts=file_path.stem.split('__')
        if len(parts)>=1:
            base_name=parts[0]
            new_name=f"{base_name}.py"
            new_path=dir_path/new_name
            if not new_path.exists():
                file_path.rename(new_path)

if __name__=="__main__":
    directory=r"C:\code\prediction"
    rename_files(directory)
