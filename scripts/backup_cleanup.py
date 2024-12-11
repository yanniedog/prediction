# backup_cleanup.py
import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def backup_cleanup():
    try:
        cwd = Path.cwd()
        archive_dir = Path(r"C:\code\(archive) prediction\automated_zip_backups")
        if not archive_dir.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
        py_files = list(cwd.glob("*.py"))
        bak_files = []
        for py_file in py_files:
            last_modified_dt = datetime.fromtimestamp(py_file.stat().st_mtime)
            formatted_dt = last_modified_dt.strftime("%Y-%m-%d__%H%M%S")
            backup_file = cwd / f"{py_file.stem}__{formatted_dt}.bak"
            if not backup_file.exists():
                shutil.copy2(py_file, backup_file)
            bak_files.append(backup_file)
        if not bak_files:
            return
        zip_creation_dt = datetime.now().strftime("%Y-%m-%d__%H%M%S")
        zip_filename = f"prediction_project_zip_backup_{zip_creation_dt}.zip"
        zip_path = cwd / zip_filename
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for bak_file in bak_files:
                zipf.write(bak_file, arcname=bak_file.name)
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zip_contents = zipf.namelist()
            missing_files = [bak.name for bak in bak_files if bak.name not in zip_contents]
            if missing_files:
                return
        destination = archive_dir / zip_filename
        shutil.move(str(zip_path), destination)
        for bak_file in bak_files:
            try:
                bak_file.unlink()
            except:
                pass
    except:
        pass

if __name__ == "__main__":
    backup_cleanup()