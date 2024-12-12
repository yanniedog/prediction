# backup_cleanup.py
import shutil
import zipfile
import os
from pathlib import Path
from datetime import datetime

def backup_cleanup():
    try:
        cwd = Path.cwd()
        print(f"Current working directory: {cwd}")
        archive_dir = Path(r"C:\code\(archive) prediction\automated_zip_backups")
        print(f"Archive directory: {archive_dir}")
        archive_dir.mkdir(parents=True, exist_ok=True) if not archive_dir.exists() else None
        py_files = list(cwd.glob("*.py"))
        print(f"Found {len(py_files)} .py file(s) to backup.")
        bak_files = []
        for py in py_files:
            ts = datetime.fromtimestamp(py.stat().st_mtime).strftime("%Y-%m-%d__%H%M%S")
            bak = cwd / f"{py.stem}__{ts}.bak"
            if not bak.exists():
                shutil.copy2(py, bak)
            bak_files.append(bak)
        if not bak_files:
            print("No backup files to zip. Exiting backup_cleanup.")
            return
        zip_name = f"prediction_project_zip_backup_{datetime.now().strftime('%Y-%m-%d__%H%M%S')}.zip"
        zip_path = cwd / zip_name
        print(f"Creating zip: {zip_name}")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for bak in bak_files:
                zipf.write(bak, arcname=bak.name)
        print(f"Zip created: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            missing = [bak.name for bak in bak_files if bak.name not in zipf.namelist()]
            if missing:
                print(f"Missing in zip: {missing}")
                return
            print("All backups successfully zipped.")
        destination = archive_dir / zip_name
        shutil.move(str(zip_path), destination)
        print(f"Moved zip to archive: {destination}")
        if destination.exists():
            print(f"Zip stored in: {archive_dir}")
            for bak in bak_files:
                try:
                    bak.unlink()
                except Exception as e:
                    print(f"Delete error: {bak.name} - {e}")
            print("Backup cleanup completed successfully.")
        else:
            print("Error moving zip to archive.")
    except Exception as e:
        print(f"Backup cleanup error: {e}")

if __name__ == "__main__":
    backup_cleanup()
