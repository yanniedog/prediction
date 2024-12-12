# backup_cleanup.py
import shutil, zipfile, os
from pathlib import Path
from datetime import datetime

def backup_cleanup():
    cwd = Path.cwd()
    archive_dir = Path(r"C:\code\(archive) prediction\automated_zip_backups")
    archive_dir.mkdir(parents=True, exist_ok=True)
    py_files = list(cwd.glob("*.py"))
    bak_files = []
    for py in py_files:
        bak = cwd / f"{py.stem}__{datetime.fromtimestamp(py.stat().st_mtime).strftime('%Y-%m-%d__%H%M%S')}.bak"
        if not bak.exists():
            shutil.copy2(py, bak)
        bak_files.append(bak)
    if bak_files:
        zip_name = f"prediction_project_zip_backup_{datetime.now().strftime('%Y-%m-%d__%H%M%S')}.zip"
        zip_path = cwd / zip_name
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for bak in bak_files:
                zipf.write(bak, arcname=bak.name)
        shutil.move(zip_path, archive_dir / zip_name)
        for bak in bak_files:
            bak.unlink()
        print("Backup completed.")

if __name__ == "__main__":
    backup_cleanup()
