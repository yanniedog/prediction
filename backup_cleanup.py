# backup_cleanup.py
import shutil, zipfile, os
from pathlib import Path
from datetime import datetime
import logging

def backup_cleanup():
    logger = logging.getLogger(__name__)
    cwd = Path.cwd()
    archive_dir = Path(r"C:\code\(archive) prediction\automated_zip_backups")
    archive_dir.mkdir(parents=True, exist_ok=True)
    py_files = list(cwd.glob("*.py"))
    bak_files = []
    for py in py_files:
        bak = cwd / f"{py.stem}__{datetime.fromtimestamp(py.stat().st_mtime).strftime('%Y-%m-%d__%H%M%S')}.bak"
        if not bak.exists():
            shutil.copy2(py, bak)
            logger.debug(f"Copied {py} to {bak}")
        bak_files.append(bak)
    if bak_files:
        zip_name = f"prediction_project_zip_backup_{datetime.now().strftime('%Y-%m-%d__%H%M%S')}.zip"
        zip_path = cwd / zip_name
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for bak in bak_files:
                zipf.write(bak, arcname=bak.name)
                logger.debug(f"Added {bak} to {zip_path}")
        shutil.move(zip_path, archive_dir / zip_name)
        logger.info("Backup completed.")
        for bak in bak_files:
            bak.unlink()
            logger.debug(f"Deleted backup file {bak}")
    else:
        logger.info("No backup files to process.")

if __name__ == "__main__":
    backup_cleanup()
