import pytest
import tempfile
import os
from pathlib import Path
import backup_utils

def test_create_backup_flat_creates_file(tmp_path):
    # Create dummy .py and .json files
    py_file = tmp_path / "test.py"
    json_file = tmp_path / "test.json"
    req_file = tmp_path / "requirements.txt"
    py_file.write_text("print('hello')")
    json_file.write_text("{}")
    req_file.write_text("pytest\n")
    # Run backup
    result = backup_utils.create_backup_flat(str(tmp_path))
    backup_dir = tmp_path / backup_utils.BACKUP_SUBDIR
    backups = list(backup_dir.glob("*.txt"))
    assert result is True
    assert backup_dir.exists()
    assert any(b.name.startswith(backup_utils.FILENAME_PREFIX) for b in backups)
    # Check backup file content
    with backups[0].open("r", encoding="utf-8") as f:
        content = f.read()
        assert "test.py" in content
        assert "test.json" in content
        assert "requirements.txt" in content

def test_create_backup_flat_handles_empty(tmp_path):
    # No files in directory
    result = backup_utils.create_backup_flat(str(tmp_path))
    backup_dir = tmp_path / backup_utils.BACKUP_SUBDIR
    backups = list(backup_dir.glob("*.txt"))
    assert result is True
    assert backup_dir.exists()
    assert len(backups) == 1
    with backups[0].open("r", encoding="utf-8") as f:
        content = f.read()
        assert "Included files (0):" in content 