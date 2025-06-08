import pytest
import tempfile
import os
from pathlib import Path
import extract_project_files

def test_extract_files_copies_files(tmp_path):
    # Create dummy .py and .json files
    py_file = tmp_path / "test.py"
    json_file = tmp_path / "test.json"
    py_file.write_text("print('hello')")
    json_file.write_text("{}")
    dest_dir = tmp_path / "dest"
    extract_project_files.extract_files(str(tmp_path), str(dest_dir), patterns=["*.py", "*.json"])
    assert (dest_dir / "test.py").exists()
    assert (dest_dir / "test.json").exists() 