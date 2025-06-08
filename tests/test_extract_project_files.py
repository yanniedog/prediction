import pytest
import tempfile
import os
from pathlib import Path
import extract_project_files

def test_extract_project_files_runs(tmp_path, monkeypatch):
    # Create dummy .py and .json files
    py_file = tmp_path / "test.py"
    json_file = tmp_path / "test.json"
    py_file.write_text("print('hello')")
    json_file.write_text("{}")
    # Patch sys.argv to simulate CLI call
    monkeypatch.setattr('sys.argv', ['extract_project_files.py', str(tmp_path)])
    # Should not raise
    try:
        extract_project_files.main()
    except SystemExit:
        pass 