import pytest
import tempfile
import os
import shutil
from pathlib import Path
from extract_project_files import (
    extract_files,
    _filter_files,
    _copy_files,
    _validate_paths
)

def create_sample_files(tmp_path):
    files = []
    for i in range(3):
        file_path = tmp_path / f"file{i}.txt"
        file_path.write_text(f"content {i}")
        files.append(file_path)
    return files

def test_extract_files(tmp_path):
    # Create sample files and a destination directory
    files = create_sample_files(tmp_path)
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    # Extract files
    extract_files([str(f) for f in files], str(dest_dir))
    for f in files:
        assert (dest_dir / f.name).exists()
        assert (dest_dir / f.name).read_text() == f.read_text()

def test_filter_files(tmp_path):
    files = create_sample_files(tmp_path)
    # Filter by extension
    filtered = _filter_files([str(f) for f in files], ".txt")
    assert set(filtered) == set(str(f) for f in files)
    # Filter by non-existent extension
    filtered = _filter_files([str(f) for f in files], ".csv")
    assert filtered == []

def test_copy_files(tmp_path):
    files = create_sample_files(tmp_path)
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    _copy_files([str(f) for f in files], str(dest_dir))
    for f in files:
        assert (dest_dir / f.name).exists()
        assert (dest_dir / f.name).read_text() == f.read_text()

def test_validate_paths(tmp_path):
    files = create_sample_files(tmp_path)
    # Valid paths
    _validate_paths([str(f) for f in files], str(tmp_path))
    # Invalid file
    with pytest.raises(ValueError):
        _validate_paths([str(tmp_path / "nonexistent.txt")], str(tmp_path))
    # Invalid destination
    with pytest.raises(ValueError):
        _validate_paths([str(f) for f in files], str(tmp_path / "nonexistent_dir"))

def test_error_handling(tmp_path):
    files = create_sample_files(tmp_path)
    dest_dir = tmp_path / "dest"
    # Destination is a file, not a directory
    file_dest = tmp_path / "file_dest.txt"
    file_dest.write_text("not a dir")
    with pytest.raises(ValueError):
        extract_files([str(f) for f in files], str(file_dest))
    # Source file does not exist
    with pytest.raises(ValueError):
        extract_files([str(tmp_path / "does_not_exist.txt")], str(dest_dir))
    # No files provided
    with pytest.raises(ValueError):
        extract_files([], str(dest_dir))

def test_extract_files_copies_files(tmp_path):
    # Create dummy .py and .json files
    py_file = tmp_path / "test.py"
    json_file = tmp_path / "test.json"
    py_file.write_text("print('hello')")
    json_file.write_text("{}")
    dest_dir = tmp_path / "dest"
    extract_files([str(py_file), str(json_file)], str(dest_dir))
    assert (dest_dir / "test.py").exists()
    assert (dest_dir / "test.json").exists() 