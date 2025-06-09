import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
import sqlite3
import time
from datetime import datetime, timedelta
import os
import sys
from backup_utils import (
    BackupManager,
    _create_backup,
    _restore_backup,
    _validate_backup,
    _get_backup_info,
    _cleanup_old_backups,
    create_backup_flat
)

@pytest.fixture(scope="function")
def temp_dir() -> Path:
    """Create a temporary directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    try:
        # First try to make all files writable
        for root, dirs, files in os.walk(temp_dir):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), 0o777)
                except OSError:
                    pass
            for f in files:
                try:
                    os.chmod(os.path.join(root, f), 0o666)
                except OSError:
                    pass
        # Then try to remove the directory
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")

@pytest.fixture(scope="function")
def sample_data(temp_dir: Path) -> pd.DataFrame:
    """Create sample data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    data = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.normal(100, 1, 100),
        "high": np.random.normal(101, 1, 100),
        "low": np.random.normal(99, 1, 100),
        "close": np.random.normal(100, 1, 100),
        "volume": np.random.normal(1000, 100, 100)
    })
    data["high"] = data[["open", "close"]].max(axis=1) + abs(np.random.normal(0, 0.1, 100))
    data["low"] = data[["open", "close"]].min(axis=1) - abs(np.random.normal(0, 0.1, 100))
    return data

@pytest.fixture(scope="function")
def sample_db(temp_dir: Path, sample_data: pd.DataFrame) -> Path:
    """Create a sample database for testing."""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(db_path)
    sample_data.to_sql("prices", conn, index=False)
    conn.close()
    return db_path

@pytest.fixture(scope="function")
def backup_manager(temp_dir: Path) -> BackupManager:
    """Create a BackupManager instance for testing."""
    return BackupManager(temp_dir / "backups")

def test_backup_manager_initialization(temp_dir: Path):
    """Test backup manager initialization."""
    # Test basic initialization
    backup_dir = temp_dir / "backups"
    manager = BackupManager(backup_dir)
    assert manager.backup_dir == backup_dir
    assert backup_dir.exists()
    
    # Test with existing directory
    manager = BackupManager(backup_dir)
    assert manager.backup_dir == backup_dir
    
    # Test with invalid path (use a file instead of a directory)
    invalid_file = temp_dir / "not_a_dir.txt"
    invalid_file.touch()
    with pytest.raises(OSError):
        BackupManager(invalid_file)

def test_create_backup(backup_manager: BackupManager, sample_db: Path):
    """Test backup creation."""
    # Test basic backup creation
    backup_path = backup_manager.create_backup(sample_db)
    assert backup_path.exists()
    assert backup_path.suffix == ".zip"
    
    # Verify backup contents
    backup_info = _get_backup_info(backup_path)
    assert backup_info["source"] == str(sample_db)
    assert backup_info["timestamp"] is not None
    assert backup_info["size"] > 0
    
    # Test with invalid source
    with pytest.raises(ValueError):
        backup_manager.create_backup(Path("/invalid/path"))
    
    # Test with empty source
    empty_db = backup_manager.backup_dir / "empty.db"
    empty_db.touch()
    with pytest.raises(ValueError):
        backup_manager.create_backup(empty_db)

def test_restore_backup(backup_manager: BackupManager, sample_db: Path):
    """Test backup restoration."""
    # Create a backup first
    backup_path = backup_manager.create_backup(sample_db)
    
    # Test basic restoration
    restore_path = backup_manager.backup_dir / "restored.db"
    backup_manager.restore_backup(backup_path, restore_path)
    assert restore_path.exists()
    
    # Verify restored database
    conn = sqlite3.connect(restore_path)
    restored_data = pd.read_sql("SELECT * FROM prices", conn)
    conn.close()
    
    original_conn = sqlite3.connect(sample_db)
    original_data = pd.read_sql("SELECT * FROM prices", original_conn)
    original_conn.close()
    
    pd.testing.assert_frame_equal(restored_data, original_data)
    
    # Test with invalid backup
    with pytest.raises(ValueError):
        backup_manager.restore_backup(Path("/invalid/backup.zip"), restore_path)
    
    # Test with invalid restore path
    with pytest.raises(ValueError):
        backup_manager.restore_backup(backup_path, Path("/invalid/restore.db"))

def test_validate_backup(backup_manager: BackupManager, sample_db: Path):
    """Test backup validation."""
    # Create a backup first
    backup_path = backup_manager.create_backup(sample_db)
    
    # Test valid backup
    assert _validate_backup(backup_path)
    
    # Test invalid backup
    invalid_backup = backup_manager.backup_dir / "invalid.zip"
    invalid_backup.touch()
    assert not _validate_backup(invalid_backup)
    
    # Test non-existent backup
    assert not _validate_backup(Path("/invalid/backup.zip"))

def test_backup_info(backup_manager: BackupManager, sample_db: Path):
    """Test backup information retrieval."""
    # Create a backup first
    backup_path = backup_manager.create_backup(sample_db)
    
    # Test info retrieval
    info = _get_backup_info(backup_path)
    assert isinstance(info, dict)
    assert "source" in info
    assert "timestamp" in info
    assert "size" in info
    assert "checksum" in info
    
    # Test with invalid backup
    with pytest.raises(ValueError):
        _get_backup_info(Path("/invalid/backup.zip"))

def test_cleanup_old_backups(backup_manager: BackupManager, sample_db: Path):
    """Test cleanup of old backups."""
    # Create multiple backups with different timestamps
    backups = []
    for i in range(5):
        # Create backup with modified timestamp
        backup_path = backup_manager.create_backup(sample_db)
        timestamp = datetime.now() - timedelta(days=i)
        backup_info = {
            "source": str(sample_db),
            "timestamp": timestamp.isoformat(),
            "size": backup_path.stat().st_size,
            "checksum": "test"
        }
        with open(backup_path.with_suffix(".json"), "w") as f:
            json.dump(backup_info, f)
        # Set the file modification time to match the fake timestamp
        mod_time = timestamp.timestamp()
        os.utime(backup_path, (mod_time, mod_time))
        os.utime(backup_path.with_suffix(".json"), (mod_time, mod_time))
        backups.append(backup_path)
    
    # Test cleanup with max_age
    _cleanup_old_backups(backup_manager.backup_dir, max_age_days=2)
    remaining_backups = list(backup_manager.backup_dir.glob("*.zip"))
    assert len(remaining_backups) == 1  # Only backups less than 2 days old

def test_backup_rotation(backup_manager: BackupManager, sample_db: Path):
    """Test backup rotation."""
    # Create initial backup
    backup_path = backup_manager.create_backup(sample_db)
    
    # Test rotation with max_count
    for _ in range(5):
        backup_manager.create_backup(sample_db)
    
    backups = list(backup_manager.backup_dir.glob("*.zip"))
    assert len(backups) <= 5  # Default max_count is 5
    
    # Test rotation with custom max_count
    backup_manager.max_count = 3
    backup_manager.create_backup(sample_db)
    backups = list(backup_manager.backup_dir.glob("*.zip"))
    assert len(backups) <= 3

def test_backup_compression(backup_manager: BackupManager, sample_db: Path):
    """Test backup compression."""
    # Create a backup
    backup_path = backup_manager.create_backup(sample_db)
    
    # Verify compression
    original_size = sample_db.stat().st_size
    backup_size = backup_path.stat().st_size
    assert backup_size < original_size  # Backup should be smaller than original
    
    # Test with different compression levels
    backup_path_high = backup_manager.create_backup(sample_db, compression_level=9)
    backup_path_low = backup_manager.create_backup(sample_db, compression_level=1)
    
    assert backup_path_high.stat().st_size <= backup_path_low.stat().st_size

def test_error_handling(temp_dir: Path, sample_db: Path):
    """Test error handling for various error conditions."""
    # Test with read-only directory
    read_only_dir = temp_dir / "readonly"
    read_only_dir.mkdir()
    
    # Set directory to read-only on Windows
    if os.name == 'nt':
        import pytest
        pytest.skip("Skipping read-only directory test on Windows; not enforced by OS.")
    else:
        read_only_dir.chmod(0o444)
    
    try:
        # Create a backup manager with read-only directory
        with pytest.raises(OSError, match="Backup directory is not writable"):
            BackupManager(read_only_dir)
    finally:
        if os.name != 'nt':
            read_only_dir.chmod(0o777)
    
    # Test with invalid source file
    invalid_file = temp_dir / "nonexistent.db"
    manager = BackupManager(temp_dir / "backups")
    with pytest.raises(ValueError, match="Source file does not exist"):
        manager.create_backup(invalid_file)
    
    # Test with empty file
    empty_file = temp_dir / "empty.db"
    empty_file.touch()
    with pytest.raises(ValueError, match="Source file is empty"):
        manager.create_backup(empty_file)
    
    # Test with invalid compression level
    with pytest.raises(ValueError, match="Compression level must be between 0 and 9"):
        manager.create_backup(sample_db, compression_level=10)
    
    # Test with invalid restore path
    backup_path = manager.create_backup(sample_db)
    invalid_restore = Path("/invalid/path/restore.db")
    with pytest.raises(ValueError, match="Restore directory does not exist or is not writable"):
        manager.restore_backup(backup_path, invalid_restore)
    
    # Test with invalid backup file
    invalid_backup = temp_dir / "invalid.zip"
    invalid_backup.touch()
    with pytest.raises(ValueError, match="Invalid backup file"):
        manager.restore_backup(invalid_backup, temp_dir / "restored.db")
    
    # Test with corrupted backup file
    corrupted_backup = temp_dir / "corrupted.zip"
    with open(corrupted_backup, 'w') as f:
        f.write("not a zip file")
    with pytest.raises(ValueError, match="Failed to restore backup"):
        manager.restore_backup(corrupted_backup, temp_dir / "restored.db")

def test_backup_metadata(backup_manager: BackupManager, sample_db: Path):
    """Test backup metadata handling."""
    # Create a backup with metadata
    metadata = {
        "description": "Test backup",
        "version": "1.0",
        "tags": ["test", "backup"]
    }
    backup_path = backup_manager.create_backup(sample_db, metadata=metadata)
    
    # Verify metadata
    backup_info = _get_backup_info(backup_path)
    assert "metadata" in backup_info
    assert backup_info["metadata"] == metadata

def test_create_backup_flat_creates_file(tmp_path: Path):
    """Test that create_backup_flat creates a backup file."""
    # Create some test files
    test_files = {
        "test.py": "print('test')",
        "test.json": '{"test": true}',
        "requirements.txt": "pytest==7.0.0"
    }
    
    for filename, content in test_files.items():
        file_path = tmp_path / filename
        file_path.write_text(content)
    
    # Create backup
    result = create_backup_flat(str(tmp_path))
    assert result is True
    
    # Verify backup file exists
    backup_files = list(tmp_path.glob("project_backup_*.txt"))
    assert len(backup_files) == 1
    
    # Verify backup contents
    backup_content = backup_files[0].read_text()
    for filename in test_files:
        assert f"=== FILE: {filename}" in backup_content
        assert test_files[filename] in backup_content 

@pytest.fixture(scope="function", autouse=True)
def cleanup_temp_files():
    """Ensure temporary files are cleaned up after each test."""
    yield
    # Force garbage collection to release file handles
    import gc
    gc.collect()
    
    # Additional cleanup for Windows
    if os.name == 'nt':
        import time
        time.sleep(0.1)  # Give Windows a moment to release file handles 