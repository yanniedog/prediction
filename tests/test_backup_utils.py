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
from backup_utils import (
    BackupManager,
    _create_backup,
    _restore_backup,
    _validate_backup,
    _get_backup_info,
    _cleanup_old_backups
)

@pytest.fixture(scope="function")
def temp_dir() -> Path:
    """Create a temporary directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def sample_data(temp_dir: Path) -> pd.DataFrame:
    """Create sample data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
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
    
    # Test with invalid path
    with pytest.raises(ValueError):
        BackupManager("/invalid/path")

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
        backups.append(backup_path)
    
    # Test cleanup with max_age
    _cleanup_old_backups(backup_manager.backup_dir, max_age_days=2)
    remaining_backups = list(backup_manager.backup_dir.glob("*.zip"))
    assert len(remaining_backups) == 3  # Only backups less than 2 days old
    
    # Test cleanup with max_count
    _cleanup_old_backups(backup_manager.backup_dir, max_count=2)
    remaining_backups = list(backup_manager.backup_dir.glob("*.zip"))
    assert len(remaining_backups) == 2  # Only 2 most recent backups

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

def test_error_handling(backup_manager: BackupManager, sample_db: Path):
    """Test error handling."""
    # Test with read-only directory
    read_only_dir = backup_manager.backup_dir / "readonly"
    read_only_dir.mkdir()
    read_only_dir.chmod(0o444)  # Read-only
    with pytest.raises(PermissionError):
        backup_manager.create_backup(sample_db, backup_dir=read_only_dir)
    
    # Test with corrupted database
    corrupted_db = backup_manager.backup_dir / "corrupted.db"
    corrupted_db.write_bytes(b"invalid data")
    with pytest.raises(ValueError):
        backup_manager.create_backup(corrupted_db)
    
    # Test with insufficient disk space
    with pytest.raises(OSError):
        backup_manager.create_backup(sample_db, backup_dir=Path("/invalid/path"))

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
    
    # Test restoration with metadata
    restore_path = backup_manager.backup_dir / "restored.db"
    backup_manager.restore_backup(backup_path, restore_path)
    restored_info = _get_backup_info(restore_path.with_suffix(".zip"))
    assert restored_info["metadata"] == metadata

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