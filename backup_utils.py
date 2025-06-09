#!/usr/bin/env python3

import os
import datetime
import argparse
import sys
import shutil
import json
import zipfile
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Union

# --- Configuration ---
BACKUP_SUBDIR = "project_backup"
FILENAME_PREFIX = "prediction_gemini_"
# Define allowed file extensions EXCLUDING generic .txt
# requirements.txt will be handled explicitly later.
FILE_EXTENSIONS = ('.py', '.json')
SEPARATOR_LINE_LENGTH = 70 # Length of the separator line like "====="

class BackupManager:
    """Manages database backups with rotation and validation."""
    
    def __init__(self, backup_dir: Union[str, Path], max_count: int = 5):
        """
        Initialize the backup manager.
        
        Args:
            backup_dir: Directory to store backups
            max_count: Maximum number of backups to keep
        """
        self.backup_dir = Path(backup_dir)
        self.max_count = max_count
        
        # Create backup directory if it doesn't exist
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Could not create backup directory: {e}")
            
        if not self.backup_dir.is_dir():
            raise ValueError(f"Backup path is not a directory: {backup_dir}")
            
        # Test if directory is writable
        test_file = self.backup_dir / ".test_write"
        try:
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            raise OSError(f"Backup directory is not writable: {e}")
    
    def create_backup(self, source_path: Union[str, Path], compression_level: int = 6, metadata: Optional[Dict] = None) -> Path:
        """
        Create a backup of a database file.
        
        Args:
            source_path: Path to the database file to backup
            compression_level: ZIP compression level (0-9, where 9 is maximum compression)
            metadata: Optional metadata to store with the backup
            
        Returns:
            Path to the created backup file
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise ValueError(f"Source file does not exist: {source_path}")
        if not source_path.is_file():
            raise ValueError(f"Source path is not a file: {source_path}")
        if source_path.stat().st_size == 0:
            raise ValueError(f"Source file is empty: {source_path}")
        if not 0 <= compression_level <= 9:
            raise ValueError("Compression level must be between 0 and 9")
        # Check if file is a valid SQLite database
        if not self._is_valid_sqlite_db(source_path):
            raise ValueError(f"Source file is not a valid SQLite database: {source_path}")
            
        # Generate backup filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.stem}_{timestamp}.zip"
        backup_path = self.backup_dir / backup_name
        
        # Create backup
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=compression_level) as zf:
                zf.write(source_path, source_path.name)
                
            # Create backup info file
            info = {
                "source": str(source_path),
                "timestamp": datetime.datetime.now().isoformat(),
                "size": backup_path.stat().st_size,
                "checksum": self._calculate_checksum(backup_path)
            }
            
            if metadata:
                info["metadata"] = metadata
            
            info_path = backup_path.with_suffix('.json')
            try:
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2)
            except OSError as e:
                if backup_path.exists():
                    backup_path.unlink()
                raise OSError(f"Failed to write backup info: {e}")
                
            # Cleanup old backups
            self._cleanup_old_backups()
            
            return backup_path
            
        except OSError as e:
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except OSError:
                    pass
            raise OSError(f"Failed to create backup: {e}")
        except Exception as e:
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except OSError:
                    pass
            raise ValueError(f"Failed to create backup: {e}")
    
    def restore_backup(self, backup_path: Union[str, Path], restore_path: Union[str, Path]) -> None:
        """
        Restore a database from backup.
        
        Args:
            backup_path: Path to the backup file
            restore_path: Path where to restore the database
        """
        backup_path = Path(backup_path)
        restore_path = Path(restore_path)
        
        if not backup_path.exists():
            raise ValueError(f"Backup file does not exist: {backup_path}")
        if not _validate_backup(backup_path):
            raise ValueError(f"Invalid backup file: {backup_path}")
        # Check if restore_path's parent exists and is writable
        parent_dir = restore_path.parent
        if not parent_dir.exists() or not os.access(parent_dir, os.W_OK):
            raise ValueError(f"Restore directory does not exist or is not writable: {parent_dir}")
        try:
            with zipfile.ZipFile(backup_path, 'r') as zf:
                # Extract the database file
                zf.extractall(restore_path.parent)
                # Rename if needed
                if restore_path.name != zf.namelist()[0]:
                    (restore_path.parent / zf.namelist()[0]).rename(restore_path)
        except Exception as e:
            raise ValueError(f"Failed to restore backup: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """Clean up old backups based on max_count."""
        backups = sorted(
            [b for b in self.backup_dir.glob("*.zip") if _validate_backup(b)],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Remove excess backups
        for backup in backups[self.max_count:]:
            try:
                backup.unlink()
                info_path = backup.with_suffix('.json')
                if info_path.exists():
                    info_path.unlink()
            except OSError as e:
                print(f"Warning: Could not remove old backup {backup}: {e}")
    
    @staticmethod
    def _calculate_checksum(file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def _is_valid_sqlite_db(path: Path) -> bool:
        try:
            with open(path, 'rb') as f:
                header = f.read(16)
            if header != b'SQLite format 3\x00':
                return False
            # Try connecting to the database
            conn = sqlite3.connect(str(path))
            conn.execute('PRAGMA schema_version;')
            conn.close()
            return True
        except Exception:
            return False

def _create_backup(source_path: Union[str, Path], backup_dir: Union[str, Path]) -> Path:
    """Create a backup of a file."""
    manager = BackupManager(backup_dir)
    return manager.create_backup(source_path)

def _restore_backup(backup_path: Union[str, Path], restore_path: Union[str, Path]) -> None:
    """Restore a file from backup."""
    manager = BackupManager(Path(backup_path).parent)
    manager.restore_backup(backup_path, restore_path)

def _validate_backup(backup_path: Union[str, Path]) -> bool:
    """Validate a backup file."""
    backup_path = Path(backup_path)
    if not backup_path.exists() or not backup_path.is_file():
        return False
        
    # Check if it's a valid zip file
    try:
        with zipfile.ZipFile(backup_path, 'r') as zf:
            if not zf.namelist():
                return False
    except zipfile.BadZipFile:
        return False
        
    # Check if info file exists and is valid
    info_path = backup_path.with_suffix('.json')
    if not info_path.exists():
        return False
        
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        required_keys = {"source", "timestamp", "size", "checksum"}
        if not all(key in info for key in required_keys):
            return False
            
        # Verify checksum
        if info["checksum"] != BackupManager._calculate_checksum(backup_path):
            return False
            
        return True
    except (json.JSONDecodeError, OSError):
        return False

def _get_backup_info(backup_path: Union[str, Path]) -> Dict:
    """Get information about a backup file."""
    backup_path = Path(backup_path)
    if not _validate_backup(backup_path):
        raise ValueError(f"Invalid backup file: {backup_path}")
        
    info_path = backup_path.with_suffix('.json')
    with open(info_path, 'r') as f:
        return json.load(f)

def _cleanup_old_backups(backup_dir: Union[str, Path], max_age_days: Optional[int] = None, max_count: Optional[int] = None) -> None:
    """Clean up old backups based on age and/or count."""
    backup_dir = Path(backup_dir)
    manager = BackupManager(backup_dir, max_count=max_count if max_count is not None else 5)
    
    if max_age_days is not None:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
        for backup in backup_dir.glob("*.zip"):
            if _validate_backup(backup):
                # Use file modification time for age
                backup_time = datetime.datetime.fromtimestamp(backup.stat().st_mtime)
                if backup_time < cutoff:
                    try:
                        backup.unlink()
                        info_path = backup.with_suffix('.json')
                        if info_path.exists():
                            info_path.unlink()
                    except OSError as e:
                        print(f"Warning: Could not remove old backup {backup}: {e}")
    
    if max_count is not None:
        manager.max_count = max_count
        manager._cleanup_old_backups()

def create_backup_flat(project_dir: str) -> bool:
    """
    Create a flat backup of all project files.
    
    Args:
        project_dir: Directory containing project files to backup
        
    Returns:
        bool: True if backup was successful, False otherwise
    """
    project_dir = Path(project_dir)
    if not project_dir.exists() or not project_dir.is_dir():
        raise ValueError(f"Invalid project directory: {project_dir}")
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"project_backup_{timestamp}.txt"
    backup_filepath = project_dir / backup_filename
    
    try:
        with open(backup_filepath, 'w', encoding='utf-8') as outfile:
            # Write header
            outfile.write(f"Project Backup - {timestamp}\n")
            outfile.write(f"{'=' * SEPARATOR_LINE_LENGTH}\n\n")
            
            # Process each file
            for filepath in sorted(project_dir.glob('**/*')):
                if filepath.is_file() and not filepath.name.startswith('.') and (filepath.suffix in FILE_EXTENSIONS or filepath.name == 'requirements.txt'):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                        outfile.write(f"\n{'=' * SEPARATOR_LINE_LENGTH}\n")
                        outfile.write(f"=== FILE: {filepath.relative_to(project_dir)}\n")
                        outfile.write(f"{'=' * SEPARATOR_LINE_LENGTH}\n\n")
                        outfile.write(content)
                        outfile.write(f"\n{'=' * SEPARATOR_LINE_LENGTH}\n")
                        outfile.write(f"=== END: {filepath.relative_to(project_dir)}\n")
                        outfile.write(f"{'=' * SEPARATOR_LINE_LENGTH}\n")
                    except Exception as e:
                        print(f"Warning: Could not process file {filepath}: {e}", file=sys.stderr)
                        continue
                        
        print(f"\nSuccessfully created backup: {backup_filepath}")
        return True
        
    except OSError as e:
        print(f"Error: Could not write to backup file '{backup_filepath}': {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"An unexpected error occurred during backup creation: {e}", file=sys.stderr)
        if backup_filepath.exists():
            try:
                backup_filepath.unlink()
                print(f"Cleaned up partial backup file: {backup_filepath}")
            except OSError as rm_err:
                print(f"Warning: Could not remove partial backup file '{backup_filepath}': {rm_err}", file=sys.stderr)
        raise

# This allows the script to be run directly for backup purposes
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Concatenate {', '.join(FILE_EXTENSIONS)} files and the specific file 'requirements.txt' FOUND ONLY in the specified directory (non-recursive) into a single backup file within the '{BACKUP_SUBDIR}' subdirectory. Excludes other .txt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "project_dir",
        nargs='?',
        default=os.getcwd(), # Default to current working directory
        help="The directory containing the files to back up (subdirectories are ignored). Defaults to the current working directory."
    )

    args = parser.parse_args()
    project_dir_arg = args.project_dir

    # Validate input directory using pathlib
    project_path_arg = Path(project_dir_arg).resolve()
    if not project_path_arg.is_dir():
        print(f"Error: Specified path is not a valid directory: {project_path_arg}", file=sys.stderr)
        sys.exit(1)

    try:
        create_backup_flat(str(project_path_arg)) # Pass string representation back if needed
        sys.exit(0) # Exit successfully
    except Exception as main_err:
        print(f"\nBackup failed: {main_err}", file=sys.stderr)
        sys.exit(1) # Exit with error