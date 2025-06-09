import os
import shutil
from pathlib import Path
from typing import List, Optional

def extract_files(files: List[str], dest_dir: str) -> None:
    """
    Copy files to dest_dir.
    
    Args:
        files: List of file paths to copy
        dest_dir: Destination directory path
        
    Raises:
        ValueError: If dest_dir is a file or if any other error occurs
    """
    dest = Path(dest_dir)
    
    # Check if destination exists and is a file
    if dest.exists() and not dest.is_dir():
        raise ValueError(f"Destination path exists and is not a directory: {dest_dir}")
        
    # Create destination directory if it doesn't exist
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        # This should not happen due to the check above, but handle it anyway
        raise ValueError(f"Cannot create directory: {dest_dir} exists and is not a directory")
    except Exception as e:
        raise ValueError(f"Error creating destination directory: {str(e)}")
        
    # Copy each file
    for file_path in files:
        try:
            src = Path(file_path)
            if not src.exists():
                raise ValueError(f"Source file does not exist: {file_path}")
            if not src.is_file():
                raise ValueError(f"Source path is not a file: {file_path}")
                
            dst = dest / src.name
            shutil.copy2(src, dst)
        except Exception as e:
            raise ValueError(f"Error copying file {file_path}: {str(e)}")

def _filter_files(files: List[str], ext: str) -> List[str]:
    """Filter files by extension."""
    return [f for f in files if f.endswith(ext)]

def _copy_files(files: List[str], dest_dir: str) -> None:
    """Copy files to destination directory."""
    dest = Path(dest_dir)
    for f in files:
        shutil.copy2(f, dest / Path(f).name)

def _validate_paths(files: List[str], dest_dir: str) -> None:
    """Validate that all files exist and dest_dir is a directory."""
    dest = Path(dest_dir)
    if not dest.exists() or not dest.is_dir():
        raise ValueError("Destination directory does not exist or is not a directory")
    for f in files:
        if not Path(f).exists():
            raise ValueError(f"Source file does not exist: {f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract project files to a destination directory.")
    parser.add_argument("source", help="Source directory")
    parser.add_argument("dest", help="Destination directory")
    parser.add_argument("--patterns", nargs="*", default=None, help="Glob patterns to match files (default: all files)")
    args = parser.parse_args()
    source = Path(args.source)
    files = [str(f) for f in source.glob("*") if f.is_file()]
    extract_files(files, args.dest)
    print(f"Files extracted from {args.source} to {args.dest}.") 