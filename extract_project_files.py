import os
import shutil
from pathlib import Path
from typing import List, Optional

def extract_files(source_files: List[str], dest_dir: str) -> None:
    """Extract and copy project files.
    
    Args:
        source_files: List of source file paths to copy
        dest_dir: Destination directory path
        
    Raises:
        ValueError: If source_files is empty, dest_dir is not a directory,
                   source file doesn't exist, or copy operation fails
    """
    if not source_files:
        raise ValueError("No source files provided")
        
    dest_path = Path(dest_dir)
    if dest_path.exists() and not dest_path.is_dir():
        raise ValueError(f"Destination {dest_dir} exists but is not a directory")
        
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Copy each file
    for src_file in source_files:
        src_path = Path(src_file)
        if not src_path.exists():
            raise ValueError(f"Source file {src_file} does not exist")
        if not src_path.is_file():
            raise ValueError(f"Source path {src_file} is not a file")
            
        try:
            dest_file = dest_path / src_path.name
            shutil.copy2(src_path, dest_file)
        except Exception as e:
            raise ValueError(f"Failed to copy {src_file} to {dest_dir}: {str(e)}")

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