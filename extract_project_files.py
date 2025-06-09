import os
import shutil
from pathlib import Path
from typing import List, Optional

def extract_files(source_files: List[str], dest_dir: str) -> None:
    """Extract and copy project files to destination directory.
    
    Args:
        source_files: List of source file paths to extract
        dest_dir: Destination directory path
        
    Raises:
        ValueError: If source files list is empty, destination is not a directory,
                   or any source file doesn't exist
    """
    if not source_files:
        raise ValueError("No source files provided")
        
    # Validate destination directory
    if not os.path.isdir(dest_dir):
        raise ValueError(f"Destination path is not a directory: {dest_dir}")
        
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Validate and copy each source file
    for src_path in source_files:
        if not os.path.exists(src_path):
            raise ValueError(f"Source file does not exist: {src_path}")
            
        if not os.path.isfile(src_path):
            raise ValueError(f"Source path is not a file: {src_path}")
            
        # Get destination path
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))
        
        try:
            shutil.copy2(src_path, dest_path)
        except (shutil.Error, OSError) as e:
            raise ValueError(f"Error copying file {src_path}: {e}")

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