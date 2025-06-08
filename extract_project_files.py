import os
import shutil
from pathlib import Path
from typing import List, Optional

def extract_files(files: List[str], dest_dir: str) -> None:
    """
    Copy files to dest_dir.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    _validate_paths(files, dest_dir)
    _copy_files(files, dest_dir)

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