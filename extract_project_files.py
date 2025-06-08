import os
import shutil
from pathlib import Path
from typing import List, Optional

def extract_files(source_dir: str, dest_dir: str, patterns: Optional[List[str]] = None) -> None:
    """
    Copy files matching given patterns from source_dir to dest_dir.
    If patterns is None, copy all files.
    """
    source = Path(source_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if not patterns:
        patterns = ['*']
    for pattern in patterns:
        for file in source.glob(pattern):
            if file.is_file():
                shutil.copy2(file, dest / file.name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract project files to a destination directory.")
    parser.add_argument("source", help="Source directory")
    parser.add_argument("dest", help="Destination directory")
    parser.add_argument("--patterns", nargs="*", default=None, help="Glob patterns to match files (default: all files)")
    args = parser.parse_args()
    extract_files(args.source, args.dest, args.patterns)
    print(f"Files extracted from {args.source} to {args.dest}.") 