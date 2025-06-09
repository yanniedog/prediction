#!/usr/bin/env python3
"""
Script to run pytest with comprehensive logging to test.log
"""
import subprocess
import sys
import os
from pathlib import Path

def run_tests_with_comprehensive_logging():
    """Run pytest with comprehensive logging to test.log"""
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Clear previous test.log if it exists
    test_log_path = Path("test.log")
    if test_log_path.exists():
        test_log_path.unlink()
    
    print(f"Starting pytest with comprehensive logging...")
    print(f"All output will be captured in: {test_log_path.absolute()}")
    print(f"Console will show summary, detailed logs in test.log")
    print("=" * 80)
    
    # Build pytest command with comprehensive logging options
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--tb=long",
        "--showlocals", 
        "--maxfail=15",
        "-v",
        "-W", "error::UserWarning",
        "--log-file=test.log",
        "--log-file-level=DEBUG",
        "--log-cli=true",
        "--log-cli-level=INFO",
        "--capture=no",  # Don't capture output in pytest's own capture system
        "-s",  # Show print statements
    ]
    
    try:
        # Run pytest
        result = subprocess.run(
            pytest_cmd,
            capture_output=False,  # Let pytest handle its own output
            text=True,
            cwd=project_root
        )
        
        print("=" * 80)
        print(f"Pytest completed with exit code: {result.returncode}")
        
        if test_log_path.exists():
            print(f"Comprehensive test log saved to: {test_log_path.absolute()}")
            print(f"Log file size: {test_log_path.stat().st_size:,} bytes")
            
            # Show a summary of what's in the log
            with open(test_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"Log file contains {len(lines):,} lines")
                
                # Show last few lines as preview
                if lines:
                    print("\nLast 10 lines of test.log:")
                    for line in lines[-10:]:
                        print(f"  {line.rstrip()}")
        else:
            print("Warning: test.log file was not created")
            
        return result.returncode
        
    except Exception as e:
        print(f"Error running pytest: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests_with_comprehensive_logging()
    sys.exit(exit_code) 