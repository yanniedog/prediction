#!/usr/bin/env python3
"""
Script to test pytest logging setup and verify it's working correctly.
"""
import subprocess
import sys
import os
from pathlib import Path

def test_pytest_logging():
    """Test that pytest logging is working correctly."""
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("Testing pytest logging setup...")
    print("=" * 60)
    
    # Test 1: Run the logging verification tests
    print("Test 1: Running logging verification tests...")
    result1 = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_logging_verification.py",
        "-v",
        "--log-cli=true",
        "--log-cli-level=INFO",
        "--log-file=test_logging.log",
        "--log-file-level=DEBUG"
    ], capture_output=True, text=True)
    
    print(f"Exit code: {result1.returncode}")
    print(f"Stdout: {result1.stdout[:500]}...")
    if result1.stderr:
        print(f"Stderr: {result1.stderr[:500]}...")
    
    # Check if log file was created
    log_file = Path("test_logging.log")
    if log_file.exists():
        print(f"✓ Log file created: {log_file}")
        print(f"  Size: {log_file.stat().st_size} bytes")
        
        # Show some log content
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"  Lines: {len(lines)}")
            if lines:
                print("  Last 5 lines:")
                for line in lines[-5:]:
                    print(f"    {line.rstrip()}")
    else:
        print("✗ Log file was not created")
    
    print("\n" + "=" * 60)
    
    # Test 2: Run with the comprehensive logging script
    print("Test 2: Running with comprehensive logging script...")
    result2 = subprocess.run([
        sys.executable, "run_tests_with_logging.py"
    ], capture_output=True, text=True)
    
    print(f"Exit code: {result2.returncode}")
    print(f"Stdout: {result2.stdout[:500]}...")
    if result2.stderr:
        print(f"Stderr: {result2.stderr[:500]}...")
    
    # Check if test.log was created
    test_log = Path("test.log")
    if test_log.exists():
        print(f"✓ Test log created: {test_log}")
        print(f"  Size: {test_log.stat().st_size} bytes")
    else:
        print("✗ Test log was not created")
    
    print("\n" + "=" * 60)
    
    # Test 3: Run a simple pytest command
    print("Test 3: Running simple pytest command...")
    result3 = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_logging_verification.py::test_logging_success",
        "-v",
        "--log-cli=true",
        "--log-cli-level=INFO"
    ], capture_output=True, text=True)
    
    print(f"Exit code: {result3.returncode}")
    print(f"Stdout: {result3.stdout}")
    if result3.stderr:
        print(f"Stderr: {result3.stderr}")
    
    print("\n" + "=" * 60)
    print("Logging test completed!")
    
    return result1.returncode == 0 and result2.returncode == 0

if __name__ == "__main__":
    success = test_pytest_logging()
    sys.exit(0 if success else 1) 