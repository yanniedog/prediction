import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil
import threading
from functools import wraps
import time
import atexit
import signal
import logging
import psutil
import gc
import builtins

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from config import Config
from data_manager import DataManager
from indicator_factory import IndicatorFactory
from sqlite_manager_class import SQLiteManager

# Import config variables directly
from config import (
    PROJECT_ROOT, DB_DIR, REPORTS_DIR, HEATMAPS_DIR, LINE_CHARTS_DIR,
    COMBINED_CHARTS_DIR, LOG_DIR, INDICATOR_PARAMS_PATH, LEADERBOARD_DB_PATH,
    DEFAULTS
)

# Global cleanup registry with locks for thread safety
_cleanup_files = set()
_cleanup_lock = threading.Lock()

def register_cleanup(path):
    """Register a file for cleanup at exit."""
    with _cleanup_lock:
        _cleanup_files.add(str(path))

def force_close_file_handles(path):
    """Force close any open file handles for the given path."""
    try:
        path_str = str(path)
        # Only check processes that might have our file open
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # Skip processes that can't have our file open
                if proc.info['name'] not in ['python.exe', 'pythonw.exe', 'pytest.exe']:
                    continue
                # Only get open files for relevant processes
                for file in proc.open_files():
                    if file.path == path_str:
                        try:
                            # Try to close the file handle
                            os.close(file.fd)
                        except (OSError, ProcessLookupError):
                            # If we can't close it, try to kill the process
                            try:
                                proc.kill()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        break  # Found and handled the file, no need to check other files
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

def cleanup_files():
    """Clean up all registered files with improved error handling."""
    with _cleanup_lock:
        files_to_clean = _cleanup_files.copy()
        _cleanup_files.clear()
    
    for path in files_to_clean:
        try:
            path = Path(path)
            if not path.exists():
                continue
            
            # Force close any open handles
            force_close_file_handles(path)
            
            # Try to remove the file with retries
            max_retries = 3
            retry_delay = 0.1  # seconds
            for attempt in range(max_retries):
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    break  # Success, exit retry loop
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        force_close_file_handles(path)  # Try closing handles again
                    else:
                        # Log the failure but don't raise
                        logging.warning(f"Failed to clean up file after {max_retries} attempts: {path}")
                except Exception:
                    break  # Exit retry loop on other errors
        except Exception:
            pass

# Register cleanup at exit with a timeout
def cleanup_with_timeout():
    """Run cleanup with a timeout to prevent stalls."""
    cleanup_thread = threading.Thread(target=cleanup_files)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    cleanup_thread.join(timeout=2.0)  # Reduced timeout to 2 seconds

atexit.register(cleanup_with_timeout)

# --- Global Test Configuration ---
def pytest_configure(config):
    """Configure pytest with global settings."""
    # Add custom markers
    config.addinivalue_line("markers", "timeout(seconds): mark test to timeout after specified seconds")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "visualization: mark test as requiring visualization dependencies")
    config.addinivalue_line("markers", "full_data: mark test as needing full dataset")
    config.addinivalue_line("markers", "no_stdin: mark test as not using stdin")
    
    # Set up signal handlers for cleanup
    def signal_handler(signum, frame):
        cleanup_with_timeout()  # Use timeout version
        # Force cleanup matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except (ImportError, NameError):
            pass
        # Force garbage collection
        gc.collect()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # --- Custom Logging Setup ---
    root_logger = logging.getLogger()
    # Remove all handlers (pytest may add its own, so be careful)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # File handler: verbose format
    file_handler = logging.FileHandler('test.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Stream handler: simple format (just the message) – force WARNING (or lower) so that critical, error, and warning logs are shown
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)  # (or lower, e.g. logging.DEBUG if you want all logs)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)
    root_logger.addHandler(stream_handler)

    # Set higher log level for leaderboard manager to reduce output
    logging.getLogger('leaderboard_manager').setLevel(logging.WARNING)

    # Force root logger level to WARNING (or lower) so that critical, error, and warning logs are shown
    root_logger.setLevel(logging.WARNING)  # (or lower, e.g. logging.DEBUG if you want all logs)

# --- Global Fixtures ---
@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Fixture to ensure matplotlib figures are cleaned up after each test."""
    import matplotlib.pyplot as plt
    plt.close('all')  # Close all figures before test
    yield
    plt.close('all')  # Close all figures after test
    gc.collect()  # Force garbage collection

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Fixture to clean up temporary files after each test."""
    yield
    cleanup_with_timeout()  # Use timeout version
    # Force cleanup matplotlib again
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except (ImportError, NameError):
        pass
    gc.collect()  # Force garbage collection

@pytest.fixture(autouse=True)
def handle_timeout(request):
    """Fixture to handle test timeouts."""
    if hasattr(request.node, 'get_closest_marker'):
        timeout_marker = request.node.get_closest_marker('timeout')
        if timeout_marker:
            timeout_seconds = timeout_marker.args[0] if timeout_marker.args else 30
            start_time = time.time()
            yield
            duration = time.time() - start_time
            if duration > timeout_seconds:
                pytest.fail(f"Test exceeded timeout of {timeout_seconds} seconds (took {duration:.1f}s)")
        else:
            yield
    else:
        yield

@pytest.fixture(autouse=True)
def limit_test_data_size(request):
    """Fixture to limit test data size unless explicitly marked as needing full data."""
    if not hasattr(request, 'node'):
        return
    
    # Skip if test is marked as needing full data
    if request.node.get_closest_marker('full_data'):
        return
    
    # For any test using test_data fixture, limit the size
    if 'test_data' in request.fixturenames:
        original_data = request.getfixturevalue('test_data')
        if isinstance(original_data, pd.DataFrame):
            # Use only first 50 rows for most tests
            return original_data.iloc[:50].copy()
    return None

@pytest.fixture(autouse=True)
def handle_stdin(request):
    """Fixture to handle stdin for tests marked with no_stdin."""
    if request.node.get_closest_marker('no_stdin'):
        # Mock input for tests that shouldn't use stdin
        original_input = builtins.input
        builtins.input = lambda _: 'test_input'
        yield
        builtins.input = original_input
    else:
        yield

# --- Test Data Fixture ---
@pytest.fixture
def test_data():
    """Provide test data with controlled size."""
    # Generate synthetic test data
    dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
    np.random.seed(42)  # For reproducibility
    
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 50).cumsum() + 1000,
        'high': np.random.normal(101, 1, 50).cumsum() + 1000,
        'low': np.random.normal(99, 1, 50).cumsum() + 1000,
        'close': np.random.normal(100, 1, 50).cumsum() + 1000,
        'volume': np.random.lognormal(10, 1, 50)
    }, index=dates)  # Use dates directly as index
    
    # Add some realistic features
    data['close'] = data['close'].rolling(5, min_periods=1).mean()
    data['volume'] = data['volume'].rolling(3, min_periods=1).mean()
    
    # Ensure no NaN values
    data = data.ffill().bfill().astype(float)
    
    return data

@pytest.fixture(scope="function")
def temp_db(request):
    """Provide a temporary database for testing."""
    # Create a unique database file for each test
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    # Register for cleanup
    register_cleanup(db_path)
    
    # Create and yield the manager
    manager = None
    try:
        manager = SQLiteManager(str(db_path))
        # Set a shorter timeout for tests
        manager.connection.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout
        # Initialize the database schema
        if not manager.initialize_database():
            raise RuntimeError("Failed to initialize test database schema")
        yield manager
    finally:
        # Ensure connection is closed
        if manager is not None:
            try:
                manager.connection.close()
            except Exception:
                pass
            manager = None  # Clear reference
        
        # Force cleanup with retries
        max_retries = 3
        retry_delay = 0.1
        for attempt in range(max_retries):
            try:
                force_close_file_handles(db_path)
                if db_path.exists():
                    db_path.unlink()
                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    # Log the failure but don't raise
                    logging.warning(f"Failed to clean up temp DB after {max_retries} attempts: {db_path}")
        
        # Force garbage collection
        gc.collect()

@pytest.fixture(scope="session")
def config():
    """Fixture to provide configuration settings"""
    return Config().get_config_dict()

# --- Helper Functions ---
def assert_timeout(func, *args, timeout_seconds=30, **kwargs):
    """Helper to assert a function completes within timeout."""
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        if duration >= timeout_seconds:
            pytest.fail(f"Operation took {duration:.1f}s, exceeding {timeout_seconds}s timeout")
        return result
    except Exception as e:
        duration = time.time() - start_time
        if duration >= timeout_seconds:
            pytest.fail(f"Operation failed after {duration:.1f}s, exceeding {timeout_seconds}s timeout: {str(e)}")
        raise  # Re-raise the original exception if not a timeout

def skip_if_missing_dependency(dependency_name):
    """Helper to skip tests if a dependency is missing."""
    try:
        __import__(dependency_name)
    except ImportError:
        pytest.skip(f"Required dependency {dependency_name} not available")

@pytest.fixture(scope="session")
def data_manager(config):
    """Provide a DataManager instance for testing."""
    return DataManager(config)

@pytest.fixture(scope="session")
def indicator_factory():
    """Provide an IndicatorFactory instance for testing."""
    return IndicatorFactory()

@pytest.fixture(scope="session")
def sqlite_manager(config):
    """Provide a SQLiteManager instance for testing."""
    db_path = str(project_root / "test_correlation_leaderboard.db")
    manager = SQLiteManager(db_path)
    yield manager
    # Cleanup after tests
    if os.path.exists(db_path):
        os.remove(db_path) 