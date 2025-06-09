import pandas_ta_patch  # Apply patch for pandas_ta NaN import issue
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
import io
import warnings
from contextlib import contextmanager

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from config import Config
from data_manager import DataManager
from indicator_factory import IndicatorFactory
from sqlite_manager import SQLiteManager

# Import config variables directly
from config import (
    PROJECT_ROOT, DB_DIR, REPORTS_DIR, HEATMAPS_DIR, LINE_CHARTS_DIR,
    COMBINED_CHARTS_DIR, LOG_DIR, INDICATOR_PARAMS_PATH, LEADERBOARD_DB_PATH,
    DEFAULTS
)

# Global cleanup registry with locks for thread safety
_cleanup_files = set()
_cleanup_lock = threading.Lock()

# Global variables for output capture
_original_stdout = None
_original_stderr = None
_log_file_handler = None

class TeeOutput:
    """Class to tee output to both console and log file."""
    def __init__(self, original_stream, log_file_handler, stream_name):
        self.original_stream = original_stream
        self.log_file_handler = log_file_handler
        self.stream_name = stream_name
        self.buffer = ""
        self.closed = False
    
    def write(self, text):
        # Write to original stream (console)
        try:
            self.original_stream.write(text)
            self.original_stream.flush()
        except Exception:
            pass  # Ignore errors on console output
        
        # Buffer the text
        self.buffer += text
        
        # If we have a complete line, log it
        if '\n' in text:
            lines = self.buffer.split('\n')
            self.buffer = lines[-1]  # Keep incomplete line in buffer
            
            for line in lines[:-1]:
                if line.strip():  # Only log non-empty lines
                    try:
                        if not self.closed and self.log_file_handler and not self.log_file_handler.closed:
                            self.log_file_handler.write(f"[{self.stream_name}] {line}\n")
                            self.log_file_handler.flush()
                    except Exception:
                        pass  # Ignore errors on log file output
    
    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass
        
        if self.buffer.strip():
            try:
                if not self.closed and self.log_file_handler and not self.log_file_handler.closed:
                    self.log_file_handler.write(f"[{self.stream_name}] {self.buffer}\n")
                    self.log_file_handler.flush()
            except Exception:
                pass
            self.buffer = ""
    
    def fileno(self):
        return self.original_stream.fileno()
    
    def isatty(self):
        return self.original_stream.isatty()
    
    def close(self):
        self.closed = True
        try:
            self.flush()
        except Exception:
            pass

def setup_comprehensive_output_capture():
    """Setup comprehensive output capture to log file."""
    global _original_stdout, _original_stderr, _log_file_handler
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Open log file for writing
    test_log_path = Path("test.log")
    _log_file_handler = open(test_log_path, 'w', encoding='utf-8', buffering=1)  # Line buffered
    
    # Redirect stdout and stderr
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    
    sys.stdout = TeeOutput(_original_stdout, _log_file_handler, "STDOUT")
    sys.stderr = TeeOutput(_original_stderr, _log_file_handler, "STDERR")
    
    # Log the start of capture
    _log_file_handler.write(f"=== COMPREHENSIVE OUTPUT CAPTURE STARTED: {datetime.now()} ===\n")
    _log_file_handler.flush()

def restore_output_capture():
    """Restore original stdout and stderr."""
    global _original_stdout, _original_stderr, _log_file_handler
    
    # Close TeeOutput objects properly
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    if hasattr(sys.stderr, 'close'):
        sys.stderr.close()
    
    if _original_stdout:
        sys.stdout = _original_stdout
    if _original_stderr:
        sys.stderr = _original_stderr
    if _log_file_handler:
        try:
            _log_file_handler.write(f"=== COMPREHENSIVE OUTPUT CAPTURE ENDED: {datetime.now()} ===\n")
            _log_file_handler.close()
        except Exception:
            pass
        _log_file_handler = None

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
    # Setup comprehensive output capture first
    setup_comprehensive_output_capture()
    
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
        restore_output_capture()  # Restore output capture
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Log session start information
    print("=" * 80)
    print(f"PYTEST SESSION CONFIGURED: {datetime.now()}")
    print(f"Comprehensive output capture enabled - ALL output captured in: test.log")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Test discovery path: {Path(__file__).parent}")
    print("=" * 80)

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
    dates = pd.date_range(start='2020-01-01', periods=50, freq='h')
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

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(100, 1, 100).cumsum() + 1000,
        'high': np.random.normal(100, 1, 100).cumsum() + 1000,
        'low': np.random.normal(100, 1, 100).cumsum() + 1000,
        'close': np.random.normal(100, 1, 100).cumsum() + 1000,
        'volume': np.random.lognormal(10, 1, 100)
    })
    
    # Ensure proper price relationships
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    # Ensure no NaN values - only convert numeric columns to float
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    data[numeric_columns] = data[numeric_columns].ffill().bfill().astype(float)
    
    return data

@pytest.fixture
def complex_test_data():
    """Create complex test data with various scenarios."""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='h')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, 200),
        'high': np.random.uniform(200, 300, 200),
        'low': np.random.uniform(50, 100, 200),
        'close': np.random.uniform(100, 200, 200),
        'volume': np.random.uniform(1000, 5000, 200),
        'adj_close': np.random.uniform(100, 200, 200)
    })
    
    # Ensure proper price relationships
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    temp_dir = tmp_path / "temp_test_dir"
    temp_dir.mkdir(exist_ok=True)
    yield temp_dir
    # Cleanup is handled by pytest's tmp_path fixture

@pytest.fixture
def temp_params_file(tmp_path):
    """Create a temporary parameters file for testing."""
    params_file = tmp_path / "test_indicator_params.json"
    yield params_file
    # Cleanup is handled by pytest's tmp_path fixture

@pytest.fixture
def indicator_defs():
    """Provide indicator definitions for testing."""
    return {
        "RSI": {
            "name": "RSI",
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "timeperiod": {
                    "min": 2,
                    "max": 100,
                    "default": 14
                }
            }
        },
        "BB": {
            "name": "BBANDS",
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "timeperiod": {
                    "min": 2,
                    "max": 100,
                    "default": 20
                },
                "nbdevup": {
                    "min": 0.1,
                    "max": 5.0,
                    "default": 2.0
                },
                "nbdevdn": {
                    "min": 0.1,
                    "max": 5.0,
                    "default": 2.0
                }
            }
        },
        "MACD": {
            "name": "MACD",
            "type": "talib",
            "required_inputs": ["close"],
            "params": {
                "fastperiod": {
                    "min": 2,
                    "max": 200,
                    "default": 12
                },
                "slowperiod": {
                    "min": 2,
                    "max": 200,
                    "default": 26
                },
                "signalperiod": {
                    "min": 2,
                    "max": 200,
                    "default": 9
                }
            }
        }
    }

@pytest.fixture(scope="function")
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    db_path = tmp_path / "test.db"
    yield db_path
    # Ensure proper cleanup
    try:
        if db_path.exists():
            # Close any open connections
            import sqlite3
            try:
                conn = sqlite3.connect(str(db_path))
                conn.close()
            except sqlite3.Error:
                pass
            # Wait for file handles to be released
            import time
            time.sleep(0.1)
            # Remove the database file
            db_path.unlink()
            # Also remove any WAL files
            wal_path = db_path.with_suffix('.db-wal')
            if wal_path.exists():
                wal_path.unlink()
            shm_path = db_path.with_suffix('.db-shm')
            if shm_path.exists():
                shm_path.unlink()
    except Exception as e:
        logger.warning(f"Error cleaning up test database {db_path}: {e}")

@pytest.fixture(scope="function")
def temp_db_dir(tmp_path):
    """Create a temporary directory for test databases."""
    db_dir = tmp_path / "test_dbs"
    db_dir.mkdir(exist_ok=True)
    yield db_dir
    # Ensure proper cleanup
    try:
        if db_dir.exists():
            # Close any open connections
            import sqlite3
            for db_file in db_dir.glob("*.db*"):
                try:
                    conn = sqlite3.connect(str(db_file))
                    conn.close()
                except sqlite3.Error:
                    pass
            # Wait for file handles to be released
            import time
            time.sleep(0.1)
            # Remove all database files
            for db_file in db_dir.glob("*.db*"):
                db_file.unlink()
            # Remove the directory
            db_dir.rmdir()
    except Exception as e:
        logger.warning(f"Error cleaning up test database directory {db_dir}: {e}")

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

@pytest.fixture(scope="module")
def factory():
    """Create a factory instance for testing."""
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

@pytest.fixture(autouse=True, scope='session')
def mock_input_global():
    original_input = builtins.input
    builtins.input = lambda *args, **kwargs: ''
    yield
    builtins.input = original_input 

# Configure comprehensive logging to capture everything
def setup_comprehensive_logging():
    """Set up comprehensive logging to capture all output."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger to capture everything
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create file handler for test.log
    test_log_path = Path("test.log")
    file_handler = logging.FileHandler(test_log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s (%(filename)s:%(lineno)s)',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(file_handler)
    
    # Also log to console for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return test_log_path

# Set up logging at module import time
test_log_path = setup_comprehensive_logging()

class ComprehensiveOutputCapture:
    """Capture all output including print statements, stdout, stderr, and logs."""
    
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_capture = None
        self.stderr_capture = None
        
    def start_capture(self):
        """Start capturing all output."""
        # Capture stdout and stderr
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        
    def stop_capture(self):
        """Stop capturing and write to log file."""
        if self.stdout_capture and self.stderr_capture:
            # Restore original streams
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            # Get captured output
            stdout_content = self.stdout_capture.getvalue()
            stderr_content = self.stderr_capture.getvalue()
            
            # Write to log file
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                if stdout_content:
                    f.write(f"\n=== STDOUT ===\n{stdout_content}\n")
                if stderr_content:
                    f.write(f"\n=== STDERR ===\n{stderr_content}\n")
            
            # Close captures
            self.stdout_capture.close()
            self.stderr_capture.close()
            self.stdout_capture = None
            self.stderr_capture = None

@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up comprehensive test logging for the entire test session."""
    # Configure logging to work with pytest
    logger = logging.getLogger(__name__)
    
    # Ensure we have proper logging configuration
    if not logging.getLogger().handlers:
        # Set up basic logging if not already configured
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s (%(filename)s:%(lineno)s)',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    logger.info("=" * 80)
    logger.info(f"TEST SESSION STARTED: {datetime.now()}")
    logger.info("=" * 80)
    
    yield
    
    logger.info("=" * 80)
    logger.info(f"TEST SESSION ENDED: {datetime.now()}")
    logger.info("=" * 80)

@pytest.fixture(autouse=True)
def capture_test_output():
    """Capture all output for each test including print statements, stdout, stderr, and logs."""
    logger = logging.getLogger(__name__)
    
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create string buffers to capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Redirect streams
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    # Log test start
    test_name = getattr(pytest, 'current_test_name', 'Unknown')
    logger.info(f"Starting test: {test_name}")
    logger.info("-" * 60)
    
    try:
        yield
    finally:
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Get captured output
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()
        
        # Log captured output
        if stdout_content:
            logger.info("=== STDOUT CAPTURED ===")
            for line in stdout_content.splitlines():
                if line.strip():
                    logger.info(f"STDOUT: {line}")
            logger.info("=== END STDOUT ===")
        
        if stderr_content:
            logger.warning("=== STDERR CAPTURED ===")
            for line in stderr_content.splitlines():
                if line.strip():
                    logger.warning(f"STDERR: {line}")
            logger.warning("=== END STDERR ===")
        
        # Close captures
        stdout_capture.close()
        stderr_capture.close()
        
        # Log test completion
        logger.info("-" * 60)
        logger.info(f"Completed test: {test_name}")

@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before and after each test."""
    # Store original level
    original_level = logging.getLogger().level
    
    # Close all existing handlers to prevent ResourceWarnings
    for handler in logging.getLogger().handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
    
    # Reset logging
    logging.getLogger().handlers = []
    logging.getLogger().level = logging.NOTSET
    
    yield
    
    # Close any handlers that might have been created during the test
    for handler in logging.getLogger().handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
    
    # Clear all handlers and restore level only
    logging.getLogger().handlers = []
    logging.getLogger().level = original_level

# Hook to capture test names
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test names and results for logging."""
    logger = logging.getLogger(__name__)
    
    # Set test name for other fixtures
    pytest.current_test_name = f"{item.name}"
    
    # Log test phase
    logger.info(f"Test phase {call.when}: {item.name}")
    
    outcome = yield
    
    # Log test result
    if outcome.excinfo:
        logger.error(f"Test FAILED: {item.name} - {outcome.excinfo[1]}")
        logger.error(f"Exception type: {outcome.excinfo[0].__name__ if outcome.excinfo[0] else 'Unknown'}")
        logger.error(f"Exception traceback: {outcome.excinfo[2]}")
    else:
        logger.info(f"Test PASSED: {item.name}")

def _flush_all_log_handlers():
    """Flush all log handlers to ensure messages are written."""
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass

# Log every test's start, finish, and result
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Log test protocol events."""
    logger = logging.getLogger()
    logger.info(f"[TEST START] {item.nodeid}")
    _flush_all_log_handlers()
    outcome = yield
    logger.info(f"[TEST END] {item.nodeid}")
    _flush_all_log_handlers()

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    """Log test report information."""
    logger = logging.getLogger()
    if report.when == 'call':
        if report.failed:
            logger.error(f"[TEST FAIL] {report.nodeid}")
            if hasattr(report, 'longrepr') and report.longrepr:
                logger.error(f"Failure details: {report.longrepr}")
        elif report.passed:
            logger.info(f"[TEST PASS] {report.nodeid}")
        elif report.skipped:
            logger.warning(f"[TEST SKIP] {report.nodeid}")
        _flush_all_log_handlers()

# Hook to capture session start/end
def pytest_sessionstart(session):
    """Log session start with comprehensive information."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"PYTEST SESSION STARTED: {session.name}")
    logger.info(f"Session start time: {datetime.now()}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Test collection: {len(session.items)} tests collected")
    logger.info("=" * 80)

def pytest_sessionfinish(session, exitstatus):
    """Clean up after test session."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"PYTEST SESSION FINISHED:")
    logger.info(f"Session end time: {datetime.now()}")
    logger.info(f"Exit status: {exitstatus}")
    logger.info(f"Exit status name: {exitstatus.name if hasattr(exitstatus, 'name') else 'Unknown'}")
    logger.info(f"Exit status value: {exitstatus.value if hasattr(exitstatus, 'value') else 'Unknown'}")
    logger.info("=" * 80)
    
    # Restore output capture
    restore_output_capture()

# Hook to capture all warnings
def pytest_warning_recorded(warning_message, when, nodeid, location):
    """Log all warnings with detailed information."""
    logger = logging.getLogger(__name__)
    logger.warning(f"WARNING in {nodeid} at {when}: {warning_message.message}")
    if location:
        logger.warning(f"Warning location: {location}")

# Hook to capture all errors
def pytest_exception_interact(call, report):
    """Log all exceptions with detailed information."""
    logger = logging.getLogger(__name__)
    logger.error(f"EXCEPTION in {report.nodeid}: {call.excinfo}")
    if call.excinfo:
        logger.error(f"Exception type: {getattr(call.excinfo, 'type', 'Unknown')}")
        logger.error(f"Exception message: {getattr(call.excinfo, 'value', 'Unknown')}")
        logger.error(f"Exception traceback: {getattr(call.excinfo, 'traceback', 'Unknown')}")

# Hook to capture test collection
def pytest_collection_modifyitems(session, config, items):
    """Log test collection information."""
    logger = logging.getLogger(__name__)
    logger.info(f"Test collection completed: {len(items)} tests collected")
    logger.info(f"Test discovery paths: {config.getini('testpaths')}")

# Hook to capture test setup/teardown
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Log test setup."""
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up test: {item.name}")
    outcome = yield
    logger.info(f"Setup completed for test: {item.name}")

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Log test teardown."""
    logger = logging.getLogger(__name__)
    logger.info(f"Tearing down test: {item.name}")
    outcome = yield
    logger.info(f"Teardown completed for test: {item.name}")

# Ensure the log file is created and accessible
if not test_log_path.exists():
    test_log_path.touch()

print(f"Comprehensive test logging configured. All output will be captured in: {test_log_path.absolute()}")

@pytest.fixture(autouse=True)
def log_test_output(request, capsys):
    """Log test output and ensure proper cleanup."""
    logger = logging.getLogger(__name__)
    
    # Log test start
    logger.info(f"Test starting: {request.node.name}")
    
    yield
    
    # Capture and log any remaining output
    captured = capsys.readouterr()
    if captured.out:
        logger.info(f"Remaining stdout: {captured.out}")
    if captured.err:
        logger.warning(f"Remaining stderr: {captured.err}")
    
    # Log test completion
    logger.info(f"Test completed: {request.node.name}") 