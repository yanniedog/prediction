import pytest
import sqlite3
from pathlib import Path
from main import _initialize_database
from utils import is_valid_symbol, is_valid_timeframe

@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "BTCUSDT_1h.db"

@pytest.fixture
def valid_symbol():
    return "BTCUSDT"

@pytest.fixture
def valid_timeframe():
    return "1h"

@pytest.fixture
def invalid_symbol():
    return "invalid-symbol"

@pytest.fixture
def invalid_timeframe():
    return "invalid"

def test_database_initialization(temp_db_path, valid_symbol, valid_timeframe):
    """Test successful database initialization"""
    result = _initialize_database(temp_db_path, valid_symbol, valid_timeframe)
    assert result is True
    assert temp_db_path.exists()
    
    # Verify database schema
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "leaderboard" in tables
    assert "symbols" in tables
    assert "timeframes" in tables
    assert "indicators" in tables
    assert "indicator_configs" in tables
    assert "historical_data" in tables
    assert "correlations" in tables
    
    # Check leaderboard table schema
    cursor.execute("PRAGMA table_info(leaderboard)")
    columns = {row[1] for row in cursor.fetchall()}
    required_columns = {
        "lag", "correlation_type", "correlation_value", 
        "indicator_name", "config_json", "symbol", "timeframe",
        "dataset_daterange", "calculation_timestamp",
        "config_id_source_db", "source_db_name"
    }
    assert required_columns.issubset(columns)
    
    # Check indices
    cursor.execute("PRAGMA index_list(leaderboard)")
    indices = {row[1] for row in cursor.fetchall()}
    assert "idx_leaderboard_lag" in indices
    assert "idx_leaderboard_lag_val" in indices
    assert "idx_leaderboard_indicator_config" in indices
    
    conn.close()

def test_invalid_symbol(temp_db_path, invalid_symbol, valid_timeframe):
    """Test database initialization with invalid symbol"""
    with pytest.raises(ValueError, match="Invalid symbol format"):
        _initialize_database(temp_db_path, invalid_symbol, valid_timeframe)

def test_invalid_timeframe(temp_db_path, valid_symbol, invalid_timeframe):
    """Test database initialization with invalid timeframe"""
    with pytest.raises(ValueError, match="Invalid timeframe format"):
        _initialize_database(temp_db_path, valid_symbol, invalid_timeframe)

def test_symbol_validation():
    """Test symbol format validation"""
    assert is_valid_symbol("BTCUSDT") is True
    assert is_valid_symbol("ETHUSDT") is True
    assert is_valid_symbol("invalid-symbol") is False
    assert is_valid_symbol("123") is False
    assert is_valid_symbol("") is False
    assert is_valid_symbol("BTC/USDT") is False

def test_timeframe_validation():
    """Test timeframe format validation"""
    # Valid timeframes
    assert is_valid_timeframe("1m") is True
    assert is_valid_timeframe("3m") is True
    assert is_valid_timeframe("5m") is True
    assert is_valid_timeframe("15m") is True
    assert is_valid_timeframe("30m") is True
    assert is_valid_timeframe("1h") is True
    assert is_valid_timeframe("2h") is True
    assert is_valid_timeframe("4h") is True
    assert is_valid_timeframe("6h") is True
    assert is_valid_timeframe("8h") is True
    assert is_valid_timeframe("12h") is True
    assert is_valid_timeframe("1d") is True
    assert is_valid_timeframe("3d") is True
    assert is_valid_timeframe("1w") is True
    assert is_valid_timeframe("1M") is True  # Capital M for month
    
    # Invalid timeframes
    assert is_valid_timeframe("invalid") is False
    assert is_valid_timeframe("1") is False
    assert is_valid_timeframe("") is False
    assert is_valid_timeframe("1y") is False  # Years not supported
    assert is_valid_timeframe("2m") is False  # Only specific minute intervals
    assert is_valid_timeframe("3h") is False  # Only specific hour intervals
    assert is_valid_timeframe("2d") is False  # Only 1d and 3d supported
    assert is_valid_timeframe("2w") is False  # Only 1w supported
    assert is_valid_timeframe("2M") is False  # Only 1M supported

def test_database_connection_handling():
    """Test database connection handling and retry logic."""
    test_db = Path("test.db")
    if test_db.exists():
        try:
            test_db.unlink()
        except PermissionError:
            import time
            time.sleep(0.1)
            test_db.unlink()
        
    # Test connection creation
    conn1 = sqlite3.connect(str(test_db))
    assert conn1 is not None, "Failed to create first connection"
    
    # Test concurrent connections
    conn2 = sqlite3.connect(str(test_db))
    assert conn2 is not None, "Failed to create second connection"
    
    try:
        # Initialize database
        _initialize_database(str(test_db), "BTCUSDT", "1h")
        
        # Test concurrent writes with separate transactions
        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()
        
        # Insert test data in separate transactions
        cursor1.execute("BEGIN")
        cursor1.execute("INSERT INTO symbols (symbol) VALUES (?)", ("ETHUSDT",))
        cursor1.execute("COMMIT")
        
        cursor2.execute("BEGIN")
        cursor2.execute("INSERT INTO timeframes (timeframe) VALUES (?)", ("1d",))
        cursor2.execute("COMMIT")
        
        # Verify data
        cursor1.execute("SELECT symbol FROM symbols WHERE symbol = ?", ("ETHUSDT",))
        assert cursor1.fetchone()[0] == "ETHUSDT", "Failed to verify symbol insertion"
        
        cursor2.execute("SELECT timeframe FROM timeframes WHERE timeframe = ?", ("1d",))
        assert cursor2.fetchone()[0] == "1d", "Failed to verify timeframe insertion"
        
    finally:
        conn1.close()
        conn2.close()
        import time
        time.sleep(0.1)  # Give time for connections to close
        if test_db.exists():
            try:
                test_db.unlink()
            except PermissionError:
                pass  # File might still be in use, that's okay for tests

def test_database_recovery():
    """Test database recovery from corruption."""
    test_db = Path("test.db")
    if test_db.exists():
        try:
            test_db.unlink()
        except PermissionError:
            import time
            time.sleep(0.1)
            test_db.unlink()
        
    # Create and initialize database
    conn = sqlite3.connect(str(test_db))
    assert conn is not None, "Failed to create database connection"
    conn.close()
    
    try:
        _initialize_database(str(test_db), "BTCUSDT", "1h")
        
        # Corrupt database by writing invalid data
        with open(test_db, 'wb') as f:
            f.write(b'invalid data')
            
        # Remove the corrupted file and recreate
        test_db.unlink()
        
        # Attempt to initialize fresh database
        result = _initialize_database(str(test_db), "BTCUSDT", "1h")
        assert result is True, "Failed to recreate database after corruption"
        
        # Verify database was recreated
        conn = sqlite3.connect(str(test_db))
        assert conn is not None, "Failed to connect to recovered database"
        
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "symbols" in tables, "Symbols table not found in recovered database"
        assert "timeframes" in tables, "Timeframes table not found in recovered database"
        
        conn.close()
        
    finally:
        if test_db.exists():
            try:
                test_db.unlink()
            except PermissionError:
                pass  # File might still be in use, that's okay for tests 