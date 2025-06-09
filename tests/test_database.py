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
    result = _initialize_database(temp_db_path, invalid_symbol, valid_timeframe)
    assert result is False
    assert not temp_db_path.exists()

def test_invalid_timeframe(temp_db_path, valid_symbol, invalid_timeframe):
    """Test database initialization with invalid timeframe"""
    result = _initialize_database(temp_db_path, valid_symbol, invalid_timeframe)
    assert result is False
    assert not temp_db_path.exists()

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

def test_database_connection_handling(temp_db_path, valid_symbol, valid_timeframe):
    """Test database connection handling"""
    # Create database
    _initialize_database(temp_db_path, valid_symbol, valid_timeframe)

    # Test concurrent connections with retry logic
    import time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            conn1 = sqlite3.connect(temp_db_path)
            conn2 = sqlite3.connect(temp_db_path)
            cursor1 = conn1.cursor()
            cursor2 = conn2.cursor()
            cursor1.execute("INSERT INTO symbols (symbol) VALUES (?)", (valid_symbol,))
            cursor2.execute("INSERT INTO timeframes (timeframe) VALUES (?)", (valid_timeframe,))
            conn1.commit()
            conn2.commit()
            cursor1.execute("SELECT symbol FROM symbols WHERE symbol = ?", (valid_symbol,))
            assert cursor1.fetchone()[0] == valid_symbol
            cursor2.execute("SELECT timeframe FROM timeframes WHERE timeframe = ?", (valid_timeframe,))
            assert cursor2.fetchone()[0] == valid_timeframe
            conn1.close()
            conn2.close()
            break
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(0.2)
                continue
            else:
                raise

def test_leaderboard_constraints(temp_db_path, valid_symbol, valid_timeframe):
    """Test leaderboard table constraints"""
    _initialize_database(temp_db_path, valid_symbol, valid_timeframe)
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    # Insert symbol and timeframe
    cursor.execute("INSERT INTO symbols (symbol) VALUES (?)", (valid_symbol,))
    cursor.execute("INSERT INTO timeframes (timeframe) VALUES (?)", (valid_timeframe,))
    symbol_id = cursor.lastrowid
    timeframe_id = cursor.lastrowid
    # Test unique constraint (adjust columns as per schema)
    cursor.execute("""
        INSERT INTO leaderboard (
            symbol_id, timeframe_id, predictor_name, parameters, performance_metrics, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, strftime('%s','now'), strftime('%s','now'))
    """, (symbol_id, timeframe_id, "RSI", "{}", "{}"))
    # Try to insert duplicate
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute("""
            INSERT INTO leaderboard (
                symbol_id, timeframe_id, predictor_name, parameters, performance_metrics, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, strftime('%s','now'), strftime('%s','now'))
        """, (symbol_id, timeframe_id, "RSI", "{}", "{}"))
    conn.close()

def test_database_rollback(temp_db_path, valid_symbol, valid_timeframe):
    """Test database transaction rollback"""
    _initialize_database(temp_db_path, valid_symbol, valid_timeframe)
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN TRANSACTION")
        # Insert symbol
        cursor.execute("INSERT INTO symbols (symbol) VALUES (?)", (valid_symbol,))
        symbol_id = cursor.lastrowid
        # Insert invalid data to trigger rollback
        with pytest.raises(sqlite3.OperationalError):
            cursor.execute("INSERT INTO leaderboard (invalid_column) VALUES (?)", (1,))
        cursor.execute("ROLLBACK")
    except sqlite3.OperationalError:
        cursor.execute("ROLLBACK")
    # Verify rollback
    cursor.execute("SELECT COUNT(*) FROM symbols")
    assert cursor.fetchone()[0] == 0
    conn.close()

def test_database_recovery(temp_db_path, valid_symbol, valid_timeframe):
    """Test database recovery after corruption"""
    # Create and initialize database
    _initialize_database(temp_db_path, valid_symbol, valid_timeframe)
    
    # Corrupt database file
    with open(temp_db_path, 'wb') as f:
        f.write(b'corrupted data')
    
    # Attempt to initialize again
    result = _initialize_database(temp_db_path, valid_symbol, valid_timeframe)
    assert result is True
    
    # Verify database is usable
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "leaderboard" in tables
    conn.close() 