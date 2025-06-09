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
            
            # Use different symbols and timeframes for concurrent connections
            cursor1.execute("INSERT INTO symbols (symbol) VALUES (?)", ("ETHUSDT",))
            cursor2.execute("INSERT INTO timeframes (timeframe) VALUES (?)", ("4h",))
            
            conn1.commit()
            conn2.commit()
            
            cursor1.execute("SELECT symbol FROM symbols WHERE symbol = ?", ("ETHUSDT",))
            assert cursor1.fetchone()[0] == "ETHUSDT"
            cursor2.execute("SELECT timeframe FROM timeframes WHERE timeframe = ?", ("4h",))
            assert cursor2.fetchone()[0] == "4h"
            
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
    
    # Test unique constraint on (lag, correlation_type)
    cursor.execute("""
        INSERT INTO leaderboard (
            lag, correlation_type, correlation_value, indicator_name, config_json,
            symbol, timeframe, dataset_daterange, calculation_timestamp,
            config_id_source_db, source_db_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (1, 'positive', 0.5, 'RSI', '{}', 'BTCUSDT', '1h', '2023-01-01/2023-12-31', 
          '2024-01-01T00:00:00.000Z', 1, 'test.db'))
    
    # Try to insert duplicate (lag, correlation_type)
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute("""
            INSERT INTO leaderboard (
                lag, correlation_type, correlation_value, indicator_name, config_json,
                symbol, timeframe, dataset_daterange, calculation_timestamp,
                config_id_source_db, source_db_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, 'positive', 0.6, 'MACD', '{}', 'ETHUSDT', '4h', '2023-01-01/2023-12-31',
              '2024-01-01T00:00:00.000Z', 2, 'test2.db'))
    
    # Test correlation_type check constraint
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute("""
            INSERT INTO leaderboard (
                lag, correlation_type, correlation_value, indicator_name, config_json,
                symbol, timeframe, dataset_daterange, calculation_timestamp,
                config_id_source_db, source_db_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (2, 'invalid', 0.5, 'RSI', '{}', 'BTCUSDT', '1h', '2023-01-01/2023-12-31',
              '2024-01-01T00:00:00.000Z', 1, 'test.db'))
    
    conn.close()

def test_database_rollback(temp_db_path, valid_symbol, valid_timeframe):
    """Test database transaction rollback"""
    _initialize_database(temp_db_path, valid_symbol, valid_timeframe)
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN TRANSACTION")
        # Insert different symbol for testing
        cursor.execute("INSERT INTO symbols (symbol) VALUES (?)", ("ETHUSDT",))
        symbol_id = cursor.lastrowid
        # Insert invalid data to trigger rollback
        with pytest.raises(sqlite3.OperationalError):
            cursor.execute("INSERT INTO leaderboard (invalid_column) VALUES (?)", (1,))
        cursor.execute("ROLLBACK")
    except sqlite3.OperationalError:
        cursor.execute("ROLLBACK")
    # Verify rollback
    cursor.execute("SELECT COUNT(*) FROM symbols WHERE symbol = ?", ("ETHUSDT",))
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

def test_database_initialization():
    """Test database initialization with required tables and columns."""
    # Create a test database
    test_db = Path("test.db")
    if test_db.exists():
        test_db.unlink()
    
    conn = sqlite3.connect(str(test_db))
    assert conn is not None, "Failed to create database connection"
    
    try:
        # Initialize database
        _initialize_database(str(test_db), "BTCUSDT", "1h")
        
        # Verify tables exist
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        required_tables = {
            'symbols', 'timeframes', 'indicators', 'indicator_configs',
            'historical_data', 'correlations', 'leaderboard'
        }
        assert tables.issuperset(required_tables), f"Missing tables: {required_tables - tables}"
        
        # Verify columns in historical_data table
        cursor.execute("PRAGMA table_info(historical_data)")
        columns = {row[1] for row in cursor.fetchall()}
        required_columns = {
            'id', 'symbol_id', 'timeframe_id', 'open_time', 'open', 'high',
            'low', 'close', 'volume', 'created_at'
        }
        assert columns.issuperset(required_columns), f"Missing columns in historical_data: {required_columns - columns}"
        
        # Verify foreign key constraints
        cursor.execute("PRAGMA foreign_key_check")
        assert cursor.fetchone() is None, "Foreign key constraint violation"
        
        # Verify indices
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in cursor.fetchall()}
        required_indices = {
            'idx_historical_data_symbol_timeframe',
            'idx_historical_data_open_time',
            'idx_correlations_symbols',
            'idx_leaderboard_correlation'
        }
        assert indices.issuperset(required_indices), f"Missing indices: {required_indices - indices}"
        
    finally:
        conn.close()
        import time
        time.sleep(0.1)  # Ensure file is released before deletion
        if test_db.exists():
            test_db.unlink()

def test_database_connection_handling():
    """Test database connection handling and retry logic."""
    test_db = Path("test.db")
    if test_db.exists():
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
        
        # Test concurrent writes
        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()
        
        # Insert test data
        cursor1.execute("INSERT INTO symbols (symbol) VALUES (?)", ("BTCUSDT",))
        cursor2.execute("INSERT INTO timeframes (timeframe) VALUES (?)", ("1d",))
        
        conn1.commit()
        conn2.commit()
        
        # Verify data
        cursor1.execute("SELECT symbol FROM symbols WHERE symbol = ?", ("BTCUSDT",))
        assert cursor1.fetchone()[0] == "BTCUSDT", "Failed to verify symbol insertion"
        
        cursor2.execute("SELECT timeframe FROM timeframes WHERE timeframe = ?", ("1d",))
        assert cursor2.fetchone()[0] == "1d", "Failed to verify timeframe insertion"
        
    finally:
        conn1.close()
        conn2.close()
        if test_db.exists():
            test_db.unlink()

def test_database_recovery():
    """Test database recovery from corruption."""
    test_db = Path("test.db")
    if test_db.exists():
        test_db.unlink()
        
    # Create and initialize database
    conn = sqlite3.connect(str(test_db))
    assert conn is not None, "Failed to create database connection"
    
    try:
        _initialize_database(str(test_db), "BTCUSDT", "1h")
        
        # Corrupt database by writing invalid data
        with open(test_db, 'wb') as f:
            f.write(b'invalid data')
            
        # Attempt to initialize corrupted database
        _initialize_database(str(test_db), "BTCUSDT", "1h")
        
        # Verify database was recreated
        conn = sqlite3.connect(str(test_db))
        assert conn is not None, "Failed to connect to recovered database"
        
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        required_tables = {
            'symbols', 'timeframes', 'indicators', 'indicator_configs',
            'historical_data', 'correlations', 'leaderboard'
        }
        assert tables.issuperset(required_tables), "Database not properly recovered"
        
    finally:
        if conn:
            conn.close()
        if test_db.exists():
            test_db.unlink() 