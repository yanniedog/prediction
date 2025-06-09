import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from sqlite_manager import SQLiteManager

@pytest.fixture(scope="function")
def temp_db_dir() -> Path:
    """Create a temporary directory for the test database."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def db_manager(temp_db_dir: Path) -> SQLiteManager:
    """Create a SQLiteManager instance with a temporary database."""
    db_path = temp_db_dir / "test.db"
    manager = SQLiteManager(db_path)
    yield manager
    manager.close()

@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    data = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.normal(100, 1, 100),
        "high": np.random.normal(101, 1, 100),
        "low": np.random.normal(99, 1, 100),
        "close": np.random.normal(100, 1, 100),
        "volume": np.random.normal(1000, 100, 100)
    })
    return data

def test_db_initialization(temp_db_dir: Path):
    """Test database initialization."""
    db_path = temp_db_dir / "test.db"
    
    # Test new database creation
    manager = SQLiteManager(db_path)
    assert db_path.exists()
    assert manager.is_connected()
    
    # Test connection to existing database
    manager2 = SQLiteManager(db_path)
    assert manager2.is_connected()
    
    # Cleanup
    manager.close()
    manager2.close()

def test_table_operations(db_manager: SQLiteManager):
    """Test table operations."""
    # Test table creation
    db_manager.create_table("test_table", {
        "id": "INTEGER PRIMARY KEY",
        "name": "TEXT",
        "value": "REAL"
    })
    
    # Verify table exists
    tables = db_manager.get_tables()
    assert "test_table" in tables
    
    # Test table deletion
    db_manager.drop_table("test_table")
    tables = db_manager.get_tables()
    assert "test_table" not in tables

def test_data_insertion(db_manager: SQLiteManager, sample_data: pd.DataFrame):
    """Test data insertion operations."""
    # Create test table
    db_manager.create_table("price_data", {
        "timestamp": "TEXT",
        "open": "REAL",
        "high": "REAL",
        "low": "REAL",
        "close": "REAL",
        "volume": "REAL"
    })
    
    # Test single row insertion - convert timestamp to string
    row = sample_data.iloc[0].to_dict()
    row['timestamp'] = str(row['timestamp'])  # Convert timestamp to string
    db_manager.insert_row("price_data", row)
    
    # Verify insertion
    result = db_manager.execute_query("SELECT * FROM price_data")
    assert len(result) == 1
    
    # Test bulk insertion
    db_manager.insert_many("price_data", sample_data.to_dict("records"))
    
    # Verify bulk insertion
    result = db_manager.execute_query("SELECT * FROM price_data")
    assert len(result) == len(sample_data) + 1  # +1 for single row insertion

def test_data_retrieval(db_manager: SQLiteManager, sample_data: pd.DataFrame):
    """Test data retrieval operations."""
    # Create and populate test table
    db_manager.create_table("price_data", {
        "timestamp": "TEXT",
        "open": "REAL",
        "high": "REAL",
        "low": "REAL",
        "close": "REAL",
        "volume": "REAL"
    })
    
    # Convert timestamps to strings before insertion
    data_records = sample_data.to_dict("records")
    for record in data_records:
        record['timestamp'] = str(record['timestamp'])
    
    db_manager.insert_many("price_data", data_records)

    # Test basic query
    result = db_manager.execute_query("SELECT * FROM price_data")
    assert len(result) == len(sample_data)
    
    # Test query with conditions
    result = db_manager.execute_query(
        "SELECT * FROM price_data WHERE close > ?",
        (sample_data["close"].mean(),)
    )
    assert len(result) > 0
    assert all(row["close"] > sample_data["close"].mean() for row in result)
    
    # Test query with joins
    db_manager.create_table("indicators", {
        "timestamp": "TEXT",
        "rsi": "REAL"
    })
    db_manager.insert_many("indicators", [
        {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "rsi": np.random.random()}
        for ts in sample_data["timestamp"]
    ])
    
    result = db_manager.execute_query("""
        SELECT p.*, i.rsi
        FROM price_data p
        JOIN indicators i ON p.timestamp = i.timestamp
    """)
    assert len(result) == len(sample_data)
    assert all("rsi" in row for row in result)

def test_transaction_management(db_manager: SQLiteManager):
    """Test transaction management."""
    # Create test table
    db_manager.create_table("test_table", {
        "id": "INTEGER PRIMARY KEY",
        "value": "TEXT"
    })
    
    # Test successful transaction
    with db_manager.transaction():
        db_manager.insert_row("test_table", {"value": "test1"})
        db_manager.insert_row("test_table", {"value": "test2"})
    
    result = db_manager.execute_query("SELECT * FROM test_table")
    assert len(result) == 2
    
    # Test failed transaction
    try:
        with db_manager.transaction():
            db_manager.insert_row("test_table", {"value": "test3"})
            db_manager.execute_query("INVALID SQL")  # This will fail
    except sqlite3.Error:
        pass
    
    result = db_manager.execute_query("SELECT * FROM test_table")
    assert len(result) == 2  # No new rows should be added

def test_data_update(db_manager: SQLiteManager):
    """Test data update operations."""
    # Create and populate test table
    db_manager.create_table("test_table", {
        "id": "INTEGER PRIMARY KEY",
        "value": "TEXT"
    })
    db_manager.insert_row("test_table", {"value": "old_value"})
    
    # Test single row update
    db_manager.update_row("test_table", {"value": "new_value"}, {"id": 1})
    
    result = db_manager.execute_query("SELECT * FROM test_table WHERE id = 1")
    assert result[0]["value"] == "new_value"
    
    # Test bulk update
    db_manager.insert_many("test_table", [
        {"value": "value1"},
        {"value": "value2"}
    ])
    
    db_manager.update_many("test_table", {"value": "updated"}, {"value": "value1"})
    result = db_manager.execute_query("SELECT * FROM test_table WHERE value = 'updated'")
    assert len(result) == 1

def test_data_deletion(db_manager: SQLiteManager):
    """Test data deletion operations."""
    # Create and populate test table
    db_manager.create_table("test_table", {
        "id": "INTEGER PRIMARY KEY",
        "value": "TEXT"
    })
    db_manager.insert_many("test_table", [
        {"value": "value1"},
        {"value": "value2"},
        {"value": "value3"}
    ])
    
    # Test single row deletion
    db_manager.delete_row("test_table", {"value": "value1"})
    result = db_manager.execute_query("SELECT * FROM test_table")
    assert len(result) == 2
    
    # Test bulk deletion
    db_manager.delete_many("test_table", {"value": "value2"})
    result = db_manager.execute_query("SELECT * FROM test_table")
    assert len(result) == 1

def test_error_handling(db_manager: SQLiteManager):
    """Test error handling."""
    # Test invalid table name
    with pytest.raises(sqlite3.OperationalError):
        db_manager.execute_query("SELECT * FROM non_existent_table")
    
    # Test invalid SQL
    with pytest.raises(sqlite3.OperationalError):
        db_manager.execute_query("INVALID SQL")
    
    # Test invalid column name
    db_manager.create_table("test_table", {"id": "INTEGER PRIMARY KEY"})
    with pytest.raises(sqlite3.OperationalError):
        db_manager.execute_query("SELECT invalid_column FROM test_table")
    
    # Test invalid data type
    with pytest.raises(sqlite3.IntegrityError):
        db_manager.insert_row("test_table", {"id": "invalid"})  # Should be integer

def test_connection_management(db_manager: SQLiteManager):
    """Test connection management."""
    # Test connection status
    assert db_manager.is_connected()
    
    # Test connection closing
    db_manager.close()
    assert not db_manager.is_connected()
    
    # Test reconnection
    db_manager.connect()
    assert db_manager.is_connected()
    
    # Test connection with invalid database
    invalid_manager = SQLiteManager("/invalid/path/test.db")
    assert not invalid_manager.is_connected()

def test_table_schema_management(db_manager: SQLiteManager):
    """Test table schema management."""
    # Test table creation with various column types
    db_manager.create_table("test_table", {
        "id": "INTEGER PRIMARY KEY",
        "text_col": "TEXT",
        "real_col": "REAL",
        "int_col": "INTEGER",
        "blob_col": "BLOB",
        "bool_col": "BOOLEAN"
    })
    
    # Verify schema
    schema = db_manager.get_table_schema("test_table")
    assert len(schema) == 6
    assert any(col["name"] == "id" and col["type"] == "INTEGER" for col in schema)
    
    # Test schema modification
    db_manager.add_column("test_table", "new_col", "TEXT")
    schema = db_manager.get_table_schema("test_table")
    assert any(col["name"] == "new_col" for col in schema)
    
    # Test column removal
    db_manager.drop_column("test_table", "new_col")
    schema = db_manager.get_table_schema("test_table")
    assert not any(col["name"] == "new_col" for col in schema)

def test_index_management(db_manager: SQLiteManager):
    """Test index management."""
    # Create test table
    db_manager.create_table("test_table", {
        "id": "INTEGER PRIMARY KEY",
        "value": "TEXT"
    })
    
    # Test index creation
    db_manager.create_index("test_table", "idx_value", ["value"])
    indexes = db_manager.get_indexes("test_table")
    assert "idx_value" in indexes
    
    # Test index deletion
    db_manager.drop_index("idx_value")
    indexes = db_manager.get_indexes("test_table")
    assert "idx_value" not in indexes 