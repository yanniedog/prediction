import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlite_manager import SQLiteManager
import time
from unittest.mock import patch
import sqlite3
import tempfile
import shutil
from pathlib import Path

@pytest.fixture(scope="function")
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    sqlite_manager = SQLiteManager(str(db_path))
    return sqlite_manager

@pytest.fixture(scope="function")
def sqlite_manager(temp_db):
    return temp_db

@pytest.mark.timeout(10)  # 10 second timeout for each test
def test_sqlite_manager_initialization(temp_db):
    """Test SQLiteManager initialization."""
    assert temp_db is not None
    assert hasattr(temp_db, 'connection')
    assert temp_db.connection is not None

@pytest.mark.timeout(10)
def test_table_creation(temp_db):
    """Test table creation and management."""
    # Test creating a table
    table_name = 'test_table'
    columns = {
        'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT',
        'value': 'REAL',
        'timestamp': 'TIMESTAMP'
    }
    temp_db.create_table(table_name, columns)
    
    # Verify table exists
    assert temp_db.table_exists(table_name)
    
    # Test creating table with invalid name
    with pytest.raises(ValueError):
        temp_db.create_table('', columns)

@pytest.mark.timeout(10)
def test_data_insertion(temp_db):
    """Test data insertion methods."""
    # Create test table
    table_name = 'test_data'
    columns = {
        'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT',
        'value': 'REAL',
        'timestamp': 'TIMESTAMP'
    }
    temp_db.create_table(table_name, columns)
    
    # Test inserting single row
    data = {
        'name': 'test',
        'value': 42.0,
        'timestamp': datetime.now()
    }
    temp_db.insert_data(table_name, data)
    
    # Test inserting multiple rows
    multiple_data = [
        {'name': 'test1', 'value': 1.0, 'timestamp': datetime.now()},
        {'name': 'test2', 'value': 2.0, 'timestamp': datetime.now()}
    ]
    temp_db.insert_many(table_name, multiple_data)
    
    # Verify data was inserted
    result = temp_db.query(f"SELECT COUNT(*) FROM {table_name}")
    assert result[0][0] == 3

@pytest.mark.timeout(10)
def test_data_querying(temp_db):
    """Test data querying methods."""
    # Create and populate test table
    table_name = 'test_query'
    columns = {
        'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT',
        'value': 'REAL',
        'timestamp': 'TIMESTAMP'
    }
    temp_db.create_table(table_name, columns)
    
    # Insert test data
    test_data = [
        {'name': 'test1', 'value': 1.0, 'timestamp': datetime.now()},
        {'name': 'test2', 'value': 2.0, 'timestamp': datetime.now()},
        {'name': 'test3', 'value': 3.0, 'timestamp': datetime.now()}
    ]
    temp_db.insert_many(table_name, test_data)
    
    # Test basic query
    result = temp_db.query(f"SELECT * FROM {table_name}")
    assert len(result) == 3
    
    # Test query with conditions
    result = temp_db.query(f"SELECT * FROM {table_name} WHERE value > 1.0")
    assert len(result) == 2
    
    # Test query to DataFrame
    df = temp_db.query_to_dataframe(f"SELECT * FROM {table_name}")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

@pytest.mark.timeout(10)
def test_data_updating(temp_db):
    """Test data updating methods."""
    # Create and populate test table
    table_name = 'test_update'
    columns = {
        'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT',
        'value': 'REAL'
    }
    temp_db.create_table(table_name, columns)
    
    # Insert test data
    data = {'name': 'test', 'value': 1.0}
    temp_db.insert_data(table_name, data)
    
    # Update data
    temp_db.update_data(table_name, {'value': 2.0}, {'name': 'test'})
    
    # Verify update
    result = temp_db.query(f"SELECT value FROM {table_name} WHERE name = 'test'")
    assert result[0][0] == 2.0

@pytest.mark.timeout(10)
def test_data_deletion(temp_db):
    """Test data deletion methods."""
    # Create and populate test table
    table_name = 'test_delete'
    columns = {
        'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT',
        'value': 'REAL'
    }
    temp_db.create_table(table_name, columns)
    
    # Insert test data
    test_data = [
        {'name': 'test1', 'value': 1.0},
        {'name': 'test2', 'value': 2.0},
        {'name': 'test3', 'value': 3.0}
    ]
    temp_db.insert_many(table_name, test_data)
    
    # Delete specific row
    temp_db.delete_data(table_name, {'name': 'test2'})
    
    # Verify deletion
    result = temp_db.query(f"SELECT COUNT(*) FROM {table_name}")
    assert result[0][0] == 2
    
    # Delete all data
    temp_db.delete_all(table_name)
    result = temp_db.query(f"SELECT COUNT(*) FROM {table_name}")
    assert result[0][0] == 0

@pytest.mark.timeout(10)
def test_transaction_handling(temp_db):
    """Test transaction handling."""
    # Create test table
    table_name = 'test_transaction'
    columns = {
        'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT',
        'value': 'REAL'
    }
    temp_db.create_table(table_name, columns)
    
    # Test successful transaction
    with temp_db.transaction():
        temp_db.insert_data(table_name, {'name': 'test1', 'value': 1.0})
        temp_db.insert_data(table_name, {'name': 'test2', 'value': 2.0})
    
    result = temp_db.query(f"SELECT COUNT(*) FROM {table_name}")
    assert result[0][0] == 2
    
    # Test failed transaction
    try:
        with patch('sqlite_manager.logger.error'), patch('builtins.print'):
            with temp_db.transaction():
                temp_db.insert_data(table_name, {'name': 'test3', 'value': 3.0})
                raise ValueError("Simulated error")
    except ValueError:
        pass
    
    result = temp_db.query(f"SELECT COUNT(*) FROM {table_name}")
    assert result[0][0] == 2  # Should still be 2 due to rollback

@pytest.mark.timeout(10)
def test_table_management(temp_db):
    """Test table management operations."""
    # Test creating and dropping table
    table_name = 'test_management'
    columns = {
        'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT'
    }
    
    temp_db.create_table(table_name, columns)
    assert temp_db.table_exists(table_name)
    
    temp_db.drop_table(table_name)
    assert not temp_db.table_exists(table_name)
    
    # Test getting table schema
    temp_db.create_table(table_name, columns)
    schema = temp_db.get_table_schema(table_name)
    # Check that schema contains the expected column names
    column_names = [col['name'] for col in schema]
    assert 'id' in column_names
    assert 'name' in column_names

@pytest.mark.timeout(10)
def test_data_validation(temp_db):
    """Test data validation methods."""
    # Create test table with constraints
    table_name = 'test_validation'
    columns = {
        'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT NOT NULL',
        'value': 'REAL CHECK (value > 0)'
    }
    temp_db.create_table(table_name, columns)
    
    # Test valid data
    valid_data = {'name': 'test', 'value': 1.0}
    temp_db.insert_data(table_name, valid_data)
    
    # Test invalid data - NULL name (this should be handled by the insert_data method)
    with patch('sqlite_manager.logger.error'):
        # The insert_data method should handle this gracefully
        result = temp_db.insert_data(table_name, {'name': None, 'value': 1.0})
        # Should return False or handle the error gracefully
        assert not result
    
    # Test invalid data - negative value (this should be handled by the insert_data method)
    with patch('sqlite_manager.logger.error'):
        result = temp_db.insert_data(table_name, {'name': 'test', 'value': -1.0})
        # Should return False or handle the error gracefully
        assert not result

@pytest.mark.timeout(15)  # Longer timeout for backup/restore
def test_backup_and_restore(temp_db, tmp_path):
    """Test database backup and restore functionality."""
    # Create and populate test table
    table_name = 'test_backup'
    columns = {
        'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT',
        'value': 'REAL'
    }
    temp_db.create_table(table_name, columns)
    
    # Insert test data
    test_data = [
        {'name': 'test1', 'value': 1.0},
        {'name': 'test2', 'value': 2.0}
    ]
    temp_db.insert_many(table_name, test_data)
    
    # Create backup
    backup_path = tmp_path / "backup.db"
    restored_path = tmp_path / "restored.db"
    
    try:
        # Create backup
        temp_db.backup_database(str(backup_path))
        assert backup_path.exists()
        
        # Create new database for restore
        new_db = SQLiteManager(str(restored_path))
        try:
            # Restore from backup
            new_db.restore_database(str(backup_path))
            
            # Verify restored data
            result = new_db.query(f"SELECT COUNT(*) FROM {table_name}")
            assert result[0][0] == 2
            
            # Verify data integrity
            original_data = temp_db.query(f"SELECT * FROM {table_name} ORDER BY id")
            restored_data = new_db.query(f"SELECT * FROM {table_name} ORDER BY id")
            assert original_data == restored_data
            
        finally:
            # Clean up restored database
            try:
                new_db.connection.close()
            except Exception:
                pass
            if restored_path.exists():
                try:
                    restored_path.unlink()
                except Exception:
                    pass
    finally:
        # Clean up backup file
        if backup_path.exists():
            try:
                backup_path.unlink()
            except Exception:
                pass

def test_connect_and_close(temp_db):
    conn = _connect(str(temp_db))
    assert isinstance(conn, sqlite3.Connection)
    _close(conn)
    # Closing twice should not raise
    _close(conn)

def test_execute_and_commit(temp_db):
    conn = _connect(str(temp_db))
    _execute(conn, "INSERT INTO test (value) VALUES (?)", ("abc",))
    _commit(conn)
    result = _fetchall(conn, "SELECT value FROM test")
    assert result[0][0] == "abc"
    _close(conn)

def test_fetchone_and_fetchall(temp_db):
    conn = _connect(str(temp_db))
    _execute(conn, "INSERT INTO test (value) VALUES (?)", ("abc",))
    _commit(conn)
    one = _fetchone(conn, "SELECT value FROM test")
    assert one[0] == "abc"
    all_ = _fetchall(conn, "SELECT value FROM test")
    assert all_[0][0] == "abc"
    _close(conn)

def test_create_and_drop_table(temp_db):
    conn = _connect(str(temp_db))
    _create_table(conn, "newtable", "id INTEGER PRIMARY KEY, value TEXT")
    _execute(conn, "INSERT INTO newtable (value) VALUES (?)", ("test",))
    _commit(conn)
    result = _fetchone(conn, "SELECT value FROM newtable")
    assert result[0] == "test"
    _drop_table(conn, "newtable")
    _commit(conn)
    with pytest.raises(sqlite3.OperationalError):
        _fetchone(conn, "SELECT value FROM newtable")
    _close(conn)

def test_insert_update_delete_select(sqlite_manager):
    # Create test table first
    sqlite_manager.create_table("test", {"id": "INTEGER PRIMARY KEY", "value": "TEXT"})
    
    # Insert
    sqlite_manager.insert("test", {"value": "foo"})
    rows = sqlite_manager.select("test", ["id", "value"])
    assert rows[0][1] == "foo"
    # Update
    sqlite_manager.update("test", {"value": "bar"}, where="value = 'foo'")
    rows = sqlite_manager.select("test", ["id", "value"])
    assert rows[0][1] == "bar"
    # Delete
    sqlite_manager.delete("test", where="value = 'bar'")
    rows = sqlite_manager.select("test", ["id", "value"])
    assert rows == []

def test_error_handling(temp_db):
    conn = _connect(str(temp_db))
    # Invalid SQL
    with pytest.raises(sqlite3.OperationalError):
        _execute(conn, "SELECT * FROM non_existent_table")
    # Invalid table for insert
    with pytest.raises(sqlite3.OperationalError):
        _execute(conn, "INSERT INTO non_existent_table (value) VALUES ('x')")
    _close(conn)

def test_sqlite_manager_methods(sqlite_manager):
    # Create test table first
    sqlite_manager.create_table("test", {"id": "INTEGER PRIMARY KEY", "value": "TEXT"})
    
    # Insert and select
    sqlite_manager.insert("test", {"value": "baz"})
    rows = sqlite_manager.select("test", ["id", "value"])
    assert rows[0][1] == "baz"
    # Update
    sqlite_manager.update("test", {"value": "qux"}, where="value = 'baz'")
    rows = sqlite_manager.select("test", ["id", "value"])
    assert rows[0][1] == "qux"
    # Delete
    sqlite_manager.delete("test", where="value = 'qux'")
    rows = sqlite_manager.select("test", ["id", "value"])
    assert rows == []
    # Drop table
    sqlite_manager.create_table("t2", {"id": "INTEGER PRIMARY KEY", "value": "TEXT"})
    sqlite_manager.drop_table("t2")
    with pytest.raises(sqlite3.OperationalError):
        sqlite_manager.select("t2", ["id", "value"]) 