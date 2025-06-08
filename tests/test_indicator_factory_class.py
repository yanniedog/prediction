import pytest
from indicator_factory_class import SQLiteManager

def test_sqlite_manager_init(tmp_path):
    db_path = tmp_path / "test.db"
    manager = SQLiteManager(str(db_path))
    assert manager.db_path == str(db_path)
    assert manager.connection is not None
    manager.close()
    assert manager.connection is None

def test_sqlite_manager_create_and_drop_table(tmp_path):
    db_path = tmp_path / "test.db"
    manager = SQLiteManager(str(db_path))
    table_name = "test_table"
    columns = {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}
    manager.create_table(table_name, columns)
    assert manager.table_exists(table_name)
    manager.drop_table(table_name)
    assert not manager.table_exists(table_name)
    manager.close() 