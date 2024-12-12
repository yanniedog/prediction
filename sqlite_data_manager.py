# filename: sqlite_data_manager.py
import sqlite3
from pathlib import Path

def create_connection(db_path):
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def create_tables(conn):
    try:
        cursor = conn.cursor()

        # Create the indicators table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            )
            """
        )

        # Create the klines table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS klines (
                id INTEGER PRIMARY KEY,
                symbol_id INTEGER NOT NULL,
                timeframe_id INTEGER NOT NULL,
                open_time DATETIME NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                FOREIGN KEY (timeframe_id) REFERENCES timeframes(id)
            )
            """
        )

        # Create the symbols table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                symbol TEXT UNIQUE NOT NULL
            )
            """
        )

        # Create the timeframes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS timeframes (
                id INTEGER PRIMARY KEY,
                timeframe TEXT UNIQUE NOT NULL
            )
            """
        )

        # Create the correlations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS correlations (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                indicator TEXT NOT NULL,
                lag INTEGER NOT NULL,
                correlation REAL NOT NULL,
                FOREIGN KEY (indicator) REFERENCES indicators(name)
            )
            """
        )

        conn.commit()
    except sqlite3.Error as e:
        print(f"Error creating tables: {e}")

def initialize_database(db_path):
    conn = create_connection(db_path)
    if conn:
        create_tables(conn)
        conn.close()

if __name__ == "__main__":
    DB_PATH = "correlation_database.db"
    db_file = Path(DB_PATH)

    if not db_file.exists():
        print("Database not found. Creating a new one...")

    initialize_database(DB_PATH)
    print("Database initialized successfully.")
