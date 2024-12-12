import os
import sqlite3
import pandas as pd
from pathlib import Path
from config import DB_PATH

def create_connection(db_file):
    try:
        return sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(f"SQLite connection error: {e}")
        return None

def create_tables(conn):
    try:
        cursor = conn.cursor()
        tables = {
            'symbols': """
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE
                );""",
            'timeframes': """
                CREATE TABLE IF NOT EXISTS timeframes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timeframe TEXT UNIQUE
                );""",
            'indicators': """
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE
                );""",
            'klines': """
                CREATE TABLE IF NOT EXISTS klines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_id INTEGER,
                    timeframe_id INTEGER,
                    open_time TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    close_time TEXT,
                    quote_asset_volume REAL,
                    number_of_trades INTEGER,
                    taker_buy_base_asset_volume REAL,
                    taker_buy_quote_asset_volume REAL,
                    correlation_computed BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                    FOREIGN KEY (timeframe_id) REFERENCES timeframes(id)
                );""",
            'correlations': """
                CREATE TABLE IF NOT EXISTS correlations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_id INTEGER,
                    timeframe_id INTEGER,
                    indicator_id INTEGER,
                    lag INTEGER,
                    correlation_value REAL,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                    FOREIGN KEY (timeframe_id) REFERENCES timeframes(id),
                    FOREIGN KEY (indicator_id) REFERENCES indicators(id),
                    UNIQUE(symbol_id, timeframe_id, indicator_id, lag)
                );"""
        }
        for table, q in tables.items():
            cursor.execute(q)
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_open_time ON klines (open_time);",
            "CREATE INDEX IF NOT EXISTS idx_close_time ON klines (close_time);",
            "CREATE INDEX IF NOT EXISTS idx_correlation_computed ON klines (correlation_computed);",
            "CREATE INDEX IF NOT EXISTS idx_correlations ON correlations (symbol_id, timeframe_id, indicator_id, lag);"
        ]
        for idx_q in indexes:
            cursor.execute(idx_q)
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite table creation error: {e}")

def save_to_sqlite(df, db_path, symbol, timeframe):
    try:
        os.makedirs(Path(db_path).parent, exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Insert or ignore symbol
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        sym_id_row = cursor.fetchone()
        if not sym_id_row:
            print(f"Failed to retrieve symbol_id for symbol: {symbol}")
            return
        sym_id = sym_id_row[0]
        # Insert or ignore timeframe
        cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", (timeframe,))
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        tf_id_row = cursor.fetchone()
        if not tf_id_row:
            print(f"Failed to retrieve timeframe_id for timeframe: {timeframe}")
            return
        tf_id = tf_id_row[0]
        # Format datetime columns as strings
        df['open_time'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['close_time'] = df['close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        insert_q = """
            INSERT INTO klines (
                symbol_id, timeframe_id, open_time, open, high, low, close, volume, close_time, 
                quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data_to_insert = [
            (
                sym_id,
                tf_id,
                row['open_time'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
                row['close_time'],
                row['quote_asset_volume'],
                row['number_of_trades'],
                row['taker_buy_base_asset_volume'],
                row['taker_buy_quote_asset_volume']
            )
            for index, row in df.iterrows()
        ]
        cursor.executemany(insert_q, data_to_insert)
        conn.commit()
        print(f"Successfully inserted {len(data_to_insert)} records into 'klines' table.")
    except sqlite3.Error as e:
        print(f"SQLite insertion error: {e}")
    except Exception as ex:
        print(f"Unexpected error during SQLite insertion: {ex}")
    finally:
        if conn:
            conn.close()

def insert_indicator(conn, indicator_name):
    """Insert a new indicator into the 'indicators' table if it doesn't already exist."""
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (indicator_name,))
    conn.commit()

def get_indicator_id(conn, indicator_name):
    """Retrieve the ID of an indicator from the 'indicators' table."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
    result = cursor.fetchone()
    return result[0] if result else None

def insert_correlation(conn, symbol_id, timeframe_id, indicator_id, lag, correlation_value):
    """Insert or update a correlation value into the 'correlations' table."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO correlations (symbol_id, timeframe_id, indicator_id, lag, correlation_value)
        VALUES (?, ?, ?, ?, ?)
    """, (symbol_id, timeframe_id, indicator_id, lag, correlation_value))
    conn.commit()