# sqlite_data_manager.py

import json
import logging
import sqlite3
from pathlib import Path
import pandas as pd
from config import DB_PATH

logger = logging.getLogger()

def create_connection(db_path=DB_PATH):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sqlite_version();")
        version = cursor.fetchone()[0]
        print(f"Connected to SQLite database at {db_path}. SQLite version: {version}")
        return conn
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
            'indicator_configs': """
                CREATE TABLE IF NOT EXISTS indicator_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator_id INTEGER,
                    config TEXT NOT NULL,
                    FOREIGN KEY (indicator_id) REFERENCES indicators(id),
                    UNIQUE(indicator_id, config)
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

        for table_name, ddl in tables.items():
            cursor.execute(ddl)
            print(f"Table '{table_name}' ensured in database.")

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_open_time ON klines (open_time);",
            "CREATE INDEX IF NOT EXISTS idx_close_time ON klines (close_time);",
            "CREATE INDEX IF NOT EXISTS idx_correlation_computed ON klines (correlation_computed);",
            "CREATE INDEX IF NOT EXISTS idx_correlations ON correlations (symbol_id, timeframe_id, indicator_id, lag);"
        ]

        for idx_query in indexes:
            cursor.execute(idx_query)
            index_name = idx_query.split()[5]
            print(f"Index created or already exists: {index_name}")

        conn.commit()
        print("All tables and indexes are set up successfully.")
    except sqlite3.Error as e:
        print(f"SQLite table creation error: {e}")

def initialize_database(db_path=DB_PATH):
    conn = create_connection(db_path)
    if conn:
        create_tables(conn)
        conn.close()
    else:
        print("Failed to initialize the database.")

def insert_indicator_configs(conn, indicator_name, configs):
    """
    Inserts the base indicator and its configurations into the database.

    Args:
        conn (sqlite3.Connection): Database connection.
        indicator_name (str): Name of the indicator.
        configs (List[Dict]): List of configuration dictionaries.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (indicator_name,))
        conn.commit()
        cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
        indicator_id = cursor.fetchone()[0]

        for config in configs:
            config_json = json.dumps(config)
            cursor.execute("INSERT OR IGNORE INTO indicator_configs (indicator_id, config) VALUES (?, ?)", (indicator_id, config_json))
        conn.commit()
        print(f"Inserted {len(configs)} configurations for indicator '{indicator_name}'.")
    except sqlite3.Error as e:
        print(f"SQLite insertion error: {e}")

def insert_klines(conn, df, symbol, timeframe):
    try:
        required_columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns: {set(required_columns) - set(df.columns)}")

        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
        cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
        symbol_id = cursor.fetchone()[0]

        cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", (timeframe,))
        cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
        timeframe_id = cursor.fetchone()[0]

        df['open_time'] = pd.to_datetime(df['open_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['close_time'] = pd.to_datetime(df['close_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

        insert_query = """
            INSERT INTO klines (
                symbol_id, timeframe_id, open_time, open, high, low, close, volume, close_time,
                quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        data = [
            (
                symbol_id, timeframe_id,
                row['open_time'], row['open'], row['high'], row['low'], row['close'], row['volume'], row['close_time'],
                row['quote_asset_volume'], row['number_of_trades'],
                row['taker_buy_base_asset_volume'], row['taker_buy_quote_asset_volume']
            )
            for _, row in df.iterrows()
        ]

        cursor.executemany(insert_query, data)
        conn.commit()
        print(f"Successfully inserted {len(data)} records into 'klines'.")
    except (sqlite3.Error, ValueError) as e:
        print(f"SQLite insertion error: {e}")

def save_to_sqlite(df, db_path, symbol, timeframe):
    conn = create_connection(db_path)
    if conn:
        insert_klines(conn, df, symbol, timeframe)
        conn.close()
    else:
        print("Cannot connect to the database.")
