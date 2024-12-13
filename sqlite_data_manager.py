# sqlite_data_manager.py
import json
import logging
import sqlite3
from pathlib import Path
import pandas as pd
import re
from config import DB_PATH

logger = logging.getLogger()

def create_connection(db_path=DB_PATH):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logger.info(f"Connected to SQLite database at {db_path}. SQLite version: {cursor.execute('SELECT sqlite_version();').fetchone()[0]}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"SQLite connection error: {e}")
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
                    indicator_config_id INTEGER,
                    lag INTEGER,
                    correlation_value REAL,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                    FOREIGN KEY (timeframe_id) REFERENCES timeframes(id),
                    FOREIGN KEY (indicator_config_id) REFERENCES indicator_configs(id),
                    UNIQUE(symbol_id, timeframe_id, indicator_config_id, lag)
                );"""
        }

        for table_name, ddl in tables.items():
            cursor.execute(ddl)
            logger.info(f"Table '{table_name}' ensured in database.")

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_open_time ON klines (open_time);",
            "CREATE INDEX IF NOT EXISTS idx_close_time ON klines (close_time);",
            "CREATE INDEX IF NOT EXISTS idx_correlation_computed ON klines (correlation_computed);",
            "CREATE INDEX IF NOT EXISTS idx_correlations ON correlations (symbol_id, timeframe_id, indicator_config_id, lag);"
        ]

        for idx_query in indexes:
            cursor.execute(idx_query)
            index_name = re.findall(r'CREATE INDEX IF NOT EXISTS (\w+) ', idx_query)[0]
            logger.info(f"Index created or already exists: {index_name}")

        conn.commit()
        logger.info("All tables and indexes are set up successfully.")
    except sqlite3.Error as e:
        logger.error(f"SQLite table creation error: {e}")
        raise e

def initialize_database(db_path=DB_PATH):
    conn = create_connection(db_path)
    if conn:
        create_tables(conn)
        conn.close()
    else:
        logger.error("Failed to initialize the database.")

def insert_indicator_configs(conn, indicator_name, configs):
    try:
        create_tables(conn)  # Ensure tables are created
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (indicator_name,))
        conn.commit()
        cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Indicator '{indicator_name}' could not be inserted.")
        indicator_id = result[0]
        for config in configs:
            config_json = json.dumps(config, sort_keys=True)
            cursor.execute("INSERT OR IGNORE INTO indicator_configs (indicator_id, config) VALUES (?, ?)", (indicator_id, config_json))
        conn.commit()
        logger.info(f"Inserted {len(configs)} configurations for indicator '{indicator_name}'.")
    except sqlite3.Error as e:
        logger.error(f"SQLite insertion error: {e}")
        raise e
    except ValueError as ve:
        logger.error(ve)
        raise ve

def fetch_indicator_configs(conn, indicator_name):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT config FROM indicator_configs 
        JOIN indicators ON indicator_configs.indicator_id = indicators.id 
        WHERE indicators.name = ?
    """, (indicator_name,))
    rows = cursor.fetchall()
    return [json.loads(row[0]) for row in rows]

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
        logger.info(f"Successfully inserted {len(data)} records into 'klines'.")
    except (sqlite3.Error, ValueError) as e:
        logger.error(f"SQLite insertion error: {e}")
        raise e

def save_to_sqlite(df, db_path, symbol, timeframe):
    conn = create_connection(db_path)
    if conn:
        try:
            insert_klines(conn, df, symbol, timeframe)
        except Exception as e:
            logger.error(f"Error saving klines to SQLite: {e}")
            raise e
        finally:
            conn.close()
    else:
        logger.error("Cannot connect to the database.")
        raise ConnectionError("Cannot connect to the database.")
