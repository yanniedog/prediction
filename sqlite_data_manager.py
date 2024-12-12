# sqlite_data_manager.py
import sqlite3
import itertools

def create_tables(conn):
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
    for table_query in tables.values():
        cursor.execute(table_query)
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_open_time ON klines (open_time);",
        "CREATE INDEX IF NOT EXISTS idx_close_time ON klines (close_time);",
        "CREATE INDEX IF NOT EXISTS idx_correlation_computed ON klines (correlation_computed);",
        "CREATE INDEX IF NOT EXISTS idx_correlations ON correlations (symbol_id, timeframe_id, indicator_id, lag);"
    ]
    for index_query in indexes:
        cursor.execute(index_query)
    conn.commit()

def insert_indicator_configs(conn, indicator, configs):
    cursor = conn.cursor()
    for config in configs:
        config_name = f"{indicator}_" + "_".join([f"{k}{v}" for k, v in config.items()])
        cursor.execute("INSERT OR IGNORE INTO indicators (name) VALUES (?)", (config_name,))
    conn.commit()

def list_indicators(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM indicators")
    return sorted([row[0] for row in cursor.fetchall()])

def insert_symbol(conn, symbol):
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
    conn.commit()

def insert_timeframe(conn, timeframe):
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)", (timeframe,))
    conn.commit()

def get_symbol_id(conn, symbol):
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()
    return result[0] if result else None

def get_timeframe_id(conn, timeframe):
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
    result = cursor.fetchone()
    return result[0] if result else None

def insert_klines(conn, symbol_id, timeframe_id, klines):
    cursor = conn.cursor()
    query = """
        INSERT INTO klines (
            symbol_id, timeframe_id, open_time, open, high, low, close, volume, close_time, 
            quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    cursor.executemany(query, [
        (
            symbol_id, timeframe_id, row['open_time'], row['open'], row['high'], row['low'], row['close'],
            row['volume'], row['close_time'], row['quote_asset_volume'], row['number_of_trades'],
            row['taker_buy_base_asset_volume'], row['taker_buy_quote_asset_volume']
        ) for _, row in klines.iterrows()
    ])
    conn.commit()

def insert_correlation(conn, symbol_id, timeframe_id, indicator_id, lag, correlation_value):
    cursor = conn.cursor()
    query = """
        INSERT OR IGNORE INTO correlations (
            symbol_id, timeframe_id, indicator_id, lag, correlation_value
        ) VALUES (?, ?, ?, ?, ?)"""
    cursor.execute(query, (symbol_id, timeframe_id, indicator_id, lag, correlation_value))
    conn.commit()

def get_correlation(conn, symbol_id, timeframe_id, indicator_id, lag):
    cursor = conn.cursor()
    query = """
        SELECT correlation_value FROM correlations
        WHERE symbol_id = ? AND timeframe_id = ? AND indicator_id = ? AND lag = ?"""
    cursor.execute(query, (symbol_id, timeframe_id, indicator_id, lag))
    result = cursor.fetchone()
    return result[0] if result else None
